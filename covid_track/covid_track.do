*ECON_398 DATA Project
/*Files that I need in master folder
TG4
TG8
TB4
TB8
public_schools.csv
acs_2017
acs_2018
acs_2019
acs_2021
cdc.csv
*/

**# Race 1
*Convert raw race data into stata format
cd "C:\Users\msc19\Box\ECON_398\Data_Project\master"
import delimited "C:\Users\msc19\Box\ECON_398\Data_Project\master\TG4.csv"

drop unnamed0 
drop v1
save "G4.dta", replace
clear all

import delimited "C:\Users\msc19\Box\ECON_398\Data_Project\master\TG8.csv"
save "G8.dta", replace
clear all

import delimited "C:\Users\msc19\Box\ECON_398\Data_Project\master\TB4.csv"
save "B4.dta", replace
clear all

import delimited "C:\Users\msc19\Box\ECON_398\Data_Project\master\TB8.csv"
save "B8.dta", replace
clear all

*append raw race data into complete race data
use B4.dta
append using B8.dta
save B.dta, replace
clear

use G4.dta
append using G8.dta
save G.dta, replace

use B.dta
append using G.dta
save race.dta, replace
export delimited "race.csv", replace
clear all

*clean unnatached and duplicate observations, save as race_2
use race.dta
drop if team=="Unnatached"
sort athlete team meet year
quietly by athlete team meet year: gen dup= cond(_N==1,0,_n)
tabulate dup
drop if dup>1
drop dup
order year state team athlete mark event male meet, first
sort athlete team event
quietly by athlete team event: gen dup = cond(_N==1,0,_n)
tab dup
drop dup
save race_2.dta, replace
export delimited "race_2.csv", replace
clear all

**# Race Times to Seconds
use race_2
gen double mark_ = clock(substr(mark, 1, 8), "ms") 

gen countvar = strlen(mark)
gen count_dummy = 1 if countvar>5
drop if count_dummy==1
save final_race_under60, replace
export delimited "final_race_under60.csv", replace
clear all

import delimited "race_2.csv"
gen countvar = strlen(mark)
gen count_dummy = 1 if countvar>5
drop if count_dummy==.
save final_race_over60, replace
export delimited "final_race_over60.csv", replace

use final_race_under60
destring mark, generate(seconds) 
drop mark_ countvar count_dummy
save final_race_u60, replace
clear all

use final_race_over60

generate mark_min = substr(mark,1,1)
generate mark_sec = substr(mark,3,2)
gen mark_milisec = substr(mark,6,2)

destring mark_min, gen(mark_min_)
destring mark_sec, gen(mark_sec_) force
destring mark_milisec, gen(mark_milisec_) force

gen seconds= ((60*(mark_min_)) + (mark_sec_)+ (.01*(mark_milisec_)))

drop countvar count_dummy mark_min mark_sec mark_sec_ mark_milisec mark_milisec_
save final_race_o60, replace
clear all

use final_race_u60
append using final_race_o60
sort state team athlete year event
order seconds, after(mark)
save race_3, replace
export delimited "race_3.csv", replace

clear all

**# finalizing race
use race_3
*gnerate athlete code and prep data for calculations
encode state, generate(state_)
order state, first
drop if state== "NB"
drop if state=="OS"
order state_, first

gen event_=1 if event==400
replace event_=0 if event_==.
order event_, after(event)


destring grade, generate(grade_) force
order grade_, after(grade)
replace grade_=0 if grade_==.
drop if grade_<9

tab grade_
drop grade
rename grade_ grade

encode team, generate(team_)
order team_, after(team)
order team, last
replace team= lower(team)

egen athlete_id_=concat(state team_ athlete event_)
egen athlete_id=group(athlete_id_)
drop athlete_id_
order athlete_id, after(team_)
order year, after(grade)
order seconds, after(event_)


*clean duplicates
bysort athlete_id: gen ath_freq=_N
drop if ath_freq>4
gsort -ath_freq +athlete_id +year
bysort athlete_id year : gen ath_year_freq=_N
drop if ath_year_freq>1
gsort -ath_freq +state +athlete_id +year
xtset athlete_id year, yearly

*generate statistical improvement variables

bysort athlete_id: generate ath_abs_change=seconds-l.seconds
gen improv= -1*(ath_abs_change)
bysort male event_ state year: egen state_mean_improv= mean(improv)
gen ln_state_mean_improv = log(state_mean_improv)
bysort male event_ team year: egen team_mean_improv= mean(improv)
gen ln_team_mean_improv= log(team_mean_improv)
bysort male event_ state year: egen state_mean_seconds= mean(seconds)
bysort male event_ team year: egen team_mean_seconds= mean(seconds)

drop ath_freq ath_year_freq ath_abs_change
save race_4, replace
export delimited race_4.csv, replace

*collapse race data by school
collapse improv state_mean_improv ln_state_mean_improv team_mean_improv ln_team_mean_improv team_mean_seconds state_mean_seconds, by(state team year event male)

*save progress
save race_5, replace
export delimited "race_5.csv", replace
clear all
**# School Data Cleaning

*This is save in multiple places for me
use public_schools.dta
drop x y telephone latitude longitude naics_code naics_desc source sourcedate val_method val_date website start_grade end_grade  shelter_id country

*Cleaned public school data so that I include High Schools that have enrollment, that are regular schools, and are currently in session. Generate Binary variable for public schools
encode level_, generate(level_2)
drop if level_2!=3
drop fid level_ level_2 objectid_public objectid_private
drop if enrollment<0
drop if type!=1
drop if status!=1
drop type status 
gen public=1
gen ncesid_private= "null"
order ncesid_private, after(ncesid)
order state county name zip, first
egen name_=group(name)

save schools, replace
export delimited "schools.csv", replace
clear all

**# merge school to race
import delimited "schools.csv"

gen team_= name
gen team_1=team_
replace team_1= lower(team_1)
local x `""high" "school" "high school" "middle school" "jshs" "hs" "js" "jr" "sr" "jr." "sr." "senior" "academy" "h s" "shs" "jr-sr" "jr/sr" "sch" "junior/" " -" "college" "." " " "  " "./." "middle" "secondary" "/ " "/" "community" "county" " " " " " " " " "-" "(" ")" "[0-9]" "." "collegiate" "university" "technology" "preparatory" "center" "sci" "'
foreach case of local x {
	replace team_1 = regexr(team_1,"`case'","") if (strpos(team_1, "`case'")>1)
}
encode team_1, generate(team_2)


save schools_2, replace
export delimited "schools_2.csv", replace
clear all

import delimited "race_5.csv"
gen team_1=lower(team)
local x `" " " " " " " " " " " " " " " " " " " " " " " " " "academy" "-"  "'
foreach case of local x {
	replace team_1= regexr(team_1, "`case'", "") if (strpos(team_1, "`case'")>1)
}

*15,662 schools in race 
merge m:m state team_1 using schools_2
encode state, generate(state_)
drop if state_==.
gsort -_merge +state_ +team +male +event +year
save race_6, replace
export delimited "race_6.csv", replace
clear all

**# Clean ACS
*append all years data into one dataset
import delimited "acs_2017.csv", varnames(1) 
gen year=2017
drop if geo_id=="Geography"
keep geo_id name year dp03_0062e dp03_0120pe
destring dp03_0062e dp03_0120pe, force replace
save acs_2017, replace
clear all

import delimited "acs_2018.csv", varnames(1) 
gen year=2018
drop if geo_id=="Geography"
keep geo_id name year dp03_0062e dp03_0120pe
destring dp03_0062e dp03_0120pe, force replace
save acs_2018, replace
clear all

import delimited "acs_2019.csv", varnames(1) 
gen year= 2019
drop if geo_id=="Geography"
keep geo_id name year dp03_0062e dp03_0120pe
destring dp03_0062e dp03_0120pe, force replace
save acs_2019, replace
clear all

import delimited "acs_2021.csv", varnames(1) 
gen year=2021
drop if geo_id=="Geography"
keep geo_id name year dp03_0062e dp03_0120pe
destring dp03_0062e dp03_0120pe, force replace
save acs_2021, replace
clear all

*create an average between 2019-2021 for 2020 and then 
use acs_2019
append using acs_2021
encode geo_id, generate(geo_id_)
encode name, generate(name_)
destring dp03_0062e dp03_0120pe, force replace
collapse dp03_0062e dp03_0120pe, by(geo_id_ name_ geo_id name)
gen year= 2020
drop geo_id_ name_ 
save acs_2020, replace
export delimited "acs_2020.csv", replace
clear all

use acs_2017
append using acs_2018
append using acs_2019
append using acs_2020
append using acs_2021
save acs, replace
export delimited "acs.csv", replace

*prep dataset to merge
gen countyfips= substr(geo_id,10,5)
replace countyfips= substr(countyfips,2,4) if substr(countyfips,1,1)=="0"
destring countyfips, replace
save acs_2, replace 
export delimited "acs_2.csv", replace
clear all

use acs_2
order year countyfips name geo_id, first
sort countyfips year
keep geo_id name year countyfips dp03_0062e dp03_0120pe
gen county = substr(name, 1, strpos(name, ",") - 1) 
replace county = name if missing(county) 
gen state = substr(name, strpos(name, ",") +1, strlen(name))
replace state = name if missing(state)
replace county= lower(county)
replace state= lower(state)
order state county, after(countyfips)
encode state, generate(state_)
destring dp03_0062e dp03_0120pe, force replace
order state_, after(state)
*rename all variables
rename dp03_0062e med_house_inc
rename dp03_0120pe pct_pov_u18
*create state wide averages
bysort state_ year: egen st_med_house_inc=mean(med_house_inc)
bysort state_ year: egen st_pct_pov_u18= mean(pct_pov_u18)
save acs_3, replace
export delimited "acs_3.csv", replace
clear all

*collapse data set by state county year
use acs_3 
collapse med_house_inc pct_pov_u18 st_med_house_inc st_pct_pov_u18, by(state_ countyfips county year)
gen countyfips_=countyfips
save acs_4, replace
export delimited "acs_4.csv", replace
clear all

*merge race_6 to acs_2 
use race_6
rename _merge merge_1
merge m:m countyfips year using acs_4
rename _merge merge_2
save race_7, replace
export delimited "race_7.csv", replace
clear all

**#Clean CDC
import delimited "cdc.csv"
*generate day, month, and year variables
gen month = substr(date, 1, strpos(date, "/") - 1) 
replace month = date if missing(month) 

gen year = substr(date, -strpos(date, "/")-2, strlen(date))
replace year = date if missing(year)
replace year = substr(year,2,strlen(year)-1) if strpos(year,"/")==1
save cdc_2, replace
export delimited "cdc_2.csv", replace
clear all

use cdc_2
destring month year, replace
drop if month>6 
drop if year==2021
rename county_name county
rename state_tribe_territory state
encode county, generate(county_)
encode state, generate(state_)
bysort county_ year: gen cdc= 8-order_code
rename cdc order


collapse order, by(fips_state fips_county county state county_ state_ )
save cdc_3, replace
export delimited "cdc_3.csv", replace
clear all
**# Merge cdc_3 to race_7

use cdc_3
tostring fips_state, generate(fips_state_)
tostring fips_county, generate(fips_county_)
gen county_fips= fips_state_ + "0" + "0" + fips_county_ if strlen(fips_state_)==1 & strlen(fips_county_)==1
replace county_fips= fips_state_ + "0"  + fips_county_ if strlen(fips_state_)==1 & strlen(fips_county_)==2
replace county_fips= fips_state_ + fips_county_ if strlen(fips_state_)==1 & strlen(fips_county_)==3
replace county_fips= fips_state_ + "0" + fips_county_ if strlen(fips_state_)==2 & strlen(fips_county_)==2
replace county_fips= fips_state_ + "0" + "0"+ fips_county_ if strlen(fips_state_)==2 & strlen(fips_county_)==1
replace county_fips= fips_state_ + fips_county_ if strlen(fips_state_)==2 & strlen(fips_county_)==3
destring county_fips, generate(countyfips)
drop fips_state fips_state_ fips_county fips_county_ 
save cdc_4, replace
export delimited "cdc_4.csv", replace
clear all

use race_7
merge m:m countyfips using cdc_4
rename _merge merge_3
save race_8, replace
export delimited "race_8.csv", replace

*order and sort Data for final analysis
order state countyfips team event male year team_mean_seconds med_house_inc order, first

gsort +state +countyfips +team +event +male +year -merge_1 -merge_2 -merge_3 
replace pct_pov_u18 = pct_pov_u18[_n-1] if year==2022
replace med_house_inc = med_house_inc[_n-1] if year==2022
drop if missing(team)
drop if countyfips==.
drop if missing(team)
drop if med_house_inc==.
gen event_=1 if event==400
replace event_=0 if event_==.
order event_, after(event)
tostring event_, replace
tostring male, generate(male_)
gen obs_id= team + event_ + male_ + county_fips
encode obs_id, generate(obs_id_)
bysort team_2 event male county_fips year: gen dup= cond(_N==1,0,_n)
tabulate dup
drop if dup>0
drop dup
gen treat=1 if year>2019
replace treat=0 if treat==.
gen treat_order=treat*order
replace treat_order=0 if treat_order==.


*finilize covid order_code
xtset obs_id_ year
gsort +state +countyfips +team +event +male +year -merge_1 -merge_2 -merge_3 
destring event_, replace
*omit female due to data collection error
drop if male==0
gen ln_med_house_inc= log(med_house_inc)

*run regressions
xtreg team_mean_seconds treat order treat_order ln_med_house_inc i.state_ i.year if event_==1

xtreg team_mean_seconds treat order treat_order ln_med_house_inc i.state_ i.year if event_==0

xtreg team_mean_seconds treat order treat_order pct_pov_u18  i.state_ i.year if event_==1

xtreg team_mean_seconds treat order treat_order pct_pov_u18  i.state_ i.year if event_==0

save race_9, replace
export delimited "race_9.csv", replace

*create visualisations:

twoway scatter team_mean_seconds order if year==2020 & event_==1|| lfit team_mean_seconds order if year==2020 & event_==1, title(Boys 400m)


twoway scatter team_mean_seconds order if year==2020 & event_==0|| lfit team_mean_seconds order if year==2020 & event_==0

bysort event_ year: egen mean = mean(team_mean_seconds)

twoway scatter mean year if event_==1|| lfit mean year if event_==1
twoway scatter mean year if event_==0|| lfit mean year if event_==0

 

















