/*********************** ECON 388 Data Project 3 *******************************

NAME:McGyver Clark	
Assignment: Data Project 3

*******************************************************************************/
cd "C:\Users\msc19\Box\Econ_388\Data_Project_3"
clear 
capture log close

log using Data_Project_3.txt, replace text

********************************************************************
*Import first data file
import delimited county_allstdprice_2003_10_LABELS_2021-02-09.csv
save 2003-2010.dta, replace
clear

*Import Second data file
import delimited county_stdprices_ffs.csv
save 2011-2019.dta, replace

*append the two data files together
append using 2003-2010.dta
save full_prices.dta, replace
sort year geo_code

*Generate treated variable
gen CBP=.
replace CBP=1 if geo_code==6065
replace CBP=1 if geo_code==6071
replace CBP=1 if geo_code==12001
replace CBP=1 if geo_code==12069
replace CBP=1 if geo_code==12086
replace CBP=1 if geo_code==12095
replace CBP=1 if geo_code==12097 
replace CBP=1 if geo_code==12099
replace CBP=1 if geo_code==12117
replace CBP=1 if geo_code==18029
replace CBP=1 if geo_code==18047
replace CBP=1 if geo_code==18115
replace CBP=1 if geo_code==18161
replace CBP=1 if geo_code==20091
replace CBP=1 if geo_code==20103
replace CBP=1 if geo_code==20107
replace CBP=1 if geo_code==20121
replace CBP=1 if geo_code==20209
replace CBP=1 if geo_code==21015
replace CBP=1 if geo_code==21023
replace CBP=1 if geo_code==21037
replace CBP=1 if geo_code==21077
replace CBP=1 if geo_code==21081
replace CBP=1 if geo_code==21117
replace CBP=1 if geo_code==21191
replace CBP=1 if geo_code==29013
replace CBP=1 if geo_code==29025
replace CBP=1 if geo_code==29037
replace CBP=1 if geo_code==29047
replace CBP=1 if geo_code==29049
replace CBP=1 if geo_code==29095
replace CBP=1 if geo_code==29107
replace CBP=1 if geo_code==29165
replace CBP=1 if geo_code==29177
replace CBP=1 if geo_code==37007
replace CBP=1 if geo_code==37025
replace CBP=1 if geo_code==37071
replace CBP=1 if geo_code==37109
replace CBP=1 if geo_code==37119
replace CBP=1 if geo_code==37159
replace CBP=1 if geo_code==37179
replace CBP=1 if geo_code==39015
replace CBP=1 if geo_code==39017
replace CBP=1 if geo_code==39025
replace CBP=1 if geo_code==39035
replace CBP=1 if geo_code==39055
replace CBP=1 if geo_code==39061
replace CBP=1 if geo_code==39085
replace CBP=1 if geo_code==39093
replace CBP=1 if geo_code==39103
replace CBP=1 if geo_code==39165
replace CBP=1 if geo_code==42003
replace CBP=1 if geo_code==42005
replace CBP=1 if geo_code==42007
replace CBP=1 if geo_code==42019
replace CBP=1 if geo_code==42051
replace CBP=1 if geo_code==42125
replace CBP=1 if geo_code==42129
replace CBP=1 if geo_code==45023
replace CBP=1 if geo_code==45057
replace CBP=1 if geo_code==45091
replace CBP=1 if geo_code==48085
replace CBP=1 if geo_code==48113
replace CBP=1 if geo_code==48121
replace CBP=1 if geo_code==48139
replace CBP=1 if geo_code==48231
replace CBP=1 if geo_code==48251
replace CBP=1 if geo_code==48257
replace CBP=1 if geo_code==48367
replace CBP=1 if geo_code==48397
replace CBP=1 if geo_code==48439
replace CBP=1 if geo_code==48497
replace CBP=0 if CBP==.

*Clean data by ommiting outlier rates and focusing 
drop if adj==-99999

*Recode event_label with a value by creating a new variable 
gen event_value=1 if event_label== "Durable Medical Equipment Reimbursements per Enrollee"
replace event_value=2 if event_label== "Home Health Agency Reimbursements per Enrollee"
replace event_value=3 if event_label== "Hospice Reimbursements per Enrollee"
replace event_value=4 if event_label== "Hospital & Skilled Nursing Facility Reimbursements per Enrollee"
replace event_value=5 if event_label==  "Outpatient Facility Reimbursements per Enrollee"
replace event_value=6 if event_label== "Physician Reimbursements per Enrollee"
replace event_value=7 if event_label== "Total Medicare Reimbursements per Enrollee (Parts A and B)"
gen adj_event_value=8 if event_label== "Price-Adjusted Durable Medical Equipment Reimbursements per Enrollee"
replace adj_event_value=9 if event_label== "Price-Adjusted Home Health Agency Reimbursements per Enrollee"
replace adj_event_value=10 if event_label== "Price-Adjusted Hospice Reimbursements per Enrollee"
replace adj_event_value=11 if event_label== "Price-Adjusted Hospital & Skilled Nursing Facility Reimbursements per Enrollee"
replace adj_event_value=12 if event_label== "Price-Adjusted Outpatient Facility Reimbursements per Enrollee"
replace adj_event_value=13 if event_label== "Price-Adjusted Physician Reimbursements per Enrollee"
replace adj_event_value=14 if event_label== "Price-Adjusted Total Medicare Reimbursements per Enrollee (Parts A and B)"


*add labels to event_label
#delimit ;

;
label values event_value  event_value;
label define event_value 
	1           "Durable Medical Equipment Reimbursements per Enrollee"           
	2           "Home Health Agency Reimbursements per Enrollee"      
	3           "Hospice Reimbursements per Enrollee"
	4           "Hospital & Skilled Nursing Facility Reimbursements per Enrollee"       
	5           "Outpatient Facility Reimbursements per Enrollee"
	6           "Physician Reimbursements per Enrollee"
	7          "Total Medicare Reimbursements per Enrollee (Parts A and B)"
;
label values adj_event_value adj_event_value;
label define adj_event_value
	8           "Price-Adjusted Durable Medical Equipment Reimbursements per Enrollee"
	9           "Price-Adjusted Home Health Agency Reimbursements per Enrollee"
	10           "Price-Adjusted Hospice Reimbursements per Enrollee"
	11          "Price-Adjusted Hospital & Skilled Nursing Facility Reimbursements per Enrollee"
	12          "Price-Adjusted Outpatient Facility Reimbursements per Enrollee"
	13          "Price-Adjusted Physician Reimbursements per Enrollee"
	14          "Price-Adjusted Total Medicare Reimbursements per Enrollee (Parts A and B)"
;
	
#delimit cr

*Analyze categories of reimbursements
preserve
drop if adj_event_value==14 
collapse (mean) adjusted_rate, by(adj_event_value year)
xtset adj_event_value year
xtline adjusted_rate, overlay title("Average Annual Medicare Reimbursemnts by Category") ytitle(Average Annual Rate) scheme(538bw) legend(ring(3) pos(5) col(1)) name (Figure_1, replace)
graph save "Figure_1" "C:\Users\msc19\Box\ECON_388\Data_Project_3\Figure_1.gph", replace
graph export Figure_1.gph, as(jpg) replace
restore

*Focus data on DME
keep if event_label== "Price-Adjusted Durable Medical Equipment Reimbursements per Enrollee"

*Create post and interaction variables to run difference in difference regression
gen post=1 if year>=2011
replace post=0 if post==.
gen CBP_post= CBP*post
bysort year: egen avg_adj_t= mean(adjusted_rate) if CBP==1
bysort year: egen avg_adj_n= mean(adjusted_rate) if CBP==0

*run difference in difference regression
regress adjusted_rate CBP post CBP_post, r
regress adjusted_rate CBP post CBP_post i.year i.CBP, r 

*difference in difference with interactions
gen CBP_year = year*CBP
xtset year geo_code
xtreg adjusted_rate CBP_post i.year i.CBP, r


*Generate Graph of average annual expenditure between treated and non treated groups
twoway (connected avg_adj_t year) || (connected avg_adj_n year) || (lfit adjusted_rate year if CBP==1, pstyle(p1))  || (lfit adjusted_rate year if CBP==0, pstyle(p2)), ytitle(Average Annual Rate) legend(order(1 "CBP Treated Counties" 2 "Non-CBP Treated Counties" 3 "CBP Linear Fit" 4 "Non-CBP Linear Fit")) title("Average Annual Medicare Reimbursemnts on DME") scheme(538bw) name(Figure_2)

 graph save "Figure_2" "C:\Users\msc19\Box\ECON_388\Data_Project_3\Figure_2.gph", replace
 graph export Figure_2.gph, as(jpg) replace
 
 
*Generate Summary Statistics
sum adjusted_rate if CBP_post==1
sum adjusted_rate if post==0
sum adjusted_rate if post==1 & CBP==0
 
*close log file
log close



