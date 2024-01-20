import pandas as pd
import numpy as np
import seaborn as sns
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

#Model Settings
choose_file = 2
obs = 500
sample = 2

#chooses trunkated or full file version
if choose_file == 1:
    file = 'Appliances_5.json'
elif choose_file ==2:
    file = 'Appliances.json'

df1 = pd.read_json(file, lines=True)
df2 = pd.read_json('meta_Appliances.json', lines=True)
df3 = pd.merge(df1,df2, on='asin', how='inner')

#Takes random sample or leaves the complete data set
if sample==1:
    df = df3.sample(n=obs, random_state=42)
else:
    df = df3
df.columns

#create dataframe
vars = ['title','overall','verified','reviewTime','reviewerID','asin','style','reviewText','summary','unixReviewTime','vote','date','price','category']
df = df[vars]

#clean data
df['price'] = df['price'].str.replace('[^\d.]', '', regex=True)  
df['price'] = pd.to_numeric(df['price'], errors='coerce')
df['reviewText'] = df['reviewText'].astype(str)
df['summary'] = df['summary'].astype(str)


#Review Length
df['str_len'] = df['reviewText'].apply(lambda x: len(str(x)))
#summary length
df['str_len_sum'] = df['summary'].apply(lambda x: len(str(x)))
#verified
df['verified_binary'] = df['verified'].map({True: 1, False:0})
#clean votes
df['vote'] = df['vote'].replace('NaN',0)
df['vote'] = df['vote'].str.replace(',','').astype(float)
#reviewer encoder
label_encoder = LabelEncoder()
df['reviewer_id'] = label_encoder.fit_transform(df['reviewerID'])

#perform sentiment analysis
def get_textblob_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

df['review_sentiment'] = df['reviewText'].apply(get_textblob_sentiment)
df['summary_sentiment'] = df['summary'].apply(get_textblob_sentiment)

#get capital letter ratio
def calculate_ratio(input_string):
    if isinstance(input_string, str):
        num_uppercase = sum(1 for char in input_string if char.isupper())
        num_letters = sum(1 for char in input_string if char.isalpha())
        return num_uppercase / num_letters if num_letters > 0 else 0
    else:
        return 0 
df['capital_ratio'] = df['reviewText'].apply(calculate_ratio)


#create encoded categories
df['category'] = df['category'].astype(str)  
df_encoded = df['category'].str.strip('[]').str.get_dummies(', ')
df = pd.concat([df, df_encoded], axis=1)
label_encoder = LabelEncoder()
df['category_encode'] = label_encoder.fit_transform(df['category'])

#Sentiment Score vs Price Scatter Plot
plt.figure(figsize=(10,6))
plt.scatter(df['review_sentiment'], df['price'], c=df['category_encode'], cmap='viridis')
plt.xlabel('Sentiment Score')
plt.ylabel('Price')
plt.title('Sentiment Score and Price Scatter Plot')
plt.colorbar(label='Category')
plt.tight_layout()
plt.savefig('review_sentiment_price_scatter.jpg')
plt.close()
# plt.show()

#clean data again

df = df.dropna()
df = df.fillna(df.mean())
df = df.dropna(axis=1)


data = df[['price']]

# Standardize the features to have zero mean and unit variance
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Use K-means clustering for 6 clusters
kmeans = KMeans(n_clusters=6, random_state=42)
df['cluster'] = kmeans.fit_predict(data_scaled)
df['cluster_number'] = df['cluster'] + 1

# Visualize clusters
plt.figure(figsize=(10, 6))
plt.scatter(df['review_sentiment'], df['price'], c=df['cluster'], cmap='viridis')
plt.xlabel('Sentiment Score')
plt.ylabel('Price')
plt.title('K-Means Clustering of Price')
plt.savefig('review_sentiment_price_k-means_clustering.jpg')
plt.close()
# plt.show()

#Prove sentiment of summary is correlated with sentiment of review
vars = ['title','overall','verified','reviewerID','asin','unixReviewTime','verified_binary','price','category_encode','cluster','cluster_number','review_sentiment', 'summary_sentiment', 'str_len','str_len_sum','capital_ratio']
df = df[vars]

variables_of_interest = ['summary_sentiment', 'str_len']
variable_titles = ['Review Title Sentiment', 'Review Character Length']

fig, axes = plt.subplots(1, 2, figsize=(15, 7))  
axes = axes.flatten()

for i, (title, variable) in enumerate(zip(variable_titles, variables_of_interest)):
    sns.scatterplot(x='review_sentiment', y=variable, data=df, ax=axes[i], color='green')
    axes[i].set_ylabel(title)
    axes[i].set_xlabel('Review Sentiment')

plt.suptitle('Feature Variables on Review Sentiment')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  
plt.savefig('review_sentiment_feature_scatter.jpg')
plt.close()
# plt.show()
# df.to_json('appliances_2.json')


hist_vars = ['overall', 'unixReviewTime', 'verified_binary', 'price', 'category_encode', 'cluster_number', 'review_sentiment', 'summary_sentiment', 'str_len', 'str_len_sum', 'capital_ratio']
df = df[hist_vars]

# Calculate the number of rows and columns for the grid
num_rows = math.ceil(len(hist_vars)/2)
num_cols = 2  # Adjust the number of columns based on your preference

fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 6 * num_rows))

# Flatten the 2D array of subplots to simplify indexing
axes = axes.flatten()

for i, variable in enumerate(hist_vars):
    # Check if the variable has data
    if not df[variable].empty:
        # Calculate mean, standard deviation, and variance
        mean_val = np.mean(df[variable])
        std_dev = np.std(df[variable])
        var = np.var(df[variable])

        # Create histogram
        sns.histplot(df[variable], kde=True, ax=axes[i], color='green')

        # Mark the presence of outliers greater than z=3
        outliers = df[(df[variable] - mean_val) / std_dev > 3]
        sns.scatterplot(x=outliers[variable], y=np.zeros_like(outliers[variable]) + 1, color='red', marker='o', ax=axes[i])

        # Set plot labels and title
        axes[i].set_title(f'Histogram of {variable}')
        axes[i].set_xlabel(variable)
        axes[i].set_ylabel('Frequency')

        # Annotate with mean, standard deviation, variance
        axes[i].text(0.1, 0.9, f'Mean={mean_val:.2f}', transform=axes[i].transAxes, ha='left')
        axes[i].text(0.1, 0.8, f'Std Dev={std_dev:.2f}', transform=axes[i].transAxes, ha='left')
        axes[i].text(0.1, 0.7, f'Variance={var:.2f}', transform=axes[i].transAxes, ha='left')

        # Annotate presence of outliers
        axes[i].text(0.1, 0.6, f'Outliers (Z>3): {len(outliers)}', transform=axes[i].transAxes, ha='left', color='red')
    else:
        # Remove empty subplot
        fig.delaxes(axes[i])
        
plt.savefig('feature_histograms.jpg')
plt.tight_layout()
plt.close()
# plt.show()

clusters = df['cluster_number'].unique()

result_df = pd.DataFrame(columns=['Cluster', 'Min_Price', 'Max_Price'])

for cluster in clusters:
    cluster_data = df[df['cluster_number'] == cluster]
    min_price = cluster_data['price'].min()
    max_price = cluster_data['price'].max()
    result_df = result_df.append({'Cluster': cluster, 'Min_Price': min_price, 'Max_Price': max_price}, ignore_index=True)
    result_df = result_df.sort_values(by='Cluster')

result_df.to_csv('clusters.csv')

# %pip install scikit-learn==0.24.2
# %pip install econml
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from sklearn.model_selection import train_test_split
from econml.dml import CausalForestDML as CausalForest
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

### df is the main data Mcgyver built and df_meta is the data with prices
df = pd.read_json('/Users/zachhflynn/Library/CloudStorage/Box-Box/School/Econ_484/Project/Review_Data/appliances_1.json')
df_meta = pd.read_json('/Users/zachhflynn/Library/CloudStorage/Box-Box/School/Econ_484/Project/Meta_Data/meta_Appliances.json', lines = True)

### drop columns alredy in the main data
df_meta = df_meta.drop(['price', 'date', 'title'], axis=1)
df = df.merge(df_meta, on = 'asin')

### drop abything we don't have a price for and make price a float
df_low= df[~df.price.isna()]
df_low.price = df_low.price.apply(lambda x: float(x))


### Turn our catigorical data into dummies, this takes like a hour to run
df_low = pd.concat([df_low, pd.get_dummies(df_low['category'].explode()).groupby(level=0).sum()], axis=1).drop('category', axis = 1)
df_low = pd.concat([df_low, pd.get_dummies(df_low['brand'].explode()).groupby(level=0).sum()], axis=1).drop('brand', axis = 1)

### That code took too long to run not to save the data just incase we screw up the df_low variable.
### Also, it's called df_low beacuse when we wrote out the code intially the data fram onlky had items that were $5 or less
df_low.to_csv('/Users/zachhflynn/Library/CloudStorage/Box-Box/School/Econ_484/Project/Review_Data/exlpoded_data.csv')

### New data frame that we oare going to work with
df_wack = pd.read_csv('/Users/zachhflynn/Library/CloudStorage/Box-Box/School/Econ_484/Project/Review_Data/exlpoded_data.csv')


### Make price a log to help with understanding
df_wack['lnprice'] = df_wack.price.apply(lambda x: np.log(x))

#### Condense the data down to only numeric columns plus asin, and group by asin so we get the average for our numeric data per peroduct
numeric_columns = df_wack.select_dtypes(include=['number'])
df_wack= pd.concat([df_wack['asin'], numeric_columns], axis=1)
df_wack= df_wack.groupby('asin').mean()

### Make sentiment score discriete or else RCF gets angry
df_wack['bucket'] = pd.qcut(df_wack['review_sentiment'], q=5, duplicates='drop', labels=range(-2,3))

### see how many unique products there are
df_wack.index.nunique()



### Mcgyver had made a data set with cluster groups but when it merge it knocked our prodcuts down to like 637, so I just had made the clusters
### using the price cutoffs from k-means clustering
conditions = [
    (df_wack['price'] >= 0) & (df_wack['price'] <= 26.62),
    (df_wack['price'] > 26.62) & (df_wack['price'] <= 72.41),
    (df_wack['price'] > 72.41) & (df_wack['price'] <= 159.99),
    (df_wack['price'] > 159.99) & (df_wack['price'] <= 297.63),
    (df_wack['price'] > 297.63) & (df_wack['price'] <= 560.34),
    (df_wack['price'] > 560.34) & (df_wack['price'] <= 1449.95),
    (df_wack['price'] > 1449.95)
]

values = [1, 2, 3, 4, 5, 6, 'drop']

# Use numpy.select to create the 'cluster' column
df_wack['cluster'] = np.select(conditions, values, default=np.nan)

# Drop rows where the 'cluster' column is 'drop'
df_wack = df_wack[df_wack['cluster'] != 'drop']

# Convert 'cluster' column to appropriate data type (optional)
df_wack['cluster'] = pd.to_numeric(df_wack['cluster'])

### see how many products are in each cluster
for i in range(1,7):
    df_rcf = df_wack[df_wack.cluster == i]
    print(f'Cluster {i} shape: {df_rcf.shape}')

### see which columns had nan values and drop them or else RCF gets angry
nan_columns = df_wack.columns[df_wack.isna().any()].tolist()
nan_columns


### These lists will be used later to get the ATE for each cluster
ate1 = []
ate2 = []
ate3 = []
ate4 = []
ate5 = []
ate6 = []

### RCF is random so we run it 100 times and take the average of the average treatment effects
for _ in range(100):
    ### this splits the rcf to only run on the clusters
    for i in range(1,7):
        df_rcf = df_wack[df_wack.cluster == i]


        y = df_rcf["lnprice"]
        z = df_rcf["bucket"]
        ### These vars were either too much like lnprice/bucket or had NaN types
        x = df_rcf.drop(["summary_sentiment", 'price', "lnprice", "review_sentiment", 'reviewer_id', 'bucket', 'cluster', 'vote', 'fit', 'tech2'], axis=1)

        rcf = CausalForest(n_estimators=500, discrete_treatment=True, criterion="het").fit(y, z, X=x)

        ### allow us to et the ATE for each cluster and look at the p-values for the last run of the RCF for each cluster
        if i == 1:
            ate1.append(rcf.ate_[0])
            ate1_sum = rcf.summary()
        elif i == 2:
            ate2.append(rcf.ate_[0])
            ate2_sum = rcf.summary()
        elif i == 3:
            ate3.append(rcf.ate_[0])
            ate3_sum = rcf.summary()
        elif i == 4:
            ate4.append(rcf.ate_[0])
            ate4_sum = rcf.summary()
        elif i == 5:
            ate5.append(rcf.ate_[0])
            ate5_sum = rcf.summary()
        else:
            ate6.append(rcf.ate_[0])
            ate6_sum = rcf.summary()

        # print(f'ATE for Culster {i}: {rcf.ate_[0]}\n')

### ATE for each cluster
l = [ate1, ate2, ate3, ate4, ate5, ate6]
n = 1
for i in l:
   print(f'Avg ATE for Cluster {n}: {np.mean(i)}\n')
   n += 1

### THis and the next 5 blocks are the p-values for each cluster
ate1_sum

ate2_sum

ate3_sum

ate4_sum

ate5_sum

ate6_sum

### THis is the treatment of of all the covariates
rcf.effect_inference(x).summary_frame(alpha=0.05, value=0, decimals=3)

### population summary, not useful for us right now, but kinda interesting
rcf.effect_inference(x).population_summary(alpha=0.1, value=0, decimals=3, tol=0.001)

df_heat = pd.read_csv('/Users/zachhflynn/Library/CloudStorage/Box-Box/School/Econ_484/Project/Review_Data/heat.csv')
df_heat = df_heat[['cluster','bucket', 'ate']]
df_heat.columns

#### Make heat map for each bucket and cluster and their ATE
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.figure(figsize=(10,6))
fig = plt.figure()
ax = plt.subplot()
main = ax.scatter(
    df_heat["bucket"], df_heat["cluster"], c=df_heat["ate"], cmap="plasma", marker="s", s=500
)
plt.suptitle("Estimated Treatment effects")
plt.xlabel("Sentiment Bucket")
plt.ylabel("Price Cluster")

ax.set_xticks(np.arange(df_heat["bucket"].min(), df_heat["bucket"].max() + 1, 1))

# create an Axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(main, cax=cax)
plt.show()