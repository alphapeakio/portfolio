# import the modules and function you will use here
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
from scipy import stats
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut, KFold
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, ElasticNetCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, roc_curve, roc_auc_score, auc
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from statsmodels.stats.stattools import durbin_watson
from statsmodels.graphics.tsaplots import plot_acf
from joblib import Parallel, delayed

cd = "/Users/mcgyverclark/Desktop/Econ 484/ps5/"

#Initial Data loading and cleaning
df = pd.read_csv(cd+'carseats.csv', index_col=0)
text_columns = ['Urban','US']
df = pd.get_dummies(df, columns=text_columns)
df['ShelveLoc_'] = df['ShelveLoc'].map({'Good':3, 'Medium':2,'Bad':1})
drop = ['Urban_No','US_No']
df = df.drop(columns=drop)

#Model Settings
z_out = 1000
standardize = 1

#visualize data
data = ['CompPrice','Income','Advertising','Population','Price','Age','Education','Urban_Yes','US_Yes','ShelveLoc_']
X = df[data]
y = df['Sales']

#generate histograms for all features and target variable
fig, axes = plt.subplots(4,3, figsize=(12,12))
fig.suptitle("Histograms of Features", fontsize=16)

for i, feature in enumerate(X.columns):
    row, col = divmod(i,3)
    ax = axes[row,col]
    sns.histplot(X[feature], ax=ax, kde=True)
    ax.set_title(f'{feature} Histogram')
    ax.set_xlabel(feature)
    feature_mean = X[feature].mean()
    feature_var = X[feature].var()
    feature_std = X[feature].std()

    # Print statistics on the plot
    ax.text(0.05, 0.9, f"Mean: {feature_mean:.2f}", transform=ax.transAxes)
    ax.text(0.05, 0.82, f"Variance: {feature_var:.2f}", transform=ax.transAxes)
    ax.text(0.05, 0.74, f"Std Dev: {feature_std:.2f}", transform=ax.transAxes)

plt.tight_layout()
plt.savefig(cd+'carseat_hist.jpg')
plt.close()
# plt.show()

#Histogram of Target Variable
plt.figure(figsize=(8,4))
sns.histplot(y, kde=True)
target_mean = y.mean()
target_std = y.std()
target_var = y.var()
plt.text(0.05, 0.9, f"Mean: {target_mean:.2f}", transform=plt.gca().transAxes)
plt.text(0.05, 0.85, f"Variance: {target_var:.2f}", transform=plt.gca().transAxes)
plt.text(0.05, 0.8, f"Std Dev: {target_std:.2f}", transform=plt.gca().transAxes)
plt.title("Histogram of Target Variable (Sales)")
plt.xlabel("Sales")
plt.tight_layout()
plt.savefig(cd+'carseat_target_hist.jpg')
plt.close()
# plt.show()

# Prepare training and test data for analysis
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize Data
scaler = StandardScaler()
X_train_stand = scaler.fit_transform(X_train)
X_test_stand = scaler.transform(X_test)

if standardize == 1:
    X_train = X_train_stand
    X_test = X_test_stand
else: 
    X_train = X_train
    X_test = X_test

#Fit a regression tree to the trainin set
regression_tree = DecisionTreeRegressor()
regression_tree.fit(X_train, y_train)
y_pred = regression_tree.predict(X_test)
y_train_pred = regression_tree.predict(X_train)
mse = mean_squared_error(y_test, y_pred)
mse_train = mean_squared_error(y_train, y_train_pred)
r2 = r2_score(y_test, y_pred)
residuals = y_test - y_pred

regression_tree_summary = {
    'Regression Tree Metric': ['MSE Train','MSE Test','R2'],
    'Value': [mse_train,mse,r2]
}
df_regression_tree = pd.DataFrame(regression_tree_summary)


#visualize regression
plt.figure(figsize=(12,6))
plt.subplot(1,3,1)

plt.scatter(y_test, y_pred, c='blue', label='Actual vs. Predicted')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], lw=1, label='Perfectly PRedicted', color='red')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title(f'Regression Tree Model - Actual vs. Predicted')
plt.text(0.1, 0.75, f'MSE: {mse:.2f}')
plt.text(0.1, 0.8, f'R2: {r2:.2f}')
plt.legend()

# Check for Normality (Histogram)
plt.subplot(1,3,2)
sns.histplot(residuals, kde=True, color='blue', ax=plt.gca())
plt.title(f'Regression Tree Model - Residuals Histogram')

# Check for Homoskedasticity and outliers
plt.subplot(1,3,3)
plt.scatter(y_pred, residuals, c='blue', label='Residuals')
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.title(f'Regression Tree Model - Residual Plot')
plt.grid(True)
plt.text(0.1, 0.9, f'Mean Residual: {residuals.mean():.2f}')
plt.text(0.1, 0.85, f'Standard Deviation: {residuals.std():.2f}')
plt.tight_layout()
plt.savefig(cd+'carseat_regression.jpg')
plt.close()
# plt.show()

print(f'MSE: {mse} ')
print(f'MSE Train: {mse_train} ')

#Cross-Validation and find optimal max depth
num_folds = 3
max_depth_values = [depth for depth in range(1,31)]
mse_mean_values = []

for max_depth in max_depth_values:
    regression_tree = DecisionTreeRegressor(max_depth=max_depth)
    cv_scores = cross_val_score(regression_tree, X_train, y_train, cv=num_folds, scoring='neg_mean_squared_error')
    cv_mse_mean = -cv_scores.mean()
    mse_mean_values.append(cv_mse_mean)

optimal_max_depth = max_depth_values[np.argmin(mse_mean_values)]
optimal_mse = np.min(mse_mean_values)

print(f'Optimal max_depth: {optimal_max_depth}')
print(f'Optimal Cross-Validation MSE: {optimal_mse:.2f}')

#Plot max_depth vs. MSE
plt.figure(figsize=(8, 4))
plt.plot(max_depth_values, mse_mean_values, marker='o', linestyle='-')
plt.title('Max Depth vs. Cross-Validated MSE')
plt.xlabel('Max Depth')
plt.ylabel('Cross-Validated MSE')
plt.grid(True)
plt.savefig(cd+'carseat_cv.jpg')
plt.close()
# plt.show()

#Plot Tree with max depth 3 
regression_tree = DecisionTreeRegressor(max_depth=3)
regression_tree.fit(X_train, y_train)

plt.figure(figsize=(10, 6))
plot_tree(regression_tree, filled=True, feature_names=X.columns, class_names=["Predicted Value"], rounded=True)
plt.title("Decision Tree with Max Depth 3")
plt.savefig(cd+'carseats_decision_tree.jpg')
plt.close()
# plt.show()

#Use bagging method Random Forest, analyze data
random_forest = RandomForestRegressor(n_estimators=100, n_jobs=5 ,random_state=42)
random_forest.fit(X_train, y_train)
y_pred = random_forest.predict(X_test)
test_mse = mean_squared_error(y_test, y_pred)

#visualize bagging
plt.figure(figsize=(20, 12))
for tree_idx in range(6):  
    plt.subplot(2, 3, tree_idx + 1)
    plot_tree(random_forest.estimators_[tree_idx], filled=True, feature_names=X.columns, rounded=True)
    plt.title(f'Tree {tree_idx + 1}')
plt.tight_layout()
plt.savefig(cd+'carseats_bagging.jpg')
plt.close()
# plt.show()

#Perform feature importance
feature_importances = random_forest.feature_importances_
feature_importance_pairs = list(zip(X.columns, feature_importances))
sorted_feature_importance_pairs = sorted(feature_importance_pairs, key=lambda x: x[1], reverse=True)

#graph feature importance
plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_feature_importance_pairs)), feature_importances, align='center')
plt.yticks(range(len(sorted_feature_importance_pairs)), X.columns)
plt.xlabel('Feature Importance')
plt.title('Random Forest Feature Importances')
plt.savefig(cd+'carseat_rf_feature_importnace.jpg')
plt.close()
# plt.show()

#print features by importance
df_feature_importance = pd.DataFrame(sorted_feature_importance_pairs, columns=['Feature', 'Importance'])
df_feature_importance

#Random Forest Analysis
max_features_values = ['sqrt','log2'] + list(range(1,11)) + [None]

mse_values = []
feature_importances = []

for max_features in max_features_values:
    random_forest = RandomForestRegressor(n_estimators=100, max_features=max_features, random_state=42)
    random_forest.fit(X_train, y_train)
    y_pred = random_forest.predict(X_test)
    test_mse = mean_squared_error(y_test, y_pred)
    mse_values.append(test_mse)
    feature_importances.append(random_forest.feature_importances_)

for i, max_features in enumerate(max_features_values):
    print(f"Test MSE for max_features='{max_features}': {mse_values[i]:.2f}")

best_max_features = max_features_values[np.argmin(mse_values)]
best_feature_importances = feature_importances[max_features_values.index(best_max_features)]

print(f"Feature Importances for the best-performing model (max_features='{best_max_features}'):")
for feature, importance in sorted(zip(X.columns, best_feature_importances), key=lambda x: x[1], reverse=True):
    print(f"{feature}: {importance:.4f}")

print('The tree improves if you prune it to 6 features. The error generally goes down the more features you add but there is an optimal point at 6')



