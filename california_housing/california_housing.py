import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
from scipy import stats
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut, KFold
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, ElasticNetCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from statsmodels.stats.stattools import durbin_watson
from statsmodels.graphics.tsaplots import plot_acf
from joblib import Parallel, delayed


#Ignore warnings
warnings.filterwarnings("ignore")
#load dataset
housing = fetch_california_housing()
print(housing['DESCR'])
X = pd.DataFrame(housing['data'], columns=housing['feature_names'])
y = pd.Series(housing['target'])
###########################################################################################
#settings on sample size, standardization, and outlier threshold
random = 0
obs = 500
standardize = 1
z_out = 4
price_limit = 4.6
##########################################################################################
if random == 1:
    size = obs
    chosen_idx = np.random.choice(len(X.index), replace=False, size = size)
    X = X.iloc[chosen_idx]
    y = y.iloc[chosen_idx]
    
else:
    X = X
    y = y
    size = len(X)

#Get an idea of what the data looks like
fig, axes = plt.subplots(3, 3, figsize=(12, 12))
fig.suptitle("Histograms of Features", fontsize=16)

#make histogram plots for all features in X
for i, feature in enumerate(X.columns):
    row, col = divmod(i, 3)
    ax = axes[row, col]
    sns.histplot(X[feature], ax=ax, kde=True)
    ax.set_title(f"{feature} Histogram")
    ax.set_xlabel(feature)

    # Calculate statistics
    feature_mean = X[feature].mean()
    feature_std = X[feature].std()
    feature_var = X[feature].var()

    # Print statistics on the plot
    ax.text(0.05, 0.9, f"Mean: {feature_mean:.2f}", transform=ax.transAxes)
    ax.text(0.05, 0.85, f"Variance: {feature_var:.2f}", transform=ax.transAxes)
    ax.text(0.05, 0.8, f"Std Dev: {feature_std:.2f}", transform=ax.transAxes)

    # Identify significant outliers based on z-score
    z_scores = np.abs((X[feature] - feature_mean) / feature_std)
    significant_outliers = X[z_scores > z_out-1]  

    if not significant_outliers.empty:
        ax.text(0.05, 0.75, f'Significant Outliers (z>{z_out-1})', transform=ax.transAxes, color="red")

plt.tight_layout()
# plt.show()
plt.savefig('1_feat_hist.jpg')
plt.close()

# Histgram for the target variable
plt.figure(figsize=(8, 4))
sns.histplot(y, kde=True)
target_mean = y.mean()
target_std = y.std()
target_var = y.var()
plt.text(0.05, 0.9, f"Mean: {target_mean:.2f}", transform=plt.gca().transAxes)
plt.text(0.05, 0.85, f"Variance: {target_var:.2f}", transform=plt.gca().transAxes)
plt.text(0.05, 0.8, f"Std Dev: {target_std:.2f}", transform=plt.gca().transAxes)
plt.title("Histogram of Target Variable (Median Housing Price)")
plt.xlabel("Median Price")

z_scores = np.abs((y - target_mean) / target_std)
significant_outliers = y[z_scores > z_out-1]  

if not significant_outliers.empty:
    plt.text(0.05, 0.75, f"Significant Outliers (z > {z_out-1})", transform=plt.gca().transAxes, color="red")

# plt.show()
plt.savefig('2_target_hist.jpg')
plt.close()

############################################################################################

# Drop rows with outliers
outlier_indices = []

for col in X.columns:
    col_mean = X[col].mean()
    col_std = X[col].std()
    z_scores = np.abs((X[col] - col_mean) / col_std)
    significant_outliers = X[col][z_scores > z_out]
    outlier_indices.extend(significant_outliers.index)

# Remove rows with outliers from X and y
X = X.drop(outlier_indices)
y = y.drop(outlier_indices)

# Remove rows where y is equal to a price limit
indices_to_keep = [i for i, value in enumerate(y) if value <=price_limit]
X = X.iloc[indices_to_keep]
y = y.iloc[indices_to_keep]

# Prepare training and test data for analysis
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize Data
scaler = StandardScaler()
X_train_stand = scaler.fit_transform(X_train)
X_test_stand = scaler.transform(X_test)


if standardize == 1:
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
else: 
    X_train = X_train
    X_test = X_test

################################################################################################
#Perform analysis of OLS, RIDGE, LASSO, and ELASTIC NET models to see which ones perform the best
model_names = ['OLS', 'RIDGE', 'LASSO', 'ELASTIC NET']
models = [LinearRegression(), Ridge(), Lasso(), ElasticNet()]
mse_scores = []
r2_scores = []
mae_scores = []
coefficients = []
summary_data = []
coef_data = []
coef_names_data = []
coef_names =  list(X.columns)
coef_names_data.append(list(coef_names))

# Set the number of rows and columns for the grid
num_rows = 3
num_cols = 4 

# Create a figure with subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 12))
fig.suptitle("Model Evaluation", fontsize=16)

# Loop through the models
for i, (name, model) in enumerate(zip(model_names, models)):
    row, col = divmod(i, num_cols)

    # Fit the model and make predictions
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    coef = model.coef_
    intercept = model.intercept_
    residuals = y_test - y_pred
    durbin_watson_stat = durbin_watson(residuals)
    coefficients.append(coef)
    flat_coef = coef.flatten()
    mse_scores.append(mse)
    r2_scores.append(r2)
    mae_scores.append(mae)
    coef_data.append([name] + list(flat_coef))
    summary_data.append([name, size, len(X.columns), r2, mse, mae, durbin_watson_stat, intercept] + list(flat_coef))

    # Plot graphs for the current model
    ax1 = axes[row, col]
    ax2 = axes[row+1, col]
    ax3 = axes[row+2, col] 
    
    # Check for Linearity and Homoskedasticity and outliers
    ax1.scatter(y_test, y_pred, c='blue', label='Actual vs. Predicted')
    ax1.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], lw=1, label='Perfectly Predicted', color='red')
    ax1.set_xlabel('Actual')
    ax1.set_ylabel('Predicted')
    ax1.set_title(f'{name} Model - Actual vs. Predicted')
    ax1.text(0.1, 0.7, f'MAE: {mae:.2f}', transform=ax1.transAxes)
    ax1.text(0.1, 0.75, f'MSE: {mse:.2f}', transform=ax1.transAxes)
    ax1.text(0.1, 0.8, f'R2: {r2:.2f}', transform=ax1.transAxes)
    ax1.legend()

    # Check for Normality (Histogram)
    sns.histplot(residuals, kde=True, color='blue', ax=ax2)
    ax2.set_title(f'{name} Model - Residuals Histogram')

    # Check for Linearity and Homoskedasticity and outliers
    ax3.scatter(y_pred, residuals, c='blue', label='Residuals')
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('Residuals')
    ax3.set_title(f'{name} Model - Residual Plot')
    ax3.grid(True)
    ax3.text(0.1, 0.9, f'Mean Residual: {residuals.mean():.2f}', transform=ax3.transAxes)
    ax3.text(0.1, 0.85, f'Standard Deviation: {residuals.std():.2f}', transform=ax3.transAxes)

    # Identify and mark potential outliers
    threshold = (z_out - 1)
    outliers = np.abs(residuals) > threshold
    ax3.scatter(y_pred[outliers], residuals[outliers], c='red', label=f'Outliers: z>{z_out-1}', s=50)
    ax3.legend()
# Remove empty subplots
for i in range(num_cols * num_rows, len(model_names)):
    fig.delaxes(axes[i // num_cols, i % num_cols])

# Adjust the layout and save the figure
plt.tight_layout()
plt.savefig('3_model_evaluation_combined.jpg')
# plt.show()    

#Print all Summary Statistics
summary_df = pd.DataFrame(summary_data, columns=[
    'Model Name', 'Number of Instances', 'Number of Attributes', 'R-Squared', 'Mean Squared Error', 'Mean Absolute Error', 'Durbin-Watson Statistic', 'Intercept'
    ]+ list(coef_names))
print(summary_df)
print('')
###################################################################################

# Create a bar graph of coefficients
num_features = len(coefficients[0])
x = np.arange(num_features)
width = 0.2

plt.figure(figsize=(10, 6))
for i, name in enumerate(model_names):
    plt.bar(x + i * width, coefficients[i], width, label=name)

plt.xlabel('Features')
plt.ylabel('Coefficients')
plt.title('Coefficients of Different Models')
plt.xticks(x + width * (len(model_names)/2), coef_names, rotation = 45, ha='right') 
plt.legend()
plt.grid(axis='y')
# plt.show()
plt.savefig('4_coef.jpg')
plt.close()

# Find the index of the best model based on MSE and R2
best_r2_index = r2_scores.index(max(r2_scores))
best_mse_index = mse_scores.index(min(mse_scores))
best_mae_index = mae_scores.index(min(mae_scores))

# Create separate bar graphs for MSE and R2
x = np.arange(len(model_names))
width = 0.4

plt.figure(figsize=(12, 6))

# Bar graph for R2
plt.subplot(1, 3, 1)
plt.bar(x, r2_scores, width, label='R2', color=['g' if i == best_r2_index else 'b' for i in range(len(model_names))])
plt.xlabel('Models')
plt.ylabel('R2')
plt.title('R2 Comparison for Different Models')
plt.xticks(x, model_names)

# Bar graph for MSE
plt.subplot(1, 3, 2)
plt.bar(x, mse_scores, width, label='MSE', color=['g' if i == best_mse_index else 'b' for i in range(len(model_names))])
plt.xlabel('Models')
plt.ylabel('MSE')
plt.title('MSE Comparison for Different Models')
plt.xticks(x, model_names)

#Bar graph for MAE
plt.subplot(1, 3, 3)
plt.bar(x, mae_scores, width, label='MAE', color=['g' if i == best_mae_index else 'b' for i in range(len(model_names))])
plt.xlabel('Models')
plt.ylabel('MAE')
plt.title('MAE Comparison for Different Models')
plt.xticks(x, model_names)

plt.tight_layout()
# plt.show()
plt.savefig('5_best_model.jpg')
plt.close()

best_model_r2 = model_names[r2_scores.index(max(r2_scores))]
best_model_mse = model_names[mse_scores.index(min(mse_scores))]
best_model_mae = model_names[mae_scores.index(min(mae_scores))]

print(f"The best model based on R-squared is: {best_model_r2}")
print(f"The best model based on MSE is: {best_model_mse}")
print(f"The best model based on MAE is: {best_model_mae}")
print('')

##########################################################################################################
#Perfrom cross-validation to see if we can improve Ridge, and see if Lasso could be applicable

alphas = 10**np.linspace(-6,6,100)

ridge_scores = []
lasso_scores = []
num_folds=5

for alpha in alphas:
    ridge_model = Ridge(alpha=alpha)
    lasso_model = Lasso(alpha=alpha)
    
    ridge_cv_scores = cross_val_score(ridge_model, X_train, y_train, cv=num_folds, scoring='neg_mean_squared_error')
    lasso_cv_scores = cross_val_score(lasso_model, X_train, y_train, cv=num_folds, scoring='neg_mean_squared_error')
    
    # Convert negative MSE scores to positive values
    ridge_scores.append(-np.mean(ridge_cv_scores))
    lasso_scores.append(-np.mean(lasso_cv_scores))

best_ridge_alpha = alphas[np.argmin(ridge_scores)]
best_lasso_alpha = alphas[np.argmin(lasso_scores)]

print(f"Best Ridge Alpha: {best_ridge_alpha}")
print(f"Best Lasso Alpha: {best_lasso_alpha}")
print('')

#Plot 
plt.figure(figsize=(10, 6))

# Plot Ridge cross-validation scores
plt.semilogx(alphas, ridge_scores, label='Ridge', color='blue')

# Plot Lasso cross-validation scores
plt.semilogx(alphas, lasso_scores, label='Lasso', color='red')

# Mark the optimal alpha values with vertical lines
plt.axvline(best_ridge_alpha, color='blue', linestyle='--', label=f'Best Ridge Alpha: {best_ridge_alpha:.4f}')
plt.axvline(best_lasso_alpha, color='red', linestyle='--', label=f'Best Lasso Alpha: {best_lasso_alpha:.4f}')

# Label the axes and add a legend
plt.xlabel('Alpha (Penalty Parameter)')
plt.ylabel('Cross-Validation Score (MSE)')
plt.title('Cross-Validation Score vs. Alpha for Ridge and Lasso')
plt.legend()

# Show the plot
plt.grid()
# plt.show()
plt.savefig('6_ridge_lasso_cv.jpg')
plt.close()

#######################################################################################################
#Perform cross-validation for Elastic Net

# Define a function to perform cross-validation for a given combination of alpha and l1_ratio
def perform_cross_validation(alpha, l1_ratio, X_train, y_train, cv):
    elastic_net_model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
    y_pred = elastic_net_model.fit(X_train, y_train).predict(X_train)
    mse = mean_squared_error(y_train, y_pred)
    return mse

# Define a range of alpha and l1_ratio values
alphas = 10**np.linspace(-10, 10, 100) 
l1_ratios = np.linspace(0.1, 1.0, 10)  

elastic_net_scores_loocv = []
elastic_net_scores_kfold = []

# LOOCV
loocv = LeaveOneOut()

# Perform LOOCV with parallel processing
elastic_net_scores_loocv = Parallel(n_jobs=-1)(
    delayed(perform_cross_validation)(alpha, l1_ratio, X_train, y_train, loocv) 
    for alpha in alphas for l1_ratio in l1_ratios
)

# k-fold Cross-Validation (k=5)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Perform k-fold cross-validation with parallel processing
elastic_net_scores_kfold = Parallel(n_jobs=-1)(
    delayed(perform_cross_validation)(alpha, l1_ratio, X_train, y_train, kf) 
    for alpha in alphas for l1_ratio in l1_ratios
)

# Find the best alpha and l1_ratio based on minimum MSE
best_loocv_idx = np.argmin(elastic_net_scores_loocv)
best_kfold_idx = np.argmin(elastic_net_scores_kfold)
best_loocv_alpha, best_loocv_l1_ratio = np.unravel_index(best_loocv_idx, (len(alphas), len(l1_ratios)))
best_kfold_alpha, best_kfold_l1_ratio = np.unravel_index(best_kfold_idx, (len(alphas), len(l1_ratios)))

print(f"Best Elastic Net Alpha (LOOCV): {alphas[best_loocv_alpha]:.4f}")
print(f"Best Elastic Net L1 Ratio (LOOCV): {l1_ratios[best_loocv_l1_ratio]:.4f}")
print(f"Best Elastic Net Alpha (k-fold): {alphas[best_kfold_alpha]:.4f}")
print(f"Best Elastic Net L1 Ratio (k-fold): {l1_ratios[best_kfold_l1_ratio]:.4f}")
print('')

#Plot mesh graphs
plt.figure(figsize=(12, 5))
alpha_mesh, l1_ratio_mesh = np.meshgrid(alphas, l1_ratios)

# Swap the dimensions
alpha_mesh, l1_ratio_mesh = l1_ratio_mesh, alpha_mesh
elastic_net_scores_loocv_reshaped = np.array(elastic_net_scores_loocv).reshape(len(l1_ratios), len(alphas))
elastic_net_scores_kfold_reshaped = np.array(elastic_net_scores_kfold).reshape(len(l1_ratios), len(alphas))


# Contour plot for LOOCV scores
plt.subplot(1, 2, 1)
contour = plt.contourf(alpha_mesh, l1_ratio_mesh, elastic_net_scores_loocv_reshaped, cmap='viridis')
plt.colorbar(contour, label='Mean Squared Error')
plt.xlabel('Alpha (Penalty Parameter)')
plt.ylabel('L1 Ratio')
plt.title('LOOCV Scores')

# Contour plot for k-fold cross-validation scores
plt.subplot(1, 2, 2)
contour = plt.contourf(alpha_mesh, l1_ratio_mesh, elastic_net_scores_kfold_reshaped, cmap='viridis')
plt.colorbar(contour, label='Mean Squared Error')
plt.xlabel('Alpha (Penalty Parameter)')
plt.ylabel('L1 Ratio')
plt.title('k-fold Cross-Validation Scores')

plt.tight_layout()
# plt.show()
plt.savefig('7_en_cv.jpg')
plt.close()

############################################################################################################
#rerun all models with new hyperparameters found
tuned_model_names = ['OLS', 'RIDGE', 'LASSO', 'ELASTIC NET']
final_best_loocv_l1_ratio = best_loocv_l1_ratio if 0 <= best_loocv_l1_ratio <= 1 else 0.7
tuned_models = [LinearRegression(), Ridge(alpha=best_ridge_alpha), Lasso(alpha=best_lasso_alpha), ElasticNet(alpha=best_loocv_alpha, l1_ratio=final_best_loocv_l1_ratio)]
alphas = ['', best_ridge_alpha, best_lasso_alpha, best_loocv_alpha]
tuned_mse_scores = []
tuned_r2_scores = []
tuned_mae_scores = []
tuned_coefficients = []
tuned_coef_data = []
tuned_summary_data = []
tuned_coef_names = list(X.columns)

fig, axes = plt.subplots(num_rows, num_cols, figsize=(18,12))
fig.suptitle("Tuned Model Evaluation", fontsize=16)
                         
                         

for i,(name, tuned_model, alpha) in enumerate(zip(tuned_model_names, tuned_models,alphas)):
    row,col = divmod(i, num_cols)
    tuned_model.fit(X_train, y_train)
    tuned_y_pred = tuned_model.predict(X_test)
    tuned_mse = mean_squared_error(y_test, tuned_y_pred)
    tuned_r2 = r2_score(y_test, tuned_y_pred)
    tuned_mae = mean_absolute_error(y_test, tuned_y_pred)
    tuned_intercept = tuned_model.intercept_
    tuned_coef = tuned_model.coef_
    tuned_residuals = y_test - tuned_y_pred
    tuned_durbin_watson_stat = durbin_watson(tuned_residuals)
    tuned_flat_coef = tuned_coef.flatten()
    tuned_mse_scores.append(tuned_mse)
    tuned_r2_scores.append(tuned_r2)
    tuned_mae_scores.append(tuned_mae)
    tuned_coefficients.append(tuned_coef)
    tuned_coef_data.append([name] + list(tuned_flat_coef))
    fbllr = final_best_loocv_l1_ratio if i>2 else ""
    tuned_summary_data.append([name, size, len(X.columns), alpha, fbllr, tuned_mse, tuned_r2, tuned_mae, tuned_durbin_watson_stat, tuned_intercept] + list(tuned_flat_coef))

    # Plot graphs for the current model
    ax1 = axes[row, col]
    ax2 = axes[row+1, col]
    ax3 = axes[row+2, col] 
    
    # Check for Linearity and Homoskedasticity and outliers
    ax1.scatter(y_test, tuned_y_pred, c='blue', label='Actual vs. Predicted')
    ax1.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], lw=1, label='Perfectly Predicted', color='red')
    ax1.set_xlabel('Actual')
    ax1.set_ylabel('Predicted')
    ax1.set_title(f'{name} Tuned Model - Actual vs. Predicted')
    ax1.text(0.1, 0.6, f'MAE: {tuned_mae:.2f}', transform=ax1.transAxes)
    ax1.text(0.1, 0.65, f'MSE: {tuned_mse:.2f}', transform=ax1.transAxes)
    ax1.text(0.1, 0.7, f'R2: {tuned_r2:.2f}', transform=ax1.transAxes)
    ax1.legend()

    # Check for Normality (Histogram)
    sns.histplot(tuned_residuals, kde=True, color='blue', ax=ax2)
    ax2.set_title(f'{name} Tuned Model - Residuals Histogram')

    # Check for Linearity and Homoskedasticity and outliers
    ax3.scatter(tuned_y_pred, tuned_residuals, c='blue', label='Residuals')
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('Residuals')
    ax3.set_title(f'{name} Tuned Model - Residual Plot')
    ax3.grid(True)
    ax3.text(0.1, 0.7, f'Mean Residual: {tuned_residuals.mean():.2f}', transform=ax3.transAxes)
    ax3.text(0.1, 0.75, f'Standard Deviation: {tuned_residuals.std():.2f}', transform=ax3.transAxes)

    # Identify and mark potential outliers
    threshold = (z_out - 1)
    tuned_outliers = np.abs(tuned_residuals) > threshold
    ax3.scatter(tuned_y_pred[tuned_outliers], tuned_residuals[tuned_outliers], c='red', label=f'Outliers: z>{z_out-1}', s=50)
    ax3.legend()
   
# Remove empty subplots
for i in range(num_cols * num_rows, len(tuned_model_names)):
    fig.delaxes(axes[i // num_cols, i % num_cols])

# Adjust the layout and save the figure
plt.tight_layout()
plt.savefig('8_tuned_model_evaluation_combined.jpg')
# plt.show()  

#Print all tuned Summary Statistics
tuned_summary_df = pd.DataFrame(tuned_summary_data, columns=[
    'Model Name', 'Number of Instances', 'Number of Attributes', 'Alpha', 'L1_Ratio',
    'Mean Squared Error', 'R-squared', 'Mean Absolute Error', 'Durbin-Watson Statistic', 'Intercept'] + list(tuned_coef_names))
print(tuned_summary_df)
print('')

# Create a bar graph of coefficients
num_features = len(tuned_coefficients[0])
x = np.arange(num_features)
width = 0.2

plt.figure(figsize=(10, 6))
for i, name in enumerate(tuned_model_names):
    plt.bar(x + i * width, tuned_coefficients[i], width, label=name)

plt.xlabel('Features')
plt.ylabel('Coefficients')
plt.title('Coefficients of Different Models')
plt.xticks(x + width * (len(tuned_model_names) / 2), tuned_coef_names, rotation=45, ha='right')
plt.legend()
plt.grid(axis='y')
# plt.show()
plt.savefig('9_tuned_coef.jpg')
plt.close()

# Find the index of the best model based on R2, MSE, MAE
tuned_best_r2_index = tuned_r2_scores.index(max(tuned_r2_scores))
tuned_best_mse_index = tuned_mse_scores.index(min(tuned_mse_scores))
tuned_best_mae_index = tuned_mae_scores.index(min(tuned_mae_scores))

# Create separate bar graphs for R2, MSE, MAE
x = np.arange(len(tuned_model_names))
width = 0.4

plt.figure(figsize=(12, 6))

# Bar graph for R2
plt.subplot(1, 3, 1)
plt.bar(x, tuned_r2_scores, width, label='R2', color=['g' if i == tuned_best_r2_index else 'b' for i in range(len(tuned_model_names))])
plt.xlabel('Models')
plt.ylabel('R2')
plt.title('R2 Comparison for Different Tuned Models')
plt.xticks(x, tuned_model_names)

# Bar graph for MSE
plt.subplot(1, 3, 2)
plt.bar(x, tuned_mse_scores, width, label='MSE', color=['g' if i == tuned_best_mse_index else 'b' for i in range(len(tuned_model_names))])
plt.xlabel('Models')
plt.ylabel ('MSE')
plt.title('MSE Comparison for Different Tuned Models')
plt.xticks(x, tuned_model_names)

# Bar graph for MAE
plt.subplot(1, 3, 3)
plt.bar(x, tuned_mae_scores, width, label='MAE', color=['g' if i == tuned_best_mae_index else 'b' for i in range(len(tuned_model_names))])
plt.xlabel('Models')
plt.ylabel('MAE')
plt.title('MAE Comparison for Different Tuned Models')
plt.xticks(x, tuned_model_names)

plt.tight_layout()
# plt.show()
plt.savefig('10_tuned_best_model.jpg')
plt.close()

#Print best model reccomendations
tuned_best_model_r2 = tuned_model_names[tuned_best_r2_index]
tuned_best_model_mse = tuned_model_names[tuned_best_mse_index]
tuned_best_model_mae = tuned_model_names[tuned_best_mae_index]

print(f"The best tuned model based on R-squared is: {tuned_best_model_r2}")
print(f"The best tuned model based on MSE is: {tuned_best_model_mse}")
print(f"The best tuned model based on MAE is: {tuned_best_model_mae}")
print('')
