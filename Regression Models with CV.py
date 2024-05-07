from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def perform_basic_regression(X, y):
    model = LinearRegression()
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-cv_scores)
    return np.mean(rmse_scores), rmse_scores

def perform_regression_with_scaling(X, y):
    pipeline = Pipeline([
        ('scaler', StandardScaler()), 
        ('linear_regression', LinearRegression())
    ])
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-cv_scores)
    return np.mean(rmse_scores), rmse_scores

def perform_regression_with_feature_selection(X, y, n_features=20):
    forest = RandomForestRegressor(n_estimators=100)
    forest.fit(X, y)
    importances = forest.feature_importances_
    indices = np.argsort(importances)[-n_features:]
    X_selected = X.iloc[:, indices]
    model = LinearRegression()
    cv_scores = cross_val_score(model, X_selected, y, cv=5, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-cv_scores)
    return np.mean(rmse_scores), rmse_scores

file_path = './A2data.tsv'  # Replace with your file path
data = pd.read_csv(file_path, sep='\t')
X = data.drop(['InstanceID', 'Y'], axis=1)
y = data['Y']

# Running all three regression methods
basic_avg_rmse, basic_rmse_scores = perform_basic_regression(X, y)
scaled_avg_rmse, scaled_rmse_scores = perform_regression_with_scaling(X, y)
selected_avg_rmse, selected_rmse_scores = perform_regression_with_feature_selection(X, y)

# Print the results
print("Basic Linear Regression:")
print(f"Average RMSE: {basic_avg_rmse:.4f} ± {np.std(basic_rmse_scores):.4f}")
print(f"RMSE Scores: {basic_rmse_scores}\n")

print("Linear Regression with Feature Scaling:")
print(f"Average RMSE: {scaled_avg_rmse:.4f} ± {np.std(scaled_rmse_scores):.4f}")
print(f"RMSE Scores: {scaled_rmse_scores}\n")

print("Linear Regression with Feature Selection:")
print(f"Average RMSE: {selected_avg_rmse:.4f} ± {np.std(selected_rmse_scores):.4f}")
print(f"RMSE Scores: {selected_rmse_scores}")

# Plotting the results in a grouped bar chart
n_groups = 5  # Number of CV folds
index = np.arange(n_groups)
bar_width = 0.2

plt.bar(index, basic_rmse_scores, bar_width, label='Basic')
plt.bar(index + bar_width, scaled_rmse_scores, bar_width, label='Scaled')
plt.bar(index + 2 * bar_width, selected_rmse_scores, bar_width, label='Selected')

plt.xlabel('CV Fold')
plt.ylabel('RMSE')
plt.title('RMSE Scores by Method and CV Fold')
plt.xticks(index + bar_width, ('1', '2', '3', '4', '5'))
plt.legend()

plt.tight_layout()
plt.show()
