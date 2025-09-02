# Predictive Maintenance in Manufacturing

This project addresses the challenge of predictive maintenance in a manufacturing firm specializing in heavy machinery. The goal is to predict equipment failures and schedule maintenance efficiently, thereby reducing downtime costs. We utilize sensor data including temperature, vibration, pressure, and runtime hours to predict 'days until failure'.

## Problem Statement

An initial linear regression model exhibited overfitting, performing poorly on new sensor readings. This document outlines the steps taken to resolve this issue using ensemble learning techniques, evaluate model performance, and propose further improvements.

## 1. Overfitting Analysis and Bias-Variance Tradeoff

### Overfitting Explained

In the context of predictive maintenance, overfitting occurs when a model, such as a linear regression model, learns the training data too well, including its noise and random fluctuations, rather than the underlying patterns. For our sensor data, this means the model might be memorizing specific sensor readings and their corresponding 'days until failure' from the training set, instead of generalizing to new, unseen sensor data. This leads to high accuracy on the training data but poor performance (high error) on new, unseen data.

### Bias-Variance Tradeoff

*   **Bias**: This refers to the simplifying assumptions made by a model to make the target function easier to learn. High bias can cause a model to miss the relevant relations between features and target outputs (underfitting). In our case, a very simple model might not capture the complex relationships between sensor readings and equipment failure.
*   **Variance**: This refers to the model's sensitivity to small fluctuations in the training data. High variance can cause a model to model the random noise in the training data, rather than the intended outputs (overfitting). Our initial linear regression model likely suffered from high variance, making it too sensitive to the specific noise in the training sensor data.

The **bias-variance tradeoff** is a central concept in machine learning. As model complexity increases, bias tends to decrease (the model can capture more complex patterns), but variance tends to increase (the model becomes more sensitive to noise). Conversely, as model complexity decreases, bias tends to increase, and variance tends to decrease. The goal is to find a balance that minimizes the total error on unseen data.

For our problem, the initial linear regression model had high variance, indicating overfitting. We need a technique that can reduce this variance without significantly increasing bias, thereby improving generalization to new sensor readings.



## 2. Ensemble Learning for Overfitting Resolution

To address the overfitting observed with the linear regression model, we will employ an ensemble learning technique. Specifically, we choose **Random Forest Regressor**.

### Justification for Random Forest

Random Forest is an ensemble method that operates by constructing a multitude of decision trees at training time and outputting the mean prediction (regression) of the individual trees. It is particularly effective at reducing variance, which is the primary cause of overfitting in our linear regression model. Here's why it's a suitable choice:

*   **Variance Reduction**: By averaging the predictions of many decorrelated decision trees, Random Forest significantly reduces the model's variance without substantially increasing its bias. Each tree is trained on a random subset of the data (bootstrapping) and considers only a random subset of features for each split, leading to diverse trees.
*   **Robustness to Noise**: The averaging process makes the model more robust to noise and outliers in the training data, as individual noisy observations are less likely to influence the final prediction.
*   **Feature Importance**: Random Forest can also provide insights into feature importance, helping us understand which sensor readings are most critical for predicting equipment failure.

### Implementation Steps (Python with scikit-learn)

We will perform the following steps:

1.  **Import Libraries**: Import necessary modules from `scikit-learn` (e.g., `RandomForestRegressor`, `train_test_split`, `mean_squared_error`, `r2_score`).
2.  **Load Data**: Load the `Question1datasets.csv` into a pandas DataFrame.
3.  **Split Data**: Divide the dataset into training and testing sets using an 80/20 split. This ensures we evaluate the model on unseen data.
4.  **Fit the Model**: Train the Random Forest Regressor on the training data.
5.  **Predict**: Make predictions on the test data.
6.  **Evaluate**: Assess the model's performance using RMSE and R².




### Random Forest Results

After implementing the Random Forest Regressor, we observe the following performance:

*   **Linear Regression - Training RMSE**: 126.98
*   **Linear Regression - Training R^2**: 0.04
*   **Linear Regression - Test RMSE**: 151.46
*   **Linear Regression - Test R^2**: -0.09

*   **Random Forest - Training RMSE**: 49.96
*   **Random Forest - Training R^2**: 0.85
*   **Random Forest - Test RMSE**: 159.22
*   **Random Forest - Test R^2**: -0.20

While the Random Forest model shows significantly better performance on the training data (higher R^2 and lower RMSE), its performance on the test set is still not satisfactory, with a negative R^2. This indicates that even the Random Forest model is struggling to generalize to unseen data, possibly due to the nature of the synthetic dataset or the need for further hyperparameter tuning or feature engineering.

## 3. Model Evaluation and Cross-Validation

To more robustly assess the generalization performance of our models, especially the Random Forest Regressor, we will use k-fold cross-validation. This technique helps in getting a more reliable estimate of model performance by reducing the variability associated with a single train-test split.

### K-Fold Cross-Validation (k=5)

We will perform 5-fold cross-validation. The dataset will be divided into 5 equal folds. The model will be trained on 4 folds and tested on the remaining 1 fold. This process will be repeated 5 times, with each fold serving as the test set exactly once. The average of the performance metrics (RMSE and R²) across all folds will provide a more stable and reliable measure of the model's true performance.

### Implementation Steps

1.  **Import `KFold` and `cross_val_score`**: From `sklearn.model_selection`.
2.  **Define Cross-Validation Strategy**: Initialize `KFold` with `n_splits=5`.
3.  **Calculate Cross-Validation Scores**: Use `cross_val_score` to get RMSE and R² for both Linear Regression and Random Forest models across the folds.
4.  **Report Averages**: Present the average RMSE and R² for both models.




### Cross-Validation Results

Here are the average RMSE and R² scores from the 5-fold cross-validation:

*   **Linear Regression - Avg CV RMSE**: 134.20 (+/- 16.00)
*   **Linear Regression - Avg CV R^2**: -0.04 (+/- 0.08)

*   **Random Forest - Avg CV RMSE**: 141.67 (+/- 14.38)
*   **Random Forest - Avg CV R^2**: -0.16 (+/- 0.16)

The cross-validation results confirm that both models, including the Random Forest, are struggling to generalize effectively to unseen data. The negative R² values indicate that the models perform worse than simply predicting the mean of the target variable. This suggests that the current features might not be sufficient to capture the underlying patterns in the data, or the relationships are highly non-linear and complex, requiring more advanced modeling techniques or significant feature engineering.

## 4. Feature Engineering and Hybrid Approaches

Given the poor performance of both models, feature engineering is crucial to improve the model's ability to capture relevant patterns. We will explore normalizing features and creating interaction terms.

### Feature Engineering Proposals

1.  **Normalization**: Sensor readings like Temperature, Vibration, and Pressure often operate on different scales. Normalizing these features (e.g., using Min-Max scaling or Z-score standardization) can help algorithms that are sensitive to feature scales (though tree-based models are less sensitive, it's good practice).
2.  **Interaction Terms**: Creating new features by combining existing ones can capture more complex relationships. For example, `Vibration * Runtime` could indicate cumulative stress on the machinery, which might be a stronger predictor of failure than either feature alone.
3.  **Polynomial Features**: Introducing polynomial features (e.g., `Temperature^2`) can help capture non-linear relationships.

### Hybrid Prediction (Conceptual)

While not directly implemented in this script due to the scope, a hybrid approach could involve:

*   **Anomaly Detection with K-Means**: Before regression, K-Means clustering could be used to identify clusters of 'normal' and 'anomalous' sensor readings. Anomalous clusters might indicate impending failure or unusual operating conditions. This unsupervised step could help in two ways:
    *   **Filtering/Weighting**: Anomalous data points could be weighted differently or even filtered out for the regression model, allowing it to focus on typical failure patterns.
    *   **Categorical Feature**: The cluster assignment (e.g., 'normal', 'mild anomaly', 'severe anomaly') could be used as a new categorical feature in the supervised regression model, providing additional context.

This integration would allow the supervised algorithm (like Random Forest) to leverage insights from the unsupervised anomaly detection, potentially leading to a more robust predictive model.

### Implementation Steps for Feature Engineering

We will implement normalization and an interaction term to see their impact on the Random Forest model.




### Feature Engineering Results

After applying feature scaling and adding an interaction term (`Vibration * Runtime`), we re-evaluated the models. Here are the results:

*   **Linear Regression (FE) - Avg CV RMSE**: 133.97 (+/- 15.77)
*   **Linear Regression (FE) - Avg CV R^2**: -0.03 (+/- 0.09)

*   **Random Forest (FE) - Avg CV RMSE**: 139.38 (+/- 16.94)
*   **Random Forest (FE) - Avg CV R^2**: -0.13 (+/- 0.19)

The feature engineering steps did not lead to a significant improvement in model performance. The cross-validated R² scores remain negative, indicating that the models are still not capturing the underlying patterns in the data effectively. This suggests that the synthetic dataset might be inherently noisy or that more sophisticated feature engineering or more complex models are required.

## Conclusion

This project demonstrated the process of addressing overfitting in a predictive maintenance context. We started with a simple linear regression model that overfit the data and then moved to a more complex Random Forest model. While the Random Forest showed better performance on the training data, both models struggled to generalize to unseen data, as confirmed by k-fold cross-validation. Feature engineering, including normalization and interaction terms, did not yield significant improvements.

For future work, we recommend:

*   **Hyperparameter Tuning**: A more thorough hyperparameter search (e.g., using GridSearchCV or RandomizedSearchCV) for the Random Forest model could yield better results.
*   **More Advanced Models**: Exploring other models like Gradient Boosting (e.g., XGBoost, LightGBM) might capture more complex relationships in the data.
*   **Richer Dataset**: The synthetic dataset used here may be a limiting factor. A real-world dataset with more features and a clearer signal would likely lead to better model performance.
*   **Anomaly Detection Integration**: Fully implementing the hybrid approach with K-Means for anomaly detection could provide valuable insights and improve the regression model's accuracy.




## Code Snippets

### `predictive_maintenance.py`

```python
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the dataset
df = pd.read_csv("/home/ubuntu/upload/Question1datasets.csv")

# Define features (X) and target (y)
X = df[["Temperature", "Vibration", "Pressure", "Runtime"]]
y = df["Days to Failure"]

# --- Feature Engineering ---
# 1. Normalization (StandardScaler)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# 2. Interaction Term: Vibration * Runtime
X_scaled_df["Vibration_x_Runtime"] = X_scaled_df["Vibration"] * X_scaled_df["Runtime"]

X_fe = X_scaled_df # Use the engineered features for modeling

# Split data into training and testing sets (80/20 split) - for initial comparison
X_train, X_test, y_train, y_test = train_test_split(X_fe, y, test_size=0.2, random_state=42)

# --- Linear Regression Model (for comparison) ---
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

y_train_pred_lr = linear_model.predict(X_train)
y_test_pred_lr = linear_model.predict(X_test)

rmse_train_lr = np.sqrt(mean_squared_error(y_train, y_train_pred_lr))
r2_train_lr = r2_score(y_train, y_train_pred_lr)

rmse_test_lr = np.sqrt(mean_squared_error(y_test, y_test_pred_lr))
r2_test_lr = r2_score(y_test, y_test_pred_lr)

print(f"Linear Regression (FE) - Training RMSE: {rmse_train_lr:.2f}")
print(f"Linear Regression (FE) - Training R^2: {r2_train_lr:.2f}")
print(f"Linear Regression (FE) - Test RMSE: {rmse_test_lr:.2f}")
print(f"Linear Regression (FE) - Test R^2: {r2_test_lr:.2f}")

# --- Random Forest Regressor ---
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_train_pred_rf = rf_model.predict(X_train)
y_test_pred_rf = rf_model.predict(X_test)

rmse_train_rf = np.sqrt(mean_squared_error(y_train, y_train_pred_rf))
r2_train_rf = r2_score(y_train, y_train_pred_rf)

rmse_test_rf = np.sqrt(mean_squared_error(y_test, y_test_pred_rf))
r2_test_rf = r2_score(y_test, y_test_pred_rf)

print(f"\nRandom Forest (FE) - Training RMSE: {rmse_train_rf:.2f}")
print(f"Random Forest (FE) - Training R^2: {r2_train_rf:.2f}")
print(f"Random Forest (FE) - Test RMSE: {rmse_test_rf:.2f}")
print(f"Random Forest (FE) - Test R^2: {r2_test_rf:.2f}")

# --- K-Fold Cross-Validation (k=5) with Feature Engineered Data ---
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Define custom scorer for RMSE (cross_val_score minimizes, so we need negative RMSE)
rmse_scorer = make_scorer(mean_squared_error, greater_is_better=False)

print("\n--- Cross-Validation Results (with Feature Engineering) ---")

# Cross-validation for Linear Regression with FE
rmse_scores_lr_fe = np.sqrt(-cross_val_score(linear_model, X_fe, y, cv=kf, scoring=rmse_scorer))
r2_scores_lr_fe = cross_val_score(linear_model, X_fe, y, cv=kf, scoring=\'r2\')

print(f"Linear Regression (FE) - Avg CV RMSE: {np.mean(rmse_scores_lr_fe):.2f} (+/- {np.std(rmse_scores_lr_fe):.2f})")
print(f"Linear Regression (FE) - Avg CV R^2: {np.mean(r2_scores_lr_fe):.2f} (+/- {np.std(r2_scores_lr_fe):.2f})")

# Cross-validation for Random Forest with FE
rmse_scores_rf_fe = np.sqrt(-cross_val_score(rf_model, X_fe, y, cv=kf, scoring=rmse_scorer))
r2_scores_rf_fe = cross_val_score(rf_model, X_fe, y, cv=kf, scoring=\'r2\')

print(f"Random Forest (FE) - Avg CV RMSE: {np.mean(rmse_scores_rf_fe):.2f} (+/- {np.std(rmse_scores_rf_fe):.2f})")
print(f"Random Forest (FE) - Avg CV R^2: {np.mean(r2_scores_rf_fe):.2f} (+/- {np.std(r2_scores_rf_fe):.2f})")
```


