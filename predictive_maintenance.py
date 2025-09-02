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
r2_scores_lr_fe = cross_val_score(linear_model, X_fe, y, cv=kf, scoring='r2')

print(f"Linear Regression (FE) - Avg CV RMSE: {np.mean(rmse_scores_lr_fe):.2f} (+/- {np.std(rmse_scores_lr_fe):.2f})")
print(f"Linear Regression (FE) - Avg CV R^2: {np.mean(r2_scores_lr_fe):.2f} (+/- {np.std(r2_scores_lr_fe):.2f})")

# Cross-validation for Random Forest with FE
rmse_scores_rf_fe = np.sqrt(-cross_val_score(rf_model, X_fe, y, cv=kf, scoring=rmse_scorer))
r2_scores_rf_fe = cross_val_score(rf_model, X_fe, y, cv=kf, scoring='r2')

print(f"Random Forest (FE) - Avg CV RMSE: {np.mean(rmse_scores_rf_fe):.2f} (+/- {np.std(rmse_scores_rf_fe):.2f})")
print(f"Random Forest (FE) - Avg CV R^2: {np.mean(r2_scores_rf_fe):.2f} (+/- {np.std(r2_scores_rf_fe):.2f})")


