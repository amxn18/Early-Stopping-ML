# xgboost_early_stopping.py

# ✅ Example of using Early Stopping in XGBoost

from xgboost import XGBRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Step 1: Create a regression dataset
X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)

# Step 2: Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Initialize the XGBoost Regressor
model = XGBRegressor(
    n_estimators=1000,         # Try up to 1000 trees
    learning_rate=0.1,
    objective='reg:squarederror',
    random_state=42
)

# Step 4: Fit the model with early stopping
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],      # Validation set for monitoring
    eval_metric='rmse',             # Metric to watch
    early_stopping_rounds=10,       # Stop if no improvement in 10 rounds
    verbose=True
)

# Step 5: Make predictions and evaluate
y_pred = model.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
print(f"Validation MSE: {mse:.4f}")
