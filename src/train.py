import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, mean_squared_log_error

from xgboost import XGBRegressor
import joblib
from pathlib import Path

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("reports")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Load prepared data
X_train = pd.read_csv(DATA_DIR / "X_train.csv")
X_valid = pd.read_csv(DATA_DIR / "X_valid.csv")

y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze("columns")
y_valid = pd.read_csv(DATA_DIR / "y_valid.csv").squeeze("columns")

# Define models
models = {
    "LinearRegression": LinearRegression(),

    "RandomForest": RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBRegressor(
        n_estimators=300,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )
}

results = []

for name, model in models.items():
    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    # Predict validation data
    predictions = model.predict(X_valid)

    # Rental counts cannot be negative
    predictions = np.maximum(predictions, 0)

    # Calculate evaluation metrics
    rmse = root_mean_squared_error(y_valid, predictions)

    rmsle = np.sqrt(
        mean_squared_log_error(y_valid, predictions)
    )

    results.append({
        "Model": name,
        "RMSE": rmse,
        "RMSLE": rmsle
    })

    # Save the model
    joblib.dump(model, MODEL_DIR / f"{name}.joblib")

    print(f"RMSE:  {rmse:.4f}")
    print(f"RMSLE: {rmsle:.4f}")

# Sort models by RMSLE
results_df = pd.DataFrame(results).sort_values("RMSLE")

results_df.to_csv(RESULTS_DIR / "baseline_results.csv", index=False)

print("\n--- Final Model Comparison ---")
print(results_df.to_string(index=False))

print("\nBest model based on RMSLE:")
print(results_df.iloc[0]["Model"])
