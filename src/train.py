import numpy as np
import pandas as pd
import mlflow

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, mean_squared_log_error

from xgboost import XGBRegressor
import joblib
from pathlib import Path


# Directories
DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("reports")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# MLflow experiment setup
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Bike Sharing Baseline Models")


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

    with mlflow.start_run(run_name=name):

        # Train model
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

        # Log model parameters
        mlflow.log_param("model_name", name)

        parameters = {
            key: str(value)
            for key, value in model.get_params().items()
        }
        mlflow.log_params(parameters)

        # Log evaluation metrics
        mlflow.log_metric("RMSE", float(rmse))
        mlflow.log_metric("RMSLE", float(rmsle))

        # Save model locally
        model_path = MODEL_DIR / f"{name}.joblib"
        joblib.dump(model, model_path)

        # Log saved model as an MLflow artifact
        mlflow.log_artifact(str(model_path), artifact_path="models")

        # Collect results
        results.append({
            "Model": name,
            "RMSE": rmse,
            "RMSLE": rmsle
        })

        print(f"RMSE:  {rmse:.4f}")
        print(f"RMSLE: {rmsle:.4f}")


# Sort models by RMSLE
results_df = pd.DataFrame(results).sort_values("RMSLE")

# Save results locally
results_path = RESULTS_DIR / "baseline_results.csv"
results_df.to_csv(results_path, index=False)

# Log comparison report to MLflow
with mlflow.start_run(run_name="Model Comparison Report"):
    mlflow.log_artifact(str(results_path), artifact_path="reports")


print("\n--- Final Model Comparison ---")
print(results_df.to_string(index=False))

print("\nBest model based on RMSLE:")
print(results_df.iloc[0]["Model"])