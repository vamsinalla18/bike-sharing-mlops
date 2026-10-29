import json
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_log_error, root_mean_squared_error
from xgboost import XGBRegressor

from src.common import dvc_md5, load_params

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("reports")
RAW_DVC_FILE = "data/raw/train.csv.dvc"


def build_models(train_cfg):
    """Build the models from params.yaml so hyperparameters are versioned."""
    seed = train_cfg["random_state"]
    cfg = train_cfg["models"]
    return {
        "LinearRegression": LinearRegression(**(cfg.get("LinearRegression") or {})),
        "RandomForest": RandomForestRegressor(
            **cfg["RandomForest"], random_state=seed, n_jobs=-1
        ),
        "XGBoost": XGBRegressor(
            **cfg["XGBoost"], objective="reg:squarederror",
            random_state=seed, n_jobs=-1,
        ),
    }


def main():
    params = load_params()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Tie every run to the exact data version it was trained on
    data_stats = json.loads(Path(params["data"]["stats_path"]).read_text())
    data_tags = {
        "data_md5": dvc_md5(RAW_DVC_FILE),
        "data_rows": str(data_stats["rows"]),
        "data_start": data_stats["date_start"],
        "data_end": data_stats["date_end"],
    }

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Bike Sharing Baseline Models")

    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    X_valid = pd.read_csv(DATA_DIR / "X_valid.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze("columns")
    y_valid = pd.read_csv(DATA_DIR / "y_valid.csv").squeeze("columns")

    models = build_models(params["train"])
    results = []

    for name, model in models.items():
        print(f"\nTraining {name}...")

        with mlflow.start_run(run_name=name):
            mlflow.set_tags(data_tags)

            model.fit(X_train, y_train)

            # Rental counts cannot be negative
            predictions = np.maximum(model.predict(X_valid), 0)

            rmse = root_mean_squared_error(y_valid, predictions)
            rmsle = np.sqrt(mean_squared_log_error(y_valid, predictions))

            mlflow.log_param("model_name", name)
            mlflow.log_params({k: str(v) for k, v in model.get_params().items()})
            mlflow.log_metric("RMSE", float(rmse))
            mlflow.log_metric("RMSLE", float(rmsle))

            model_path = MODEL_DIR / f"{name}.joblib"
            joblib.dump(model, model_path)
            mlflow.log_artifact(str(model_path), artifact_path="models")

            results.append({"Model": name, "RMSE": float(rmse), "RMSLE": float(rmsle)})
            print(f"RMSE:  {rmse:.4f}")
            print(f"RMSLE: {rmsle:.4f}")

    results_df = pd.DataFrame(results).sort_values("RMSLE")
    results_path = RESULTS_DIR / "baseline_results.csv"
    results_df.to_csv(results_path, index=False)

    # Machine-readable metrics so `dvc metrics show/diff` works across versions
    metrics = {r["Model"]: {"RMSE": r["RMSE"], "RMSLE": r["RMSLE"]} for r in results}
    best = results_df.iloc[0]
    metrics["best"] = {"model": best["Model"], "RMSLE": float(best["RMSLE"])}
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    with mlflow.start_run(run_name="Model Comparison Report"):
        mlflow.set_tags(data_tags)
        mlflow.log_artifact(str(results_path), artifact_path="reports")

    print("\n--- Final Model Comparison ---")
    print(results_df.to_string(index=False))
    print(f"\nBest model based on RMSLE: {best['Model']}")


if __name__ == "__main__":
    main()
