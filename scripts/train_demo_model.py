"""Train a synthetic model for infrastructure demos, not real predictions.

Usage: python -m scripts.train_demo_model
The output is ignored by Git and must not be treated as the DVC-trained model.
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from scripts.synthetic_data import make_synthetic


def main():
    frame = make_synthetic()
    dates = pd.to_datetime(frame["datetime"])
    features = frame[
        ["season", "holiday", "workingday", "weather", "temp", "atemp", "humidity", "windspeed"]
    ].copy()
    features["hour"] = dates.dt.hour
    features["day"] = dates.dt.day
    features["month"] = dates.dt.month
    features["year"] = dates.dt.year
    features["dayofweek"] = dates.dt.day_of_week

    model = RandomForestRegressor(
        n_estimators=40, max_depth=12, random_state=0, n_jobs=-1
    )
    model.fit(features, frame["count"])
    destination = Path("models/RandomForest.joblib")
    destination.parent.mkdir(exist_ok=True)
    joblib.dump(model, destination)
    print(f"Synthetic demo model written to {destination} ({len(frame)} rows)")


if __name__ == "__main__":
    main()
