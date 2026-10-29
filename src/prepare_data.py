import pandas as pd
from pathlib import Path

from src.common import load_params

PROCESSED_DIR = Path("data/processed")


def prepare_features(df):
    """Create features and separate the target without reading files."""
    df = df.copy()

    # Sort chronologically
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    # Feature engineering
    df["hour"] = df["datetime"].dt.hour
    df["day"] = df["datetime"].dt.day
    df["month"] = df["datetime"].dt.month
    df["year"] = df["datetime"].dt.year
    df["dayofweek"] = df["datetime"].dt.day_of_week

    # Remove target-leaking and unnecessary columns
    df = df.drop(columns=["casual", "registered", "datetime"])

    # Separate features and target
    X = df.drop(columns=["count"])
    y = df["count"]

    return X, y


def split_chronological(X, y, train_fraction):
    """Chronological split (no shuffling, so validation is strictly later)."""
    split_index = int(len(X) * train_fraction)
    return (
        X.iloc[:split_index], X.iloc[split_index:],
        y.iloc[:split_index], y.iloc[split_index:],
    )


def main():
    params = load_params()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(params["data"]["raw_path"])
    X, y = prepare_features(df)

    X_train, X_valid, y_train, y_valid = split_chronological(
        X, y, params["prepare"]["train_fraction"]
    )

    X_train.to_csv(PROCESSED_DIR / "X_train.csv", index=False)
    X_valid.to_csv(PROCESSED_DIR / "X_valid.csv", index=False)
    y_train.to_csv(PROCESSED_DIR / "y_train.csv", index=False)
    y_valid.to_csv(PROCESSED_DIR / "y_valid.csv", index=False)

    print("Data preparation completed!")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_valid)}")
    print(f"Number of features: {X_train.shape[1]}")


if __name__ == "__main__":
    main()
