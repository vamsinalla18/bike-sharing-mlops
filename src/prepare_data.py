import pandas as pd
from pathlib import Path

RAW_PATH = "data/raw/train.csv"
PROCESSED_DIR = Path("data/processed")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Load and sort the dataset chronologically
df = pd.read_csv(RAW_PATH)
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

# Chronological 80/20 split
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_valid = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_valid = y.iloc[split_index:]

# Save the datasets
X_train.to_csv(PROCESSED_DIR / "X_train.csv", index=False)
X_valid.to_csv(PROCESSED_DIR / "X_valid.csv", index=False)
y_train.to_csv(PROCESSED_DIR / "y_train.csv", index=False)
y_valid.to_csv(PROCESSED_DIR / "y_valid.csv", index=False)

print("Data preparation completed!")
print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_valid)}")
print(f"Number of features: {X_train.shape[1]}")
print(f"Training period: first {split_index} observations")
print(f"Validation period: last {len(df) - split_index} observations")
print("\nFeature columns:")
print(X_train.columns.tolist())
