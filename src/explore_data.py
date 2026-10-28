import pandas as pd

DATA_PATH = "data/raw/train.csv"

df = pd.read_csv(DATA_PATH)

# 1. Dataset shape
print("\n--- Dataset Shape ---")
print(df.shape)

# 2. First five rows
print("\n--- First Five Rows ---")
print(df.head())

# 3. Column information
print("\n--- Column Information ---")
df.info()

# 4. Missing values
print("\n--- Missing Values ---")
print(df.isnull().sum())

# 5. Duplicate rows
print("\n--- Duplicate Rows ---")
print(df.duplicated().sum())

# 6. Convert datetime
df["datetime"] = pd.to_datetime(df["datetime"])

print("\n--- Date Range ---")
print("Start:", df["datetime"].min())
print("End:", df["datetime"].max())

# 7. Target statistics
print("\n--- Target Statistics ---")
print(df["count"].describe())

# 8. Target leakage check
print("\n--- Target Leakage Check ---")
print("casual + registered equals count:",
      (df["casual"] + df["registered"]).equals(df["count"]))

# 9. Sort chronologically
df = df.sort_values("datetime").reset_index(drop=True)

print("\n--- Earliest Records ---")
print(df[["datetime", "count"]].head())

print("\n--- Latest Records ---")
print(df[["datetime", "count"]].tail())
