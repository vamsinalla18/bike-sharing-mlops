"""Validate the raw data and record simple statistics about this data version.

The stats file is a DVC metrics file, so `dvc metrics diff v1 v3` shows how the
data itself changed between versions (rows, date range, ...).
"""
import json
from pathlib import Path

import pandas as pd

from src.common import load_params

REQUIRED_COLUMNS = [
    "datetime", "season", "holiday", "workingday", "weather", "temp", "atemp",
    "humidity", "windspeed", "casual", "registered", "count",
]


def validate(df):
    """Return a stats dict, or raise ValueError listing every problem found."""
    problems = []

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if df.empty:
        raise ValueError("Dataset is empty")

    if df[REQUIRED_COLUMNS].isnull().any().any():
        problems.append("contains null values")

    dt = pd.to_datetime(df["datetime"], errors="coerce")
    if dt.isnull().any():
        problems.append("unparseable datetime values")
    if dt.duplicated().any():
        problems.append("duplicate timestamps")
    if (df["count"] < 0).any():
        problems.append("negative counts")
    if not (df["casual"] + df["registered"]).equals(df["count"]):
        problems.append("casual + registered != count")

    if problems:
        raise ValueError("Data validation failed: " + "; ".join(problems))

    return {
        "rows": int(len(df)),
        "date_start": str(dt.min()),
        "date_end": str(dt.max()),
        "mean_count": round(float(df["count"].mean()), 3),
        "max_count": int(df["count"].max()),
    }


def main():
    params = load_params()
    df = pd.read_csv(params["data"]["raw_path"])
    stats = validate(df)

    out = Path(params["data"]["stats_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2))
    print("Data validation passed:", stats)


if __name__ == "__main__":
    main()
