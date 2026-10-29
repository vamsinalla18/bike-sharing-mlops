"""Generate a Kaggle-shaped fake dataset (for tests and for trying the workflow
without the real data). Usage: python scripts/synthetic_data.py out.csv"""
import sys

import numpy as np
import pandas as pd


def make_synthetic(start="2011-01-01", end="2012-12-19", seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, end, freq="h")
    idx = idx[idx.day <= 19]  # the real train set only has days 1-19 of each month
    hour = idx.hour.to_numpy()
    growth = 1 + 0.5 * (idx.year.to_numpy() - 2011)
    base = (40 + 120 * np.exp(-((hour - 17) ** 2) / 18)) * growth
    count = np.maximum(rng.poisson(base), 1)
    casual = (count * 0.2).astype(int)
    return pd.DataFrame({
        "datetime": idx.strftime("%Y-%m-%d %H:%M:%S"),
        "season": (idx.month.to_numpy() - 1) // 3 + 1,
        "holiday": 0,
        "workingday": (idx.dayofweek.to_numpy() < 5).astype(int),
        "weather": rng.integers(1, 4, len(idx)),
        "temp": np.round(rng.normal(20, 8, len(idx)), 2),
        "atemp": np.round(rng.normal(23, 8, len(idx)), 3),
        "humidity": rng.integers(20, 100, len(idx)),
        "windspeed": np.round(rng.uniform(0, 30, len(idx)), 4),
        "casual": casual,
        "registered": count - casual,
        "count": count,
    })


if __name__ == "__main__":
    make_synthetic().to_csv(sys.argv[1], index=False)
