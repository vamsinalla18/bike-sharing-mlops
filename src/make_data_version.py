"""Create a dataset version by slicing the full source file by date.

Usage:
    python -m src.make_data_version --version v1
"""
import argparse
from pathlib import Path

import pandas as pd
import yaml

SOURCE_PATH = "data/source/train_full.csv"
OUT_PATH = "data/raw/train.csv"
VERSIONS_FILE = "data_versions.yaml"


def slice_by_end_date(df, end_date):
    """Keep rows up to and including end_date (None keeps everything)."""
    if end_date is None:
        return df.copy()
    dt = pd.to_datetime(df["datetime"])
    return df[dt <= pd.Timestamp(end_date)].copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--source", default=SOURCE_PATH)
    parser.add_argument("--out", default=OUT_PATH)
    parser.add_argument("--versions-file", default=VERSIONS_FILE)
    args = parser.parse_args()

    versions = yaml.safe_load(Path(args.versions_file).read_text())
    if args.version not in versions:
        raise SystemExit(f"Unknown version '{args.version}'. Options: {list(versions)}")

    spec = versions[args.version]
    df = pd.read_csv(args.source)
    subset = slice_by_end_date(df, spec["end_date"])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(args.out, index=False)
    print(f"{args.version} ({spec['description']}): wrote {len(subset)} of {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()
