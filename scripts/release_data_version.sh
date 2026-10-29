#!/usr/bin/env bash
# Create, track, run, commit, tag and push one dataset version.
# Usage: scripts/release_data_version.sh v1
set -euo pipefail

VERSION="${1:?usage: $0 <version, e.g. v1>}"

echo ">> Building dataset $VERSION"
python -m src.make_data_version --version "$VERSION"

echo ">> Tracking data with DVC"
dvc add data/raw/train.csv

echo ">> Running pipeline"
dvc repro

echo ">> Committing"
git add data/raw/train.csv.dvc dvc.lock \
        reports/metrics.json reports/data_stats.json reports/baseline_results.csv
git commit -m "Data $VERSION: pipeline run and metrics"
git tag -a "data-$VERSION" -m "Dataset $VERSION"

echo ">> Pushing data and models to the DVC remote"
dvc push

echo "Done. Compare versions with: dvc metrics diff data-v1 data-$VERSION"
