# Bike Sharing Demand – MLOps pipeline with DVC data versioning

Predicts hourly bike rental demand (Kaggle *Bike Sharing Demand*) with a
reproducible pipeline where **every version of the dataset is tracked by DVC,
tagged in git, and linked to the models trained on it**.

```
data/source/train_full.csv   (DVC)  full history, the source of truth
        │  src/make_data_version.py  (slice by date, see data_versions.yaml)
        ▼
data/raw/train.csv           (DVC)  the current dataset version (v1 / v2 / v3)
        │
        ├─ validate  → reports/data_stats.json   (schema checks + rows/date range)
        ├─ prepare   → data/processed/*.csv      (features + chronological split)
        └─ train     → models/*.joblib, reports/metrics.json  (+ MLflow runs)
```

| File | Purpose |
|---|---|
| `dvc.yaml` | pipeline: validate → prepare → train |
| `params.yaml` | split ratio + model hyperparameters (versioned, `dvc params diff`) |
| `data_versions.yaml` | defines v1 (2011), v2 (to mid-2012), v3 (everything) |
| `scripts/release_data_version.sh` | builds, tracks, runs, commits, tags and pushes one version |

## Setup
```bash
pip install -r requirements.txt
dvc pull                      # fetch data/source/train_full.csv from the remote
```

## Create the dataset versions
```bash
scripts/release_data_version.sh v1
scripts/release_data_version.sh v2
scripts/release_data_version.sh v3
```
Each run produces a git commit + tag (`data-v1`, …) and pushes data and models to the DVC remote.

## Compare versions
```bash
dvc metrics diff data-v1 data-v3     # model scores AND data stats (rows, date range)
dvc params diff  data-v1 data-v3
git log --oneline --decorate
```

## Go back to an old version and reproduce it
```bash
git checkout data-v1
dvc checkout data/raw/train.csv.dvc   # restore that version's data
dvc repro                             # rebuilds processed data / models (cache used where possible)
git checkout main
dvc checkout data/raw/train.csv.dvc
```
> Use the explicit `dvc checkout data/raw/train.csv.dvc`. A bare `dvc checkout` removes
> `data/raw/train.csv` in this layout because the file is also a stage dependency.

## Experiments
Change `params.yaml` (e.g. `train.models.XGBoost.max_depth`) and run `dvc repro`;
compare with `dvc params diff` / `dvc metrics diff`. Every MLflow run is tagged with
`data_md5`, `data_rows`, `data_start`, `data_end` so a model can always be traced to its data
(`mlflow ui --backend-store-uri sqlite:///mlflow.db`).

## Tests / CI
`python -m pytest tests/` covers feature engineering, data slicing (versions must be strict
prefixes of each other), validation rules and param-driven model building. CI also runs `dvc dag`.
