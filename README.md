# Bike Sharing Demand – MLOps Pipeline with DVC Data Versioning

Predicts hourly bike rental demand (Kaggle *Bike Sharing Demand*) with a reproducible pipeline where **every version of the dataset is tracked by DVC, tagged in Git, and linked to the models trained on it**.
```
data/source/train_full.csv   (DVC)  full history, the source of truth
        │  src/make_data_version.py  (slice by date, see data_versions.yaml)
        ▼
data/raw/train.csv           (DVC)  the current dataset version (v1 / v2 / v3)
        │
        ├─ validate  → reports/data_stats.json
        ├─ prepare   → data/processed/*.csv
        └─ train     → models/*.joblib, reports/metrics.json (+ MLflow runs)
```

| File                              | Purpose                                                           |
| --------------------------------- | ----------------------------------------------------------------- |
| `dvc.yaml`                        | Pipeline: validate → prepare → train                              |
| `params.yaml`                     | Split ratio and model hyperparameters                             |
| `data_versions.yaml`              | Defines v1, v2, and v3 dataset versions                           |
| `scripts/release_data_version.sh` | Builds, tracks, runs, commits, tags, and pushes a dataset version |

## Setup
```
pip install -r requirements.txt
dvc pull
```

## Create Dataset Versions
```
scripts/release_data_version.sh v1
scripts/release_data_version.sh v2
scripts/release_data_version.sh v3
```

Each run produces a Git commit and tag (`data-v1`, etc.) and pushes data and models to the DVC remote.

## Compare Dataset Versions
```
dvc metrics diff data-v1 data-v3
dvc params diff data-v1 data-v3
git log --oneline --decorate
```

## Restore an Older Version
```
git checkout data-v1
dvc checkout data/raw/train.csv.dvc
dvc repro
git checkout main
dvc checkout data/raw/train.csv.dvc
```

Use the explicit `dvc checkout data/raw/train.csv.dvc` command for this project layout.

## MLflow Experiment Tracking

Change model hyperparameters in `params.yaml` and run:
```
dvc repro
dvc params diff
dvc metrics diff
```

MLflow tracks model parameters, RMSE/RMSLE metrics, model artifacts, and dataset metadata such as `data_md5`, `data_rows`, `data_start`, and `data_end`.

Start the MLflow UI with:
```
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Tests and CI

Run the tests locally:
```
python -m pytest tests/ -v
```

The GitHub Actions workflow runs the tests and checks the DVC pipeline definition using `dvc dag`.

## Machine-Specific Paths: Jenkins and DVC

**Important:** The project currently contains machine-specific paths. Update them before running the project on another computer or Jenkins agent.

### 1. Jenkinsfile

Open `Jenkinsfile` and update the following environment variables:
```
environment {
    PYTHON = '/path/to/python3.12'
    PROJECT_DIR = '/path/to/bike-sharing-mlops'
}
```

- `PYTHON`: Path to the Python 3.12 executable on your machine.
- `PROJECT_DIR`: Absolute path to your local project directory.

The current Jenkins pipeline copies project files from `PROJECT_DIR` into the Jenkins workspace. Ensure the paths exist and are accessible to the Jenkins process.

### 2. DVC Remote Configuration

Open `.dvc/config`. The current configuration uses a local filesystem path:
```
[core]
    remote = localstore

['remote "localstore"']
    url = /path/to/dvc-storage/bike-sharing
```

Replace the example path with the actual DVC storage directory available on your machine.

If you want multiple machines to share the dataset, configure a shared DVC remote, such as Google Drive or Amazon S3, and ensure the required credentials and permissions are available.

**Note:** Updating the path only makes DVC point to the intended storage location. The required dataset and DVC cache must also be available there.

### 3. Workflow Separation

Jenkins and GitHub Actions/DVC are separate workflows in this project:

- **Jenkins + MLflow:** Runs tests, prepares data, trains models, tracks experiments, and verifies generated artifacts.
- **GitHub Actions + DVC:** Runs tests and checks the DVC pipeline definition.

## Prepare the Serving Model

The API loads `models/RandomForest.joblib` at startup. This file is not tracked by Git.

If you have access to the configured DVC remote, restore the required data and model artifacts. Otherwise, provide a compatible training dataset and run the training pipeline.
```
pip install -r requirements.txt
dvc repro
```

For an infrastructure-only demonstration without real data:
```
pip install -r requirements.docker.txt
python -m scripts.train_demo_model
```

This creates a model trained on synthetic data. Its predictions are not real bike-demand estimates.

## Run with Docker Compose

Set a Grafana password and start the services:
```
export GRAFANA_ADMIN_PASSWORD='choose-a-password'
docker compose up --build
```

If the default host ports are occupied, configure `API_PORT`, `FRONTEND_PORT`, `PROMETHEUS_PORT`, or `GRAFANA_PORT`.

- Streamlit frontend: `http://localhost:8501`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000`
- API metrics: `http://localhost:8001/metrics`

The Streamlit frontend can generate sample inputs and send prediction requests to the API. Prometheus collects metrics, and Grafana displays the monitoring dashboard.

Stop the stack with:
```
docker compose down
```

## Run on Kubernetes

Requires `kubectl`, a Kubernetes cluster with a default StorageClass, and a container image containing the trained model.

For a local kind cluster:
```
docker build -t bike-sharing:local .
kind load docker-image bike-sharing:local
kubectl apply -f k8s/namespace.yaml

export GRAFANA_ADMIN_PASSWORD='choose-a-password'

kubectl -n bike-sharing create secret generic grafana-admin \
  --from-literal=password="$GRAFANA_ADMIN_PASSWORD"

kubectl apply -k k8s/

kubectl -n bike-sharing rollout status deployment/api
kubectl -n bike-sharing rollout status deployment/frontend
kubectl -n bike-sharing rollout status deployment/prometheus
kubectl -n bike-sharing rollout status deployment/grafana
```

For another cluster, push the image to a container registry and update the image reference in the Kubernetes deployment files.

Forward the services to access their interfaces:
```
kubectl -n bike-sharing port-forward svc/frontend 8501:8501
kubectl -n bike-sharing port-forward svc/prometheus 9090:9090
kubectl -n bike-sharing port-forward svc/grafana 3000:3000
```

The monitoring dashboard displays API availability, request rates, and prediction latency.

[Watch the Kubernetes end-to-end demo](https://github.com/vamsinalla18/bike-sharing-mlops/blob/main/docs/kubernetes-demo.mp4). It uses the synthetic demo model described above.
