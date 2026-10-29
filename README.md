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

## Prepare the serving image

The API loads `models/RandomForest.joblib` at startup. This file is not tracked
by Git. Restore it with DVC if you have access to its remote, or put a
compatible training CSV at `data/raw/train.csv` and run:

```bash
pip install -r requirements.txt
dvc repro
```

The checked-in DVC remote points to an absolute path on the original author's
machine. A fresh clone must reconfigure that remote or provide the training
data. For an infrastructure-only demo without real data, run:

```bash
pip install -r requirements.docker.txt
python -m scripts.train_demo_model
```

This creates a model trained on synthetic data. Its predictions are not real
bike-demand estimates. The Docker build fails if
`models/RandomForest.joblib` is missing.

## Run with Docker Compose

Build and start everything:

```bash
export GRAFANA_ADMIN_PASSWORD='choose-a-password'
docker compose up --build
```

If these host ports are occupied, set `API_PORT`, `FRONTEND_PORT`,
`PROMETHEUS_PORT`, or `GRAFANA_PORT` before starting Compose. Container-to-container
addresses stay unchanged.

The image installs only `requirements.docker.txt`, a small runtime dependency
list; the larger `requirements.txt` remains available for local pipeline/DVC work.

Open the Streamlit frontend at http://localhost:8501 and select **Generate 10 data
points and predictions**. The button generates 10 valid randomized inputs and sends
10 individual `POST /predict` requests to the API.

Open Prometheus at http://localhost:9090. Useful queries are:

```promql
prediction_requests_total
sum(rate(http_request_duration_seconds_count[1m]))
prediction_request_duration_seconds_bucket
```

Prometheus exposes a `HighPredictionRequestVolume` alert at
http://localhost:9090/alerts when more than 25 prediction requests arrive in five
minutes. After starting the stack, wait at least one 5-second scrape interval
before generating traffic so Prometheus records the initial counter value.

The raw API metrics are also available at http://localhost:8001/metrics. To test the
container without the frontend, in a second terminal run:

```bash
curl http://localhost:8001/health
curl http://localhost:8001/metrics | grep prediction
```

Open Grafana at http://localhost:3000, log in as `admin` with the password
above, and open **Dashboards → Bike Sharing → Bike Sharing API**. The Prometheus
data source and dashboard load automatically. Stop with `docker compose down`.

## Run on Kubernetes

Requires `kubectl`, a cluster with a default StorageClass, and an image
containing the trained model. For a local kind cluster:

```bash
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

For another cluster, push the image to a registry and replace
`bike-sharing:local` in both deployments with its full registry name and
immutable tag. With minikube, use `minikube image load bike-sharing:local` in
place of the kind load command.

In separate terminals, forward the three user interfaces:

```bash
kubectl -n bike-sharing port-forward svc/frontend 8501:8501
kubectl -n bike-sharing port-forward svc/prometheus 9090:9090
kubectl -n bike-sharing port-forward svc/grafana 3000:3000
```

Open the same local URLs as above. The Prometheus dashboard shows API
availability, request rates, and prediction latency. Generate requests from
the frontend and wait for the 5-second scrape interval. Prometheus and Grafana
use persistent claims (`2Gi` and `1Gi`); adjust them for your cluster. This
deployment runs one API replica because Prometheus scrapes the API Service as
a single target. To scale the API, change the scrape configuration to discover
and scrape each pod.
