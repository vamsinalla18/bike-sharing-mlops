pipeline {
    agent any

    environment {
        PYTHON = '/Users/vamsinalla/.local/bin/python3.12'
        PROJECT_DIR = '/Users/vamsinalla/Documents/bike-sharing-mlops'
    }

    stages {

        stage('Checkout Project Files') {
            steps {
                sh '''
                    set -e

                    echo "Copying project files into Jenkins workspace..."

                    test -f "$PROJECT_DIR/requirements.txt"
                    test -f "$PROJECT_DIR/params.yaml"
                    test -d "$PROJECT_DIR/src"
                    test -d "$PROJECT_DIR/tests"

                    cp "$PROJECT_DIR/requirements.txt" .
                    cp "$PROJECT_DIR/params.yaml" .

                    rm -rf src tests scripts
                    cp -R "$PROJECT_DIR/src" .
                    cp -R "$PROJECT_DIR/tests" .

                    if [ -d "$PROJECT_DIR/scripts" ]; then
                        cp -R "$PROJECT_DIR/scripts" .
                    fi

                    mkdir -p data/raw data/processed models reports

                    if [ -d "$PROJECT_DIR/data/raw" ]; then
                        cp -R "$PROJECT_DIR/data/raw/." data/raw/
                    fi

                    echo "Project files copied."
                    echo "Workspace: $WORKSPACE"
                    ls -la
                '''
            }
        }

        stage('Check Python') {
            steps {
                sh '''
                    set -e
                    "$PYTHON" --version
                '''
            }
        }

        stage('Create Virtual Environment') {
            steps {
                sh '''
                    set -e
                    "$PYTHON" -m venv "$WORKSPACE/.venv"
                    "$WORKSPACE/.venv/bin/python" --version
                '''
            }
        }

        stage('Install Dependencies') {
            steps {
                sh '''
                    set -e

                    VENV_PYTHON="$WORKSPACE/.venv/bin/python"

                    "$VENV_PYTHON" -m pip install --upgrade pip
                    "$VENV_PYTHON" -m pip install -r requirements.txt
                    "$VENV_PYTHON" -m pip install pytest mlflow

                    echo "Dependencies installed successfully."
                '''
            }
        }

        stage('Check Dataset') {
            steps {
                sh '''
                    set -e

                    if [ ! -f data/raw/train.csv ]; then
                        if [ -f "$PROJECT_DIR/data/raw/train.csv" ]; then
                            mkdir -p data/raw
                            cp "$PROJECT_DIR/data/raw/train.csv" data/raw/train.csv
                            echo "Dataset copied into Jenkins workspace."
                        else
                            echo "ERROR: Dataset not found."
                            echo "Checked workspace and $PROJECT_DIR/data/raw/train.csv"
                            exit 1
                        fi
                    fi

                    echo "Dataset found:"
                    ls -lh data/raw/train.csv

                    if [ -f data/raw/train.csv.dvc ]; then
                        echo "DVC metadata found."
                    else
                        echo "WARNING: data/raw/train.csv.dvc is missing."
                        echo "The training script may use 'unknown' for data_md5."
                    fi
                '''
            }
        }

        stage('Run Tests') {
            steps {
                sh '''
                    set -e

                    export PYTHONPATH="$WORKSPACE"
                    "$WORKSPACE/.venv/bin/python" -m pytest tests/ -v
                '''
            }
        }

        stage('Prepare Data') {
            steps {
                sh '''
                    set -e

                    export PYTHONPATH="$WORKSPACE"

                    echo "Preparing data..."
                    "$WORKSPACE/.venv/bin/python" src/prepare_data.py

                    echo "Checking generated processed data..."
                    test -f data/processed/X_train.csv
                    test -f data/processed/X_valid.csv
                    test -f data/processed/y_train.csv
                    test -f data/processed/y_valid.csv

                    echo "Processed datasets verified."
                '''
            }
        }

        stage('Generate Data Statistics') {
            steps {
                sh '''
                    set -e

                    echo "Generating data statistics..."

                    "$WORKSPACE/.venv/bin/python" - <<'PYTHON'
import json
from pathlib import Path
import pandas as pd

raw_path = Path("data/raw/train.csv")
stats_path = Path("reports/data_stats.json")

if not raw_path.exists():
    raise FileNotFoundError(f"Dataset not found: {raw_path}")

df = pd.read_csv(raw_path)

if "datetime" not in df.columns:
    raise ValueError("Dataset is missing the datetime column")

dates = pd.to_datetime(df["datetime"], errors="raise")

stats = {
    "rows": int(len(df)),
    "date_start": str(dates.min()),
    "date_end": str(dates.max())
}

stats_path.parent.mkdir(parents=True, exist_ok=True)
stats_path.write_text(json.dumps(stats, indent=2))

print("Data statistics generated:")
print(json.dumps(stats, indent=2))
PYTHON

                    echo "Verifying statistics file..."
                    test -f reports/data_stats.json
                    cat reports/data_stats.json
                '''
            }
        }

        stage('Train Models with MLflow') {
            steps {
                sh '''
                    set -e

                    export PYTHONPATH="$WORKSPACE"

                    echo "Checking required files..."
                    test -f params.yaml
                    test -f reports/data_stats.json
                    test -f data/raw/train.csv.dvc || echo "WARNING: DVC metadata is missing."

                    echo "Training models with MLflow..."
                    "$WORKSPACE/.venv/bin/python" src/train.py
                '''
            }
        }

        stage('Verify Artifacts') {
            steps {
                sh '''
                    set -e

                    echo "Checking trained models..."
                    test -f models/RandomForest.joblib
                    test -f models/XGBoost.joblib
                    test -f models/LinearRegression.joblib

                    echo "Checking model comparison report..."
                    test -f reports/baseline_results.csv
                    test -f reports/metrics.json
                    test -f reports/data_stats.json

                    echo "Trained models:"
                    ls -lh models/

                    echo "Reports:"
                    ls -lh reports/

                    echo "All required artifacts verified."
                '''
            }
        }
    }

    post {
        success {
            echo 'Jenkins ML pipeline completed successfully!'
        }

        failure {
            echo 'Pipeline failed. Check the failed stage in Console Output.'
        }

        always {
            echo 'Jenkins pipeline execution finished.'
        }
    }
}