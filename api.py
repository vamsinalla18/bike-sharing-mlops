from datetime import datetime
import time
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

# Initialize FastAPI
app = FastAPI(title="Bike Sharing Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Adds standard HTTP request count, latency, and response-size metrics at /metrics.
Instrumentator().instrument(app).expose(app)

# Business metrics specific to model predictions.
PREDICTION_REQUESTS = Counter(
    "prediction_requests_total",
    "Total prediction requests handled by the model",
    ["status"],
)
PREDICTION_LATENCY = Histogram(
    "prediction_request_duration_seconds",
    "Time spent generating a bike-rental prediction",
)

# Expose both labeled counter series immediately. This gives Prometheus a zero
# baseline at its first scrape, before any prediction requests are made.
PREDICTION_REQUESTS.labels(status="success")
PREDICTION_REQUESTS.labels(status="error")

# Load trained model
MODEL_PATH = Path(__file__).parent / "models" / "RandomForest.joblib"
model = joblib.load(MODEL_PATH)

# Features must match the model's training order
FEATURES = [
    "season",
    "holiday",
    "workingday",
    "weather",
    "temp",
    "atemp",
    "humidity",
    "windspeed",
    "hour",
    "day",
    "month",
    "year",
    "dayofweek",
]


class BikeInput(BaseModel):
    datetime: datetime
    season: int = Field(ge=1, le=4)
    holiday: int = Field(ge=0, le=1)
    workingday: int = Field(ge=0, le=1)
    weather: int = Field(ge=1, le=4)
    temp: float
    atemp: float
    humidity: float = Field(ge=0, le=100)
    windspeed: float = Field(ge=0)


@app.get("/")
def home():
    return {"message": "Bike Sharing Prediction API is running!"}


@app.get("/health")
def health():
    return {"status": "healthy", "model": "RandomForest"}


@app.post("/predict")
def predict(data: BikeInput):
    start = time.perf_counter()
    try:
        dt = data.datetime

        # Build input features
        features = {
            "season": data.season,
            "holiday": data.holiday,
            "workingday": data.workingday,
            "weather": data.weather,
            "temp": data.temp,
            "atemp": data.atemp,
            "humidity": data.humidity,
            "windspeed": data.windspeed,
            "hour": dt.hour,
            "day": dt.day,
            "month": dt.month,
            "year": dt.year,
            "dayofweek": dt.weekday(),
        }

        # Preserve the exact training feature order
        input_df = pd.DataFrame([features], columns=FEATURES)

        prediction = float(model.predict(input_df)[0])
        prediction = max(0.0, prediction)

        PREDICTION_REQUESTS.labels(status="success").inc()
        return {
            "predicted_bike_rentals": round(prediction),
            "model": "RandomForest",
        }

    except Exception as exc:
        PREDICTION_REQUESTS.labels(status="error").inc()
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        PREDICTION_LATENCY.observe(time.perf_counter() - start)
