from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Initialize FastAPI
app = FastAPI(title="Bike Sharing Prediction API")

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

        return {
            "predicted_bike_rentals": round(prediction),
            "model": "RandomForest",
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
