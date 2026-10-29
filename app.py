import requests
import streamlit as st
from datetime import datetime, date, time
import os
import random

API_URL = os.getenv("API_URL", "http://127.0.0.1:8001/predict")

st.set_page_config(
    page_title="Bike Rental Predictor",
    page_icon="🚲",
    layout="centered",
)

st.title("🚲 Bike Rental Prediction")
st.write("Predict bike rentals using our trained Machine Learning model.")

st.divider()

st.subheader("📅 Date and Time")

selected_date = st.date_input("Select date", value=date(2012, 6, 15))
selected_time = st.time_input("Select time", value=time(8, 0))

st.subheader("🌦️ Weather Conditions")

season = st.selectbox(
    "Season",
    options=[1, 2, 3, 4],
    format_func=lambda x: {
        1: "Spring",
        2: "Summer",
        3: "Fall",
        4: "Winter",
    }[x],
)

weather = st.selectbox(
    "Weather",
    options=[1, 2, 3, 4],
    format_func=lambda x: {
        1: "Clear / Few clouds",
        2: "Mist / Cloudy",
        3: "Light rain / Light snow",
        4: "Heavy rain / Heavy snow",
    }[x],
)

temp = st.slider("Temperature", 0.0, 41.0, 25.0)
atemp = st.slider("Feels-like temperature", 0.0, 50.0, 27.0)
humidity = st.slider("Humidity (%)", 0, 100, 60)
windspeed = st.number_input(
    "Windspeed",
    min_value=0.0,
    value=10.0,
)

st.subheader("📋 Additional Information")

holiday = st.selectbox("Is it a holiday?", [0, 1],
                       format_func=lambda x: "Yes" if x else "No")

workingday = st.selectbox(
    "Is it a working day?",
    [0, 1],
    format_func=lambda x: "Yes" if x else "No",
)

if st.button("Predict Bike Rentals", type="primary", use_container_width=True):
    date_time = datetime.combine(selected_date, selected_time)

    payload = {
        "datetime": date_time.isoformat(),
        "season": season,
        "holiday": holiday,
        "workingday": workingday,
        "weather": weather,
        "temp": temp,
        "atemp": atemp,
        "humidity": humidity,
        "windspeed": windspeed,
    }

    try:
        with st.spinner("Predicting bike rentals..."):
            response = requests.post(API_URL, json=payload, timeout=30)

        if response.ok:
            result = response.json()
            prediction = result["predicted_bike_rentals"]

            st.success("Prediction completed!")
            st.metric("Predicted Bike Rentals", f"{prediction:,} bikes")
            st.caption(f"Model used: {result['model']}")
        else:
            st.error(f"API error ({response.status_code}): {response.text}")

    except requests.RequestException:
        st.error(
            "Could not connect to the backend. "
            "Ensure the FastAPI server is running on port 8001."
        )


def random_payload() -> dict:
    """Create one valid, randomized request body for the prediction API."""
    generated_datetime = datetime(
        year=random.choice([2011, 2012]),
        month=random.randint(1, 12),
        day=random.randint(1, 28),
        hour=random.randint(0, 23),
    )
    return {
        "datetime": generated_datetime.isoformat(),
        "season": random.randint(1, 4),
        "holiday": random.randint(0, 1),
        "workingday": random.randint(0, 1),
        "weather": random.randint(1, 4),
        "temp": round(random.uniform(0, 41), 1),
        "atemp": round(random.uniform(0, 50), 1),
        "humidity": random.randint(0, 100),
        "windspeed": round(random.uniform(0, 45), 1),
    }


st.divider()
st.subheader("📈 Prometheus load test")
st.write(
    "Generate 10 random bike inputs and send 10 individual prediction requests. "
    "Check Prometheus afterwards to see the request count and latency change."
)

if st.button("Generate 10 data points and predictions", use_container_width=True):
    results = []
    progress = st.progress(0, text="Sending 10 prediction requests...")
    try:
        for index in range(10):
            response = requests.post(API_URL, json=random_payload(), timeout=30)
            response.raise_for_status()
            results.append(response.json()["predicted_bike_rentals"])
            progress.progress(index + 1, text=f"Completed {index + 1} of 10 requests")

        st.success("Sent 10 individual requests successfully.")
        st.dataframe(
            {"request_number": range(1, 11), "predicted_bike_rentals": results},
            hide_index=True,
            use_container_width=True,
        )
    except requests.RequestException as exc:
        st.error(f"Request failed: {exc}")
