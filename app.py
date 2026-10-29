import requests
import streamlit as st
from datetime import datetime, date, time

API_URL = "http://127.0.0.1:8001/predict"

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
