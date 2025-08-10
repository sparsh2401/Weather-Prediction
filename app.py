import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load model, preprocessor, and label encoder
with open('xgboost_weather_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('weather_preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

with open('weather_label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

st.set_page_config(page_title="Weather Type Predictor", page_icon="🌦️")
st.title("🌤️ Weather Type Predictor")

# Collect user input
wind_speed = st.slider("Wind Speed (km/h)", 0.0, 100.0, 10.0)
temperature = st.slider("Temperature (°C)", -30.0, 50.0, 25.0)
uv_index = st.slider("UV Index", 0.0, 15.0, 5.0)
precip = st.slider("Precipitation (%)", 0.0, 100.0, 20.0)
pressure = st.slider("Atmospheric Pressure (hPa)", 900.0, 1100.0, 1013.0)
visibility = st.slider("Visibility (km)", 0.0, 50.0, 10.0)
humidity = st.slider("Humidity (%)", 0.0, 100.0, 60.0)

location = st.selectbox("Location", ['Urban', 'Suburban', 'Rural'])
season = st.selectbox("Season", ['Summer', 'Winter', 'Spring', 'Autumn'])
cloud_cover = st.selectbox("Cloud Cover", ['Clear', 'Partly Cloudy', 'Overcast'])

# Prepare input DataFrame
input_df = pd.DataFrame({
    'Wind Speed': [wind_speed],
    'Temperature': [temperature],
    'UV Index': [uv_index],
    'Precipitation (%)': [precip],
    'Atmospheric Pressure': [pressure],
    'Visibility (km)': [visibility],
    'Humidity': [humidity],
    'Location': [location],
    'Season': [season],
    'Cloud Cover': [cloud_cover]
})

# Predict
if st.button("Predict Weather Type"):
    try:
        input_transformed = preprocessor.transform(input_df)
        prediction_encoded = model.predict(input_transformed)[0]
        prediction_label = label_encoder.inverse_transform([prediction_encoded])[0]

        st.success(f"🌈 Predicted Weather Type: **{prediction_label}**")
    except Exception as e:
        st.error(f"⚠️ Error: {e}")
 