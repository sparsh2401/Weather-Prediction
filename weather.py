import streamlit as st
import numpy as np
import pandas as pd
import pickle
import datetime

# Load model, preprocessor, and label encoder
with open('xgboost_weather_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('weather_preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

with open('weather_label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# --- PAGE CONFIG ---
st.set_page_config(page_title="Weather Type Predictor", page_icon="🌦️", layout="centered")

st.markdown("<h1 style='text-align: center;'>🌤️ Weather Type Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Enter current weather conditions and predict the weather type!</h4>", unsafe_allow_html=True)
st.markdown("---")

# --- FEATURE INPUTS ---
with st.expander("🔍 Feature Descriptions"):
    st.markdown("""
    - **Wind Speed (km/h):** Speed of wind at surface level  
    - **Temperature (°C):** Current ambient temperature  
    - **UV Index:** Intensity of ultraviolet radiation  
    - **Precipitation (%):** Likelihood of rainfall or snow  
    - **Atmospheric Pressure (hPa):** Pressure exerted by the atmosphere  
    - **Visibility (km):** Distance at which objects can be clearly seen  
    - **Humidity (%):** Moisture content in the air  
    - **Location:** Type of area (Urban/Suburban/Rural)  
    - **Season:** Current season of the year  
    - **Cloud Cover:** Sky condition based on clouds  
    """)

col1, col2 = st.columns(2)

with col1:
    wind_speed = st.slider("🌬️ Wind Speed (km/h)", 0.0, 100.0, 10.0)
    temperature = st.slider("🌡️ Temperature (°C)", -30.0, 50.0, 25.0)
    uv_index = st.slider("☀️ UV Index", 0.0, 15.0, 5.0)
    precip = st.slider("🌧️ Precipitation (%)", 0.0, 100.0, 20.0)
    pressure = st.slider("📉 Pressure (hPa)", 900.0, 1100.0, 1013.0)

with col2:
    visibility = st.slider("👁️ Visibility (km)", 0.0, 50.0, 10.0)
    humidity = st.slider("💧 Humidity (%)", 0.0, 100.0, 60.0)
    location = st.selectbox("📍 Location", ['Urban', 'Suburban', 'Rural'])
    season = st.selectbox("📆 Season", ['Summer', 'Winter', 'Spring', 'Autumn'])
    cloud_cover = st.selectbox("☁️ Cloud Cover", ['Clear', 'Partly Cloudy', 'Overcast'])

# --- BUILD INPUT DATAFRAME ---
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

st.markdown("---")
st.subheader("📊 Input Summary")
st.dataframe(input_df)

# --- PREDICTION ---
if st.button("🚀 Predict Weather Type"):
    try:
        input_transformed = preprocessor.transform(input_df)
        prediction_encoded = model.predict(input_transformed)[0]
        prediction_label = label_encoder.inverse_transform([prediction_encoded])[0]

        st.success(f"🌈 Predicted Weather Type: **{prediction_label}**")

        # --- Option to download ---
        pred_df = input_df.copy()
        pred_df["Predicted Weather Type"] = prediction_label
        csv = pred_df.to_csv(index=False).encode('utf-8')
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        st.download_button(
            label="📥 Download Prediction as CSV",
            data=csv,
            file_name=f"weather_prediction_{now}.csv",
            mime='text/csv'
        )

    except Exception as e:
        st.error(f"⚠️ Prediction error: {e}")
