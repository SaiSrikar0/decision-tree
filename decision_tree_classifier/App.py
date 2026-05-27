"""Streamlit app to predict rain or no rain using a saved decision tree model."""
import os

import joblib
import pandas as pd
import streamlit as st


MODEL_PATH = "decision_tree_rain_model.pkl"


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


def build_input_df(temperature, humidity, wind_speed, cloud_cover, pressure):
    return pd.DataFrame(
        {
            "Temperature": [temperature],
            "Humidity": [humidity],
            "Wind_Speed": [wind_speed],
            "Cloud_Cover": [cloud_cover],
            "Pressure": [pressure],
        }
    )


st.set_page_config(page_title="Rain Prediction (Decision Tree)", layout="centered")
st.title("Rain Prediction - Decision Tree")
st.write("Predict rain or no rain from weather features.")

model = load_model()
if model is None:
    st.error(
        "Model file not found. Run decision_tree_rain_classifier.ipynb to train and save decision_tree_rain_model.pkl."
    )
    st.stop()

col1, col2 = st.columns(2)
with col1:
    temperature = st.number_input("Temperature", value=25.0)
    humidity = st.number_input("Humidity", value=70.0)
    wind_speed = st.number_input("Wind Speed", value=5.0)
with col2:
    cloud_cover = st.number_input("Cloud Cover", value=40.0)
    pressure = st.number_input("Pressure", value=1000.0)

if st.button("Predict"):
    input_df = build_input_df(temperature, humidity, wind_speed, cloud_cover, pressure)
    prediction = model.predict(input_df)[0]
    st.success(f"Prediction: {prediction}")
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_df)[0]
        classes = list(model.classes_)
        if "rain" in classes:
            rain_prob = proba[classes.index("rain")]
            st.write(f"Rain Probability: {rain_prob:.2%}")
