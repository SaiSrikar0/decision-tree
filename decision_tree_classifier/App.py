"""Streamlit app to predict rain or no rain using a saved decision tree model."""
import os

import joblib
import pandas as pd
import streamlit as st
from sklearn.tree import DecisionTreeClassifier


# Resolve paths relative to this file so deployments find assets
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "decision_tree_rain_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "weather_forecast_data.csv")


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        return None


def train_and_save_model():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Training data not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    X = df[["Temperature", "Humidity", "Wind_Speed", "Cloud_Cover", "Pressure"]]
    y = df["Rain"]
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X, y)
    joblib.dump(clf, MODEL_PATH)
    return clf


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
    st.warning("Model file not found in the app bundle.")
    col_train, col_skip = st.columns([1, 1])
    with col_train:
        if st.button("Train model now using bundled CSV"):
            with st.spinner("Training model..."):
                try:
                    model = train_and_save_model()
                    st.success("Model trained and saved.")
                except Exception as e:
                    st.error(f"Training failed: {e}")
                    st.stop()
    with col_skip:
        if st.button("Stop (no model)"):
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
