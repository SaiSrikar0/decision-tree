"""Streamlit app to predict insurance charges using a saved decision tree model."""
import os

import joblib
import pandas as pd
import streamlit as st
from sklearn.tree import DecisionTreeRegressor


# Resolve paths relative to this file so deployments find assets
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "decision_tree_insurance_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "insurance.csv")


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
    X = df[["age", "sex", "bmi", "children", "smoker", "region"]]
    X = pd.get_dummies(X, drop_first=True)
    y = df["charges"]
    reg = DecisionTreeRegressor(random_state=42)
    reg.fit(X, y)
    joblib.dump(reg, MODEL_PATH)
    return reg


def load_options():
    if not os.path.exists(DATA_PATH):
        return ["female", "male"], ["no", "yes"], ["southwest", "southeast", "northwest", "northeast"]
    df = pd.read_csv(DATA_PATH)
    sexes = sorted(df["sex"].dropna().unique().tolist())
    smokers = sorted(df["smoker"].dropna().unique().tolist())
    regions = sorted(df["region"].dropna().unique().tolist())
    return sexes, smokers, regions


def build_input_df(age, sex, bmi, children, smoker, region):
    return pd.DataFrame(
        {
            "age": [age],
            "sex": [sex],
            "bmi": [bmi],
            "children": [children],
            "smoker": [smoker],
            "region": [region],
        }
    )


st.set_page_config(page_title="Insurance Charges Predictor (Decision Tree)", layout="centered")
st.title("Insurance Charges Predictor - Decision Tree")
st.write("Predict insurance charges from customer details.")

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

sexes, smokers, regions = load_options()

col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input("Age", min_value=0, max_value=100, value=30, step=1)
    sex = st.selectbox("Sex", sexes)
with col2:
    bmi = st.number_input("BMI", min_value=10.0, max_value=80.0, value=27.0, step=0.1)
    children = st.number_input("Children", min_value=0, max_value=10, value=0, step=1)
with col3:
    smoker = st.selectbox("Smoker", smokers)
    region = st.selectbox("Region", regions)

if st.button("Predict"):
    input_df = build_input_df(age, sex, bmi, children, smoker, region)
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Charges: {prediction:.2f}")
