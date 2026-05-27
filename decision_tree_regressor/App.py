"""Streamlit app to predict insurance charges using a saved decision tree model."""
import os

import joblib
import pandas as pd
import streamlit as st


MODEL_PATH = "decision_tree_insurance_model.pkl"
DATA_PATH = "insurance.csv"


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


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
    st.error(
        "Model file not found. Run decision_tree_insurance_regression.ipynb to train and save decision_tree_insurance_model.pkl."
    )
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
