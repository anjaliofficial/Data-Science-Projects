import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.title("🔍 Stroke Risk Prediction")

@st.cache_data
def load_artifacts():
    model = joblib.load("models/xgb_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    feature_order = joblib.load("models/feature_names.pkl")
    return model, scaler, feature_order

model, scaler, feature_order = load_artifacts()

# User input
st.subheader("Enter patient details")
age = st.number_input("Age", min_value=0, max_value=120, value=50)
hypertension = st.selectbox("Hypertension", [0, 1])
heart_disease = st.selectbox("Heart Disease", [0, 1])
avg_glucose_level = st.number_input("Avg Glucose Level", value=100.0)
bmi = st.number_input("BMI", value=25.0)

gender = st.selectbox("Gender", ["Male","Female","Other"])
ever_married = st.selectbox("Ever Married", ["Yes","No"])
work_type = st.selectbox("Work Type", ["Private","Self-employed","Govt_job","Children","Never_worked"])
residence_type = st.selectbox("Residence Type", ["Urban","Rural"])
smoking_status = st.selectbox("Smoking Status", ["never smoked","formerly smoked","smokes","Unknown"])

# Preprocess
input_df = pd.DataFrame([{
    'age': age,
    'hypertension': hypertension,
    'heart_disease': heart_disease,
    'avg_glucose_level': avg_glucose_level,
    'bmi': bmi,
    'gender': gender,
    'ever_married': ever_married,
    'work_type': work_type,
    'Residence_type': residence_type,
    'smoking_status': smoking_status
}])

input_df = pd.get_dummies(input_df, drop_first=True)
for col in feature_order:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[feature_order]
input_scaled = scaler.transform(input_df)

# Predict
if st.button("Predict"):
    pred_prob = model.predict_proba(input_scaled)[0][1]
    st.write(f"Stroke risk probability: **{pred_prob*100:.2f}%**")
