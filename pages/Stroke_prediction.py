import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.title("🔍 Stroke Risk Prediction")

# Load model artifacts
@st.cache_data
def load_artifacts():
    model = joblib.load("models/xgb_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    feature_order = joblib.load("models/feature_names.pkl")
    return model, scaler, feature_order

model, scaler, feature_order = load_artifacts()

# Input form
with st.form("patient_form"):
    st.subheader("Patient Data")
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    hypertension = st.selectbox("Hypertension", [0, 1])
    heart_disease = st.selectbox("Heart Disease", [0, 1])
    avg_glucose_level = st.number_input("Average Glucose Level", min_value=50.0, max_value=500.0, value=100.0)
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    ever_married = st.selectbox("Ever Married", ["Yes", "No"])
    work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "Children", "Never_worked"])
    residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
    smoking_status = st.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes", "Unknown"])
    submitted = st.form_submit_button("Predict Stroke Risk")

if submitted:
    # Prepare dataframe
    df = pd.DataFrame([{
        'age': age, 'hypertension': hypertension, 'heart_disease': heart_disease,
        'avg_glucose_level': avg_glucose_level, 'bmi': bmi,
        'gender': gender, 'ever_married': ever_married,
        'work_type': work_type, 'Residence_type': residence_type,
        'smoking_status': smoking_status
    }])
    
    # Ensure all feature columns exist
    for col in feature_order:
        if col not in df.columns:
            df[col] = 0
    
    # Encode categorical columns
    categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    df_encoded = pd.get_dummies(df, columns=categorical_cols)
    
    # Align with training features
    for col in feature_order:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded[feature_order]
    
    # Scale numeric features
    X_scaled = scaler.transform(df_encoded)
    
    # Predict
    risk_prob = model.predict_proba(X_scaled)[:, 1][0]
    st.metric("Stroke Risk Probability", f"{risk_prob*100:.2f}%")
