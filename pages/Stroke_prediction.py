import streamlit as st
import pandas as pd
import joblib

st.title("🔍 Stroke Risk Prediction")

# Load artifacts
model = joblib.load("models/xgb_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_order = joblib.load("models/feature_names.pkl")

with st.form("stroke_form"):
    age = st.number_input("Age", 0, 120, 45)
    bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
    avg_glucose_level = st.number_input("Average Glucose Level", 40.0, 300.0, 100.0)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    ever_married = st.selectbox("Ever Married", ["No", "Yes"])
    hypertension = st.selectbox("Hypertension", ["No", "Yes"])
    heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
    work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "Children", "Never_worked"])
    Residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
    smoking_status = st.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes", "Unknown"])

    submitted = st.form_submit_button("Predict Stroke Risk")

if submitted:
    input_data = pd.DataFrame([{
        'age': age,
        'hypertension': 1 if hypertension=="Yes" else 0,
        'heart_disease': 1 if heart_disease=="Yes" else 0,
        'bmi': bmi,
        'avg_glucose_level': avg_glucose_level,
        'gender': 0 if gender=="Male" else (1 if gender=="Female" else 2),
        'ever_married': 1 if ever_married=="Yes" else 0,
        'work_type': {"Private":0, "Self-employed":1, "Govt_job":2, "Children":3, "Never_worked":4}[work_type],
        'Residence_type': 0 if Residence_type=="Urban" else 1,
        'smoking_status': {"formerly smoked":0, "never smoked":1, "smokes":2, "Unknown":3}[smoking_status]
    }])

    input_data = input_data.reindex(columns=feature_order)
    input_scaled = scaler.transform(input_data)
    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    st.subheader("Prediction Results")
    st.write(f"**Probability of Stroke:** {prob:.2%}")
    st.write(f"**Predicted Outcome:** {'🟥 High Risk' if pred==1 else '🟩 Low Risk'}")
