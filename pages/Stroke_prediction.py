# pages/Stroke_prediction.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Stroke Prediction", layout="centered")
st.title("🧠 Stroke Prediction App")

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")

MODEL_PATH = os.path.join(MODEL_DIR, "xgb_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "feature_names.pkl")

# -----------------------------
# Load model artifacts
# -----------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_order = joblib.load(FEATURES_PATH)
    return model, scaler, feature_order

model, scaler, feature_order = load_artifacts()

# -----------------------------
# User inputs
# -----------------------------
st.sidebar.header("Patient Information")

def user_input_features():
    gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
    age = st.sidebar.slider("Age", 0, 100, 50)
    hypertension = st.sidebar.selectbox("Hypertension", [0, 1])
    heart_disease = st.sidebar.selectbox("Heart Disease", [0, 1])
    ever_married = st.sidebar.selectbox("Ever Married", ["Yes", "No"])
    work_type = st.sidebar.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "Children", "Never_worked"])
    residence_type = st.sidebar.selectbox("Residence Type", ["Urban", "Rural"])
    avg_glucose_level = st.sidebar.number_input("Average Glucose Level", 50.0, 300.0, 100.0)
    bmi = st.sidebar.number_input("BMI", 10.0, 60.0, 25.0)
    smoking_status = st.sidebar.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes", "Unknown"])

    data = {
        "gender": gender,
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "ever_married": ever_married,
        "work_type": work_type,
        "Residence_type": residence_type,
        "avg_glucose_level": avg_glucose_level,
        "bmi": bmi,
        "smoking_status": smoking_status
    }
    return pd.DataFrame([data])

input_df = user_input_features()

# -----------------------------
# Preprocessing
# -----------------------------
# Fill missing numeric values if any
numeric_cols = ["age", "avg_glucose_level", "bmi"]
for col in numeric_cols:
    if col in input_df.columns:
        input_df[col] = input_df[col].fillna(input_df[col].median())

# One-hot encode categorical variables
categorical_cols = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
X = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)

# Align columns with training feature order
for col in feature_order:
    if col not in X.columns:
        X[col] = 0
X = X[feature_order]

# Scale numeric features
X_scaled = scaler.transform(X)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Stroke Risk"):
    pred_proba = model.predict_proba(X_scaled)[:, 1][0]
    pred_label = "High Risk" if pred_proba >= 0.5 else "Low Risk"
    st.subheader("Prediction Result")
    st.write(f"**Risk:** {pred_label}")
    st.write(f"**Probability of Stroke:** {pred_proba:.2%}")

# -----------------------------
# Optional: show user input
# -----------------------------
with st.expander("Show Input Features"):
    st.dataframe(input_df)
