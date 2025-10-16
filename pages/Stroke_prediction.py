import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Stroke Prediction & SHAP Explanation", layout="centered")
st.title("🧠 Stroke Prediction App with SHAP Interpretation")

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
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        feature_order = joblib.load(FEATURES_PATH)
        return model, scaler, feature_order
    except FileNotFoundError:
        st.error("Model artifacts not found. Place model, scaler, feature_names in 'models'.")
        st.stop()
        
model, scaler, feature_order = load_artifacts()

# -----------------------------
# User Inputs
# -----------------------------
st.sidebar.header("Patient Information")

def user_input_features():
    gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
    age = st.sidebar.slider("Age", 0, 100, 50)
    hypertension = st.sidebar.selectbox("Hypertension (0=No, 1=Yes)", [0, 1])
    heart_disease = st.sidebar.selectbox("Heart Disease (0=No, 1=Yes)", [0, 1])
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
numeric_cols = ["age", "avg_glucose_level", "bmi"]
categorical_cols = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]

# Fill numeric missing (robustness)
for col in numeric_cols:
    input_df[col] = input_df[col].fillna(input_df[col].median())

# One-hot encode
X = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)

# Align features with training
for col in feature_order:
    if col not in X.columns:
        X[col] = 0
X = X[feature_order]

# Scale
X_scaled = scaler.transform(X)

# -----------------------------
# Prediction
# -----------------------------
st.markdown("---")
if st.button("Predict Stroke Risk", type="primary"):
    pred_proba = model.predict_proba(X_scaled)[:, 1][0]
    st.subheader("Prediction Result")
    if pred_proba >= 0.5:
        st.error(f"🔴 High Risk: Probability of Stroke is {pred_proba:.2%}")
    else:
        st.success(f"🟢 Low Risk: Probability of Stroke is {pred_proba:.2%}")

    # -----------------------------
    # SHAP Interpretation
    # -----------------------------
    st.markdown("### 🧪 SHAP Feature Interpretation")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)

    # Force plot for individual prediction
    st.subheader("Individual Prediction Force Plot")
    shap.initjs()
    force_html = shap.force_plot(explainer.expected_value, shap_values, X, matplotlib=False)
    st.components.v1.html(force_html.html(), height=400)

    # SHAP summary plot (bar)
    st.subheader("Feature Importance (Mean |SHAP|)")
    fig, ax = plt.subplots(figsize=(10, 5))
    shap.summary_plot(shap_values, X, feature_names=feature_order, plot_type="bar", show=False)
    st.pyplot(fig)

# -----------------------------
# Optional: Show input
# -----------------------------
with st.expander("Show Input Features"):
    st.dataframe(input_df)
