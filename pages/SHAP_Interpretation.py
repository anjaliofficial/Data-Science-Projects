import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

st.title("🧪 SHAP Feature Interpretation")

# Load model artifacts
@st.cache_data
def load_artifacts():
    model = joblib.load("models/xgb_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    feature_order = joblib.load("models/feature_names.pkl")
    return model, scaler, feature_order

model, scaler, feature_order = load_artifacts()

# Load sample dataset
df = pd.read_csv("scripts/stroke_cleaned.csv")  # Make sure you have a clean CSV
df.fillna(df.median(numeric_only=True), inplace=True)

# Preprocess categorical columns
categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
df = pd.get_dummies(df, columns=categorical_cols, dummy_na=True)

# Align features
for col in feature_order:
    if col not in df.columns:
        df[col] = 0
X_scaled = scaler.transform(df[feature_order])

# SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_scaled)

st.subheader("SHAP Summary Plot")
fig, ax = plt.subplots(figsize=(10,6))
shap.summary_plot(shap_values, X_scaled, feature_names=feature_order, show=False)
st.pyplot(fig)
