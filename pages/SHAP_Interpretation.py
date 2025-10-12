import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="🧪 SHAP Feature Interpretation", layout="wide")
st.title("🧪 SHAP Feature Interpretation")
st.markdown("Understand how the XGBoost model makes predictions by analyzing feature contributions.")

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

MODEL_PATH = os.path.join(MODEL_DIR, "xgb_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "feature_names.pkl")
CSV_PATH = os.path.join(DATA_DIR, "stroke_cleaned.csv") # Ensure your cleaned CSV is here

# -----------------------------
# Load artifacts and Data
# -----------------------------
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        feature_order = joblib.load(FEATURES_PATH)
        return model, scaler, feature_order
    except FileNotFoundError:
        return None, None, None

model, scaler, feature_order = load_artifacts()

if model is None:
    st.error("Model artifacts not found. Please run the training script first.")
    st.stop()

if not os.path.exists(CSV_PATH):
    st.error(f"Data file not found: {CSV_PATH}. SHAP analysis requires the training data.")
    st.stop()

df = pd.read_csv(CSV_PATH)
# Fill missing BMI
df['bmi'].fillna(df['bmi'].median(), inplace=True)

# -----------------------------
# Preprocess dataset to match training
# -----------------------------
categorical_cols = ['gender','ever_married','work_type','Residence_type','smoking_status']
numeric_cols = ['age','avg_glucose_level','bmi','hypertension','heart_disease']

# Select features used in training and one-hot encode
X = pd.get_dummies(df[numeric_cols + categorical_cols], drop_first=True)

# Align columns to model's feature order
for col in feature_order:
    if col not in X.columns:
        X[col] = 0
X = X[feature_order]

# Scale features
X_scaled = scaler.transform(X)

# -----------------------------
# SHAP Analysis
# -----------------------------
st.subheader("1. SHAP Summary Plot (Feature Impact & Direction)")
st.caption("Dots show the impact of each feature value on the model output. Red means higher feature value.")

# Calculate SHAP values
explainer = shap.Explainer(model, X_scaled)
shap_values = explainer(X_scaled)

# SHAP summary plot
fig, ax = plt.subplots(figsize=(12, 8))
shap.summary_plot(shap_values, X_scaled, feature_names=feature_order, show=False)
st.pyplot(fig)

st.markdown("---")
st.subheader("2. SHAP Bar Plot (Mean Absolute Feature Importance)")
st.caption("The average magnitude of the feature's impact across the entire dataset.")
fig2, ax2 = plt.subplots(figsize=(12, 6))
shap.plots.bar(shap_values, max_display=15, show=False)
st.pyplot(fig2)

st.markdown("---")
st.subheader("3. Individual Prediction SHAP Force Plot")
st.caption("Select a data point to see which features push the prediction higher (red) or lower (blue) than the base value.")

index = st.slider("Select observation index", 0, X_scaled.shape[0]-1, 0)
shap.initjs()
force_plot_html = shap.plots.force(
    shap_values[index], 
    matplotlib=False, 
    feature_names=feature_order
)
st.components.v1.html(force_plot_html.html(), height=400)