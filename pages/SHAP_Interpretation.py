# pages/SHAP_Interpretation.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os

# -----------------------------
# Streamlit Page Setup
# -----------------------------
st.set_page_config(page_title="🧪 SHAP Feature Interpretation", layout="wide")
st.title("🧪 SHAP Feature Interpretation")
st.markdown("Understand how the model makes predictions by analyzing feature contributions using **SHAP values**.")

# -----------------------------
# File Paths
# -----------------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

MODEL_PATH = os.path.join(MODEL_DIR, "stroke_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "feature_names.pkl")
CSV_PATH = os.path.join(DATA_DIR, "stroke_cleaned.csv")

# -----------------------------
# Load Model + Scaler + Data
# -----------------------------
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        feature_order = joblib.load(FEATURES_PATH)
        return model, scaler, feature_order
    except Exception as e:
        st.error(f"⚠️ Error loading model/scaler/features: {e}")
        return None, None, None

model, scaler, feature_order = load_artifacts()

if model is None:
    st.error("❌ Model artifacts not found. Please run `scripts/train_model.py` first.")
    st.stop()

if not os.path.exists(CSV_PATH):
    st.error(f"❌ Data file not found: {CSV_PATH}. Please ensure your cleaned data exists.")
    st.stop()

df = pd.read_csv(CSV_PATH)

# Normalize column names
df.columns = df.columns.str.lower().str.strip()

# -----------------------------
# Data Preprocessing
# -----------------------------
categorical_cols = [c for c in ['gender', 'ever_married', 'work_type', 'residence_type', 'smoking_status'] if c in df.columns]
numeric_cols = [c for c in ['age', 'avg_glucose_level', 'bmi', 'hypertension', 'heart_disease'] if c in df.columns]

if not categorical_cols and not numeric_cols:
    st.error("❌ Required columns not found in dataset. Please verify your cleaned CSV.")
    st.stop()

# Fill missing BMI safely
if "bmi" in df.columns:
    df['bmi'] = df['bmi'].fillna(df['bmi'].median())

# One-hot encode and align with training features
X = pd.get_dummies(df[numeric_cols + categorical_cols], drop_first=True)

for col in feature_order:
    if col not in X.columns:
        X[col] = 0
X = X[feature_order]

# Scale features
X_scaled = scaler.transform(X)

# -----------------------------
# SHAP Analysis
# -----------------------------
st.markdown("## 🔍 SHAP Analysis")

with st.spinner("Computing SHAP values..."):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_scaled, check_additivity=False)

# --- Plot 1: Summary (Dot) ---
st.subheader("1️⃣ SHAP Summary Plot")
st.caption("Each dot shows how a feature impacts predictions. Red = higher feature value, Blue = lower.")
fig, ax = plt.subplots(figsize=(12, 8))
shap.summary_plot(shap_values, X, show=False)
st.pyplot(fig, clear_figure=True)

# --- Plot 2: Mean Feature Importance ---
st.subheader("2️⃣ SHAP Bar Plot (Average Feature Impact)")
st.caption("Average magnitude of feature influence on predictions.")
fig2, ax2 = plt.subplots(figsize=(12, 6))
shap.summary_plot(shap_values, X, plot_type="bar", show=False)
st.pyplot(fig2, clear_figure=True)

# --- Plot 3: Interactive Force Plot ---
st.markdown("---")
st.subheader("3️⃣ Explore Individual Predictions")
st.caption("Select a specific observation to see how features push prediction ↑ (red) or ↓ (blue).")

index = st.slider("Select data index", 0, len(X_scaled) - 1, 0)

# Initialize JS for SHAP interactive visuals
shap.initjs()

# Generate SHAP force plot
force_plot = shap.force_plot(
    explainer.expected_value,
    shap_values[index],
    X.iloc[index],
    matplotlib=False
)

# Render as HTML inside Streamlit
st.components.v1.html(shap.getjs() + force_plot.html(), height=400)

# -----------------------------
# Done
# -----------------------------
st.success("✅ SHAP interpretation successfully generated.")
