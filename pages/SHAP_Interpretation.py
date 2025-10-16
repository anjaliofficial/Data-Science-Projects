# pages/SHAP_Interpretation.py (Corrected)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="🧪 SHAP Feature Interpretation", layout="wide")
st.title("🧪 SHAP Feature Interpretation (Global Analysis)")
st.markdown("""
Understand how the model makes predictions by analyzing **feature contributions** using SHAP values across the entire dataset.
""")

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

MODEL_PATH = os.path.join(MODEL_DIR, "stroke_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "feature_names.pkl")
CSV_PATH = os.path.join(DATA_DIR, "stroke_cleaned.csv")

# -----------------------------
# Helper Function for SHAP Base Value (Reused FIX for IndexError)
# -----------------------------
def get_expected_value(explainer):
    """Safely retrieves the base value (log-odds) for the positive class (index 1)."""
    ev = explainer.expected_value
    if isinstance(ev, np.ndarray):
        # If array size is 2 or more, assume index 1 is the positive class log-odds
        if ev.ndim == 1 and ev.size >= 2:
            return ev[1]
        # If array size is 1, return the first element.
        return ev[0]
    # If it's a single float, return it directly.
    return ev


# -----------------------------
# Load Model and Artifacts
# -----------------------------
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        feature_order = joblib.load(FEATURES_PATH)
        return model, scaler, feature_order
    except Exception as e:
        st.error(f"❌ Error loading model artifacts. Please run train_model.py first: {e}")
        st.stop()

model, scaler, feature_order = load_artifacts()


# -----------------------------
# Load Data & Preprocessing
# -----------------------------
if not os.path.exists(CSV_PATH):
    st.error(f"❌ Data file not found: {CSV_PATH}. Please run 'scripts/train_model.py' to generate it.")
    st.stop()

df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.lower().str.strip()

categorical_cols = ['gender', 'ever_married', 'work_type', 'residence_type', 'smoking_status']
numeric_cols = ['age', 'avg_glucose_level', 'bmi', 'hypertension', 'heart_disease']

# Data Cleaning (must match train_model.py)
if 'bmi' in df.columns:
    df['bmi'] = df['bmi'].fillna(df['bmi'].median())
df = df.dropna()
# Ensure 'gender_other' is handled correctly or dropped if not in feature_order

X = pd.get_dummies(df[numeric_cols + categorical_cols], drop_first=True)
for col in feature_order:
    if col not in X.columns:
        X[col] = 0
X = X[feature_order]


# -----------------------------
# SHAP Explainer (FIX for TypeError: Do not pass data to TreeExplainer)
# -----------------------------
@st.cache_resource
def get_explainer(_model):
    # FIX: Only pass the model. TreeExplainer is robust enough to infer the tree structure.
    # Passing the data argument here (shap.TreeExplainer(_model, X)) was causing the TypeError.
    return shap.TreeExplainer(_model)

explainer = get_explainer(model)
base_value = get_expected_value(explainer)


# -----------------------------
# Compute SHAP values
# -----------------------------
with st.spinner("Computing SHAP values for all data..."):
    # Use the one-hot encoded, aligned features (X) for SHAP computation
    shap_values_all = explainer.shap_values(X) 
    if isinstance(shap_values_all, list):
        shap_values = shap_values_all[1]  # class 1 = stroke
    else:
        shap_values = shap_values_all

# -----------------------------
# SHAP Global Plots
# -----------------------------
st.subheader("1️⃣ SHAP Summary Plot (Feature Impact & Direction)")
fig, ax = plt.subplots(figsize=(12, 8))
# Use X as the feature display names
shap.summary_plot(shap_values, X, show=False) 
st.pyplot(fig, clear_figure=True)

st.subheader("2️⃣ Feature Importance (Mean |SHAP|)")
fig2, ax2 = plt.subplots(figsize=(12, 6))
shap.summary_plot(shap_values, X, plot_type="bar", show=False)
st.pyplot(fig2, clear_figure=True)

# -----------------------------
# Individual Prediction Exploration (Force Plot)
# -----------------------------
st.markdown("---")
st.subheader("3️⃣ Individual Prediction Force Plot Exploration")
st.markdown("Select an index from the dataset to view its individual feature contributions.")
index = st.slider("Select patient index", 0, len(X)-1, 0)

# FIX: Use the safely derived base_value and ensure features is 2D (X.iloc[[index]])
shap.initjs()
force_plot = shap.plots.force(
    base_value,             
    shap_values[index],
    X.iloc[[index]],        # Ensure a 2D structure (DataFrame with one row)
    matplotlib=False
)
st.components.v1.html(force_plot.html(), height=300)

st.success("✅ SHAP interpretation loaded successfully!")