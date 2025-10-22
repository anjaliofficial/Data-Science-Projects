import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os
import streamlit.components.v1 as components

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
# Helper Function for SHAP Base Value
# -----------------------------
def get_expected_value(explainer):
    """Safely retrieves the base value (log-odds) for the positive class (index 1)."""
    ev = explainer.expected_value
    if isinstance(ev, np.ndarray):
        if ev.ndim == 1 and ev.size >= 2:
            return ev[1]
        return ev[0]
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
        st.error(f"❌ Error loading model artifacts. Please ensure 'scripts/train_model.py' has been run successfully: {e}")
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

# One-hot encode and align features
X = pd.get_dummies(df[numeric_cols + categorical_cols], drop_first=True)
for col in feature_order:
    if col not in X.columns:
        X[col] = 0
X = X[feature_order]  # ensure correct order


# -----------------------------
# SHAP Explainer
# -----------------------------
@st.cache_resource
def get_explainer(_model):
    return shap.TreeExplainer(_model)

explainer = get_explainer(model)
base_value = get_expected_value(explainer)


# -----------------------------
# Compute SHAP values
# -----------------------------
sample_size = min(2000, len(X))
X_sample = shap.sample(X, sample_size)

with st.spinner(f"Computing SHAP values for {sample_size} samples..."):
    shap_values_all = explainer.shap_values(X_sample)

    # ✅ Handle both binary and single-output models robustly
    if isinstance(shap_values_all, list):
        if len(shap_values_all) > 1:
            shap_values = shap_values_all[1]  # class 1 = stroke
        else:
            shap_values = shap_values_all[0]
    else:
        shap_values = shap_values_all

# -----------------------------
# SHAP Global Plots
# -----------------------------
st.subheader("1️⃣ SHAP Summary Plot (Feature Impact & Direction)")
fig, ax = plt.subplots(figsize=(12, 8))
shap.summary_plot(shap_values, X_sample, show=False)
st.pyplot(fig, clear_figure=True)

st.subheader("2️⃣ Feature Importance (Mean |SHAP|)")
fig2, ax2 = plt.subplots(figsize=(12, 6))
shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
st.pyplot(fig2, clear_figure=True)


# -----------------------------
# Individual Prediction Exploration (Force Plot)
# -----------------------------
st.markdown("---")
st.subheader("3️⃣ Individual Prediction Force Plot Exploration")
st.markdown("Select an index (from the **sampled** dataset) to view its individual feature contributions.")

index = st.slider("Select sample index", 0, len(X_sample)-1, 0)

shap.initjs()
force_plot = shap.plots.force(
    base_value,
    shap_values[index],
    X_sample.iloc[index].values.reshape(1, -1),
    feature_names=feature_order,
    matplotlib=False
)
components.html(force_plot.html(), height=300)

st.success(f"✅ SHAP analysis loaded successfully using {sample_size} samples! Remember to improve your model's Class 1 Recall.")
