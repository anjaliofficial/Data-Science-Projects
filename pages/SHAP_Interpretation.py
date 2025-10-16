# pages/SHAP_Interpretation.py

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
# Load Model and Artifacts
# -----------------------------
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        feature_order = joblib.load(FEATURES_PATH)
        return model, scaler, feature_order
    except FileNotFoundError as e:
        st.error(f"❌ Required model artifact not found: {e.filename}. Please run 'scripts/train_model.py' first.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading artifacts: {e}")
        st.stop()

model, scaler, feature_order = load_artifacts()

# -----------------------------
# Load Data
# -----------------------------
if not os.path.exists(CSV_PATH):
    st.error(f"❌ Data file not found: {CSV_PATH}. Please run 'scripts/train_model.py' to generate it.")
    st.stop()

df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.lower().str.strip()

# -----------------------------
# Preprocessing
# -----------------------------
categorical_cols = ['gender', 'ever_married', 'work_type', 'residence_type', 'smoking_status']
numeric_cols = ['age', 'avg_glucose_level', 'bmi', 'hypertension', 'heart_disease']

# Fill missing BMI
if 'bmi' in df.columns:
    df['bmi'] = df['bmi'].fillna(df['bmi'].median())

# One-hot encode and align features
X = pd.get_dummies(df[numeric_cols + categorical_cols], drop_first=True)
for col in feature_order:
    if col not in X.columns:
        X[col] = 0
X = X[feature_order]

# Scale
X_scaled = scaler.transform(X)

# -----------------------------
# SHAP Explainer
# -----------------------------
@st.cache_resource
def get_explainer(_model):
    return shap.TreeExplainer(_model)

explainer = get_explainer(model)

# -----------------------------
# Compute SHAP values
# -----------------------------
with st.spinner("Computing SHAP values for all data..."):
    shap_values_all = explainer.shap_values(X_scaled)
    if isinstance(shap_values_all, list):
        shap_values = shap_values_all[1]  # class 1 = stroke
    else:
        shap_values = shap_values_all

# -----------------------------
# SHAP Global Plots
# -----------------------------
st.subheader("1️⃣ SHAP Summary Plot (Feature Impact & Direction)")
st.markdown("Each dot represents a patient. Position on the x-axis shows the feature's impact on the prediction.")
fig, ax = plt.subplots(figsize=(12, 8))
shap.summary_plot(shap_values, X, show=False)
st.pyplot(fig, clear_figure=True)

st.subheader("2️⃣ Feature Importance (Mean |SHAP|)")
st.markdown("The average magnitude of the SHAP values, showing overall feature importance.")
fig2, ax2 = plt.subplots(figsize=(12, 6))
shap.summary_plot(shap_values, X, plot_type="bar", show=False)
st.pyplot(fig2, clear_figure=True)

# -----------------------------
# Individual Prediction Exploration (The corrected section)
# -----------------------------
st.markdown("---")
st.subheader("3️⃣ Individual Prediction Exploration")
st.markdown("Select an index from the dataset to view its individual feature contributions.")
index = st.slider("Select patient index", 0, len(X_scaled)-1, 0)

# FIX: Use matplotlib=False and embed the HTML plot
shap.initjs()
force_plot = shap.plots.force(
    explainer.expected_value[1] if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value,
    shap_values[index],
    X.iloc[index],
    matplotlib=False # The fix for the error
)
st.components.v1.html(force_plot.html(), height=300)

# -----------------------------
# Prediction Probability Comparison
# -----------------------------
st.markdown("---")
st.subheader("4️⃣ Prediction Probability Comparison")
proba = model.predict_proba(X_scaled)
pred_prob = proba[index, 1]

# Baseline probability (Log-odds to Probability)
baseline_log_odds = explainer.expected_value[1] if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value
baseline = 1 / (1 + np.exp(-baseline_log_odds))

col1, col2, col3 = st.columns(3)
col1.metric("🩸 Stroke Probability", f"{pred_prob:.2%}")
col2.metric("⚖️ Baseline Probability (Avg.)", f"{baseline:.2%}")
col3.metric("📈 Change", f"{(pred_prob - baseline):+.2%}")

st.success("✅ SHAP interpretation loaded successfully!")