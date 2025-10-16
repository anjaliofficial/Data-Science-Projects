# pages/SHAP_Interpretation.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="🧪 SHAP Feature Interpretation", layout="wide")
st.title("🧪 SHAP Feature Interpretation")
st.markdown("""
Understand how the model makes predictions by analyzing **feature contributions** using SHAP values.
This helps explain why the model predicted stroke risk for each observation.
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
# Load Artifacts
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
    st.stop()

if not os.path.exists(CSV_PATH):
    st.error(f"❌ Data file not found: {CSV_PATH}")
    st.stop()

df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.lower().str.strip()

# -----------------------------
# Preprocessing
# -----------------------------
categorical_cols = [c for c in ['gender', 'ever_married', 'work_type', 'residence_type', 'smoking_status'] if c in df.columns]
numeric_cols = [c for c in ['age', 'avg_glucose_level', 'bmi', 'hypertension', 'heart_disease'] if c in df.columns]

if not categorical_cols and not numeric_cols:
    st.error("❌ Required columns not found in dataset.")
    st.stop()

if "bmi" in df.columns:
    df['bmi'] = df['bmi'].fillna(df['bmi'].median())

# One-hot encode
X = pd.get_dummies(df[numeric_cols + categorical_cols], drop_first=True)
for col in feature_order:
    if col not in X.columns:
        X[col] = 0
X = X[feature_order]

# Scale
X_scaled = scaler.transform(X)

# -----------------------------
# SHAP Analysis
# -----------------------------
st.markdown("## 🔍 SHAP Analysis")

@st.cache_resource
def compute_shap(model, X_scaled):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_scaled, check_additivity=False)
    return explainer, shap_values

with st.spinner("Computing SHAP values..."):
    explainer, shap_values = compute_shap(model, X_scaled)

# --- SHAP Summary Plot ---
st.subheader("1️⃣ SHAP Summary Plot")
fig, ax = plt.subplots(figsize=(12, 8))
shap.summary_plot(shap_values, X, show=False)
st.pyplot(fig, clear_figure=True)

# --- SHAP Bar Plot ---
st.subheader("2️⃣ Feature Importance (Mean |SHAP|)")
fig2, ax2 = plt.subplots(figsize=(12, 6))
shap.summary_plot(shap_values, X, plot_type="bar", show=False)
st.pyplot(fig2, clear_figure=True)

# --- Individual Prediction Force Plot ---
st.markdown("---")
st.subheader("3️⃣ Individual Prediction Exploration")
index = st.slider("Select patient index", 0, len(X_scaled) - 1, 0)

force_plot = shap.force_plot(
    explainer.expected_value,
    shap_values[index],
    X.iloc[[index]],
    matplotlib=False
)
st.components.v1.html(force_plot.html(), height=400)

# --- Prediction Probability Comparison ---
st.markdown("---")
st.subheader("4️⃣ Prediction Probability Comparison")
proba = model.predict_proba(X_scaled)
pred_prob = proba[index, 1]

# Compute baseline probability
baseline = 1 / (1 + np.exp(-explainer.expected_value)) if np.isscalar(explainer.expected_value) else np.mean(1 / (1 + np.exp(-explainer.expected_value)))

col1, col2, col3 = st.columns(3)
col1.metric("🩸 Stroke Probability", f"{pred_prob:.2%}")
col2.metric("⚖️ Baseline Probability", f"{baseline:.2%}")
col3.metric("📈 Change", f"{(pred_prob - baseline):+.2%}")

# Visual bar
fig3, ax3 = plt.subplots(figsize=(6, 3))
ax3.barh(["Baseline", "Predicted"], [baseline, pred_prob], color=["#AAAAAA", "#FF6B6B"])
ax3.set_xlim(0, 1)
ax3.set_xlabel("Probability of Stroke")
st.pyplot(fig3, clear_figure=True)

st.success("✅ SHAP interpretation loaded successfully!")
st.info("💡 Tip: Use this page to understand why the model predicted a higher or lower stroke risk for each patient.")
