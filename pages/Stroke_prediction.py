# pages/SHAP_Interpretation.py
import streamlit as st
import pandas as pd
import joblib
import os
import shap
import matplotlib.pyplot as plt

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
def load_model_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_order = joblib.load(FEATURES_PATH)
    return model, scaler, feature_order

model, scaler, feature_order = load_model_artifacts()

if not os.path.exists(CSV_PATH):
    st.error(f"❌ Data file not found: {CSV_PATH}")
    st.stop()

df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.lower().str.strip()

# -----------------------------
# Preprocessing
# -----------------------------
categorical_cols = [c for c in ['gender','ever_married','work_type','residence_type','smoking_status'] if c in df.columns]
numeric_cols = [c for c in ['age','avg_glucose_level','bmi','hypertension','heart_disease'] if c in df.columns]

if "bmi" in df.columns:
    df['bmi'] = df['bmi'].fillna(df['bmi'].median())

X = pd.get_dummies(df[numeric_cols + categorical_cols], drop_first=True)
for col in feature_order:
    if col not in X.columns:
        X[col] = 0
X = X[feature_order]
X_scaled = scaler.transform(X)

# -----------------------------
# Cache SHAP Explainer
# -----------------------------
@st.cache_resource
def get_explainer(_model):
    return shap.TreeExplainer(_model)

explainer = get_explainer(model)

# -----------------------------
# SHAP Analysis
# -----------------------------
st.markdown("## 🔍 SHAP Analysis")
with st.spinner("Computing SHAP values..."):
    shap_values = explainer(X_scaled, check_additivity=False)

# --- Summary Plot ---
st.subheader("1️⃣ SHAP Summary Plot")
fig, ax = plt.subplots(figsize=(12, 8))
shap.summary_plot(shap_values, X, show=False)
st.pyplot(fig, clear_figure=True)

# --- Bar Plot (Mean |SHAP|) ---
st.subheader("2️⃣ SHAP Feature Importance")
fig2, ax2 = plt.subplots(figsize=(12, 6))
shap.summary_plot(shap_values, X, plot_type="bar", show=False)
st.pyplot(fig2, clear_figure=True)

# --- Individual Force Plot ---
st.markdown("---")
st.subheader("3️⃣ Explore Individual Prediction")
index = st.slider("Select data index", 0, len(X_scaled)-1, 0)

force_plot = shap.force_plot(
    explainer.expected_value,
    shap_values[index],
    X.iloc[[index]],
    matplotlib=False
)
shap.initjs()
st.components.v1.html(force_plot.html(), height=400)
