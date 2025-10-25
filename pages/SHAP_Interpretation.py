import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import shap
import matplotlib.pyplot as plt
import shap.plots as shap_plots  # new API
import streamlit.components.v1 as components

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="🔍 Predict Stroke Risk with SHAP Insights", layout="wide")
st.title("🔍 Predict Stroke Risk with SHAP Insights")

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "../data/stroke_cleaned.csv")
MODEL_PATH = os.path.join(BASE_DIR, "../models/stroke_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "../models/scaler.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "../models/feature_names.pkl")

# -----------------------------
# Load Data
# -----------------------------
try:
    df = pd.read_csv(DATA_PATH)
    st.subheader("📊 Data Preview")
    st.dataframe(df.head())
except FileNotFoundError:
    st.error("❌ stroke_cleaned.csv not found. Please check your /data folder.")
    st.stop()

# -----------------------------
# Load Model and Artifacts
# -----------------------------
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_order = joblib.load(FEATURES_PATH)
except Exception as e:
    st.error(f"❌ Error loading model/scaler/features: {e}")
    st.stop()

# -----------------------------
# Prepare Data for SHAP
# -----------------------------
target_col = "stroke"
if target_col not in df.columns:
    st.error(f"Target column '{target_col}' not found in dataset.")
    st.stop()

X = df.drop(columns=[target_col])
y = df[target_col]

# Detect categorical columns dynamically
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
st.info(f"Auto-detected categorical columns: {categorical_cols}")

# One-hot encode categorical features
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
X_encoded = X_encoded.apply(pd.to_numeric, errors="coerce").fillna(0)

# Match model feature order
missing_cols = [c for c in feature_order if c not in X_encoded.columns]
for c in missing_cols:
    X_encoded[c] = 0
X_encoded = X_encoded[feature_order]

# -----------------------------
# SHAP Explainer
# -----------------------------
explainer = shap.TreeExplainer(model)

if len(X_encoded) > 1000:
    shap_data = X_encoded.sample(1000, random_state=42)
else:
    shap_data = X_encoded

st.write("Calculating SHAP values... please wait ⏳")
shap_values = explainer.shap_values(shap_data)

# -----------------------------
# 1️⃣ SHAP Summary Plot
# -----------------------------
st.markdown("### 📈 SHAP Summary Plot")
fig_summary, ax_summary = plt.subplots(figsize=(8, 6))
shap.summary_plot(shap_values, shap_data, show=False)
st.pyplot(fig_summary)

# -----------------------------
# 2️⃣ Mean |SHAP| Feature Importance
# -----------------------------
st.markdown("### 🔍 Feature Importance (Mean |SHAP| Values)")

if isinstance(shap_values, list):
    # Binary classifier: take class 1
    shap_vals_class1 = shap_values[1]
    mean_abs_shap = np.abs(shap_vals_class1).mean(axis=0).flatten()
else:
    # Single-output
    mean_abs_shap = np.abs(shap_values).mean(axis=0).flatten()

# Sanity check lengths
assert len(shap_data.columns) == len(mean_abs_shap), "Mismatch in feature and SHAP values length"

importance_df = pd.DataFrame({
    "Feature": shap_data.columns.tolist(),
    "Mean |SHAP|": mean_abs_shap
}).sort_values(by="Mean |SHAP|", ascending=False)

st.bar_chart(importance_df.set_index("Feature"))

# -----------------------------
# 3️⃣ Individual Prediction Exploration
# -----------------------------
st.markdown("### 🧩 Explore Individual Prediction")

if len(shap_data) > 0:
    index_choice = st.slider("Select sample index:", 0, len(shap_data) - 1, 0)
    individual_data = shap_data.iloc[[index_choice]]

    st.write("Selected sample data:")
    st.dataframe(individual_data)

    st.write("**SHAP Force Plot for selected prediction:**")
    try:
        if isinstance(shap_values, list):
            # Binary classifier
            fig_force = shap_plots.force(
                explainer.expected_value[1],
                shap_values[1][index_choice],
                individual_data,
                matplotlib=True
            )
        else:
            # Single-output
            fig_force = shap_plots.force(
                explainer.expected_value,
                shap_values[index_choice],
                individual_data,
                matplotlib=True
            )
        st.pyplot(fig_force)
    except Exception as e:
        st.error(f"❌ Error generating force plot: {e}")
else:
    st.warning("Not enough data to show individual SHAP plots.")
