import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import shap
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

st.set_page_config(page_title="🔍 Predict Stroke Risk with SHAP Insights", layout="wide")
st.title("🔍 Predict Stroke Risk with SHAP Insights")

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "../data/stroke_cleaned.csv")
MODEL_PATH = os.path.join(BASE_DIR, "../models/stroke_model.pkl")

# -----------------------------
# Load data and model
# -----------------------------
st.subheader("📊 Data Preview")

try:
    df = pd.read_csv(DATA_PATH)
    st.dataframe(df.head())
except FileNotFoundError:
    st.error("❌ stroke_cleaned.csv not found. Please check your /data folder.")
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# -----------------------------
# Prepare data for SHAP
# -----------------------------
st.subheader("🧠 SHAP Feature Analysis")

# Identify target variable
target_col = "stroke"
if target_col not in df.columns:
    st.error(f"Target column '{target_col}' not found in dataset.")
    st.stop()

X = df.drop(columns=[target_col])
y = df[target_col]

# Detect categorical columns dynamically (those with dtype == object)
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

if len(categorical_cols) > 0:
    st.info(f"Auto-detected categorical columns: {categorical_cols}")
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
else:
    X_encoded = X.copy()

# Convert to numeric (safety)
X_encoded = X_encoded.apply(pd.to_numeric, errors="coerce").fillna(0)

# -----------------------------
# Compute SHAP values
# -----------------------------
explainer = shap.TreeExplainer(model)

# Sample for performance
if len(X_encoded) > 1000:
    shap_data = X_encoded.sample(1000, random_state=42)
else:
    shap_data = X_encoded

st.write("Calculating SHAP values... please wait ⏳")
shap_values = explainer.shap_values(shap_data)

# -----------------------------
# SHAP Plots
# -----------------------------
st.markdown("### 📈 1️⃣ SHAP Summary Plot")

fig_summary, ax_summary = plt.subplots(figsize=(8, 6))
shap.summary_plot(shap_values, shap_data, show=False)
st.pyplot(fig_summary)

st.markdown("### 🔍 2️⃣ Feature Importance (Mean |SHAP| Values)")

shap.summary_plot(shap_values, shap_data, plot_type="bar", show=False)
fig_importance, ax_importance = plt.subplots(figsize=(8, 6))
shap.summary_plot(shap_values, shap_data, plot_type="bar", show=False)
st.pyplot(fig_importance)

# -----------------------------
# Individual Prediction
# -----------------------------
st.markdown("### 🧩 3️⃣ Explore Individual Prediction")

if len(shap_data) > 0:
    index_choice = st.slider("Select sample index:", 0, len(shap_data) - 1, 0)
    individual_data = shap_data.iloc[[index_choice]]

    st.write("Selected sample data:")
    st.dataframe(individual_data)

    st.write("**SHAP Force Plot for selected prediction:**")
    shap_html = shap.force_plot(
        explainer.expected_value[1],
        shap_values[1][index_choice],
        individual_data,
        matplotlib=False
    )
    components.html(shap.getjs() + shap_html.html(), height=300)
else:
    st.warning("Not enough data to show individual SHAP plots.")
