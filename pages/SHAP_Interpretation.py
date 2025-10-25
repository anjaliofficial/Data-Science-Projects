import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import os
import warnings

warnings.filterwarnings("ignore")

# -----------------------------
# Streamlit Page Setup
# -----------------------------
st.set_page_config(page_title="🔍 SHAP Model Interpretation", layout="wide")
st.title("🧠 Stroke Prediction – SHAP Interpretation Dashboard")

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "../data/stroke_cleaned.csv")
MODEL_PATH = os.path.join(BASE_DIR, "../models/stroke_model.pkl")

# -----------------------------
# Load Model & Data
# -----------------------------
try:
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)
    st.success("✅ Model and dataset loaded successfully.")
except FileNotFoundError:
    st.error("❌ Missing file: Could not find model or stroke_cleaned.csv. Check /models and /data folders.")
    st.stop()

# -----------------------------
# Prepare Data
# -----------------------------
target_col = "stroke"
if target_col not in df.columns:
    st.error(f"Target column '{target_col}' not found in dataset.")
    st.stop()

X = df.drop(columns=[target_col])
y = df[target_col]

# Detect categorical columns and one-hot encode
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
X_encoded = X_encoded.apply(pd.to_numeric, errors="coerce").fillna(0)

# Match feature order if model saved feature_names
feature_names_path = os.path.join(BASE_DIR, "../models/feature_names.pkl")
if os.path.exists(feature_names_path):
    model_features = joblib.load(feature_names_path)
    missing_cols = [c for c in model_features if c not in X_encoded.columns]
    for c in missing_cols:
        X_encoded[c] = 0
    X_encoded = X_encoded[model_features]
    st.success(f"✅ Matched model feature order ({len(model_features)} features).")
else:
    st.warning("⚠️ Model feature order file not found. SHAP may still work but verify features match.")

# Sample if too large
if len(X_encoded) > 1000:
    st.warning(f"⚠️ Dataset too large ({len(X_encoded)} rows). Sampling 1000 rows for SHAP to avoid lag.")
    X_sample = X_encoded.sample(1000, random_state=42)
else:
    X_sample = X_encoded.copy()

st.write("### 📋 Test Data Sample")
st.dataframe(X_sample.head())

# -----------------------------
# Compute SHAP Values
# -----------------------------
st.write("### ⚙️ Computing SHAP Values...")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

st.success(f"✅ SHAP values computed on {len(X_sample)} samples.")

# -----------------------------
# 1️⃣ SHAP Summary Plot
# -----------------------------
st.subheader("1️⃣ SHAP Summary Plot")
fig_summary, ax_summary = plt.subplots(figsize=(8, 5))

# Use class 1 for binary classifier if needed
if isinstance(shap_values, list):
    shap.summary_plot(shap_values[1], X_sample, show=False)
else:
    shap.summary_plot(shap_values, X_sample, show=False)

st.pyplot(fig_summary, clear_figure=True)

# -----------------------------
# 2️⃣ Mean |SHAP| Feature Importance
# -----------------------------
st.subheader("2️⃣ Mean |SHAP| Feature Importance")

if isinstance(shap_values, list):
    shap_vals_class1 = shap_values[1]
    if shap_vals_class1.ndim > 2:
        shap_vals_class1 = shap_vals_class1.reshape(shap_vals_class1.shape[-2:])
    mean_abs_shap = np.abs(shap_vals_class1).mean(axis=0)
else:
    shap_vals_single = shap_values
    if shap_vals_single.ndim > 2:
        shap_vals_single = shap_vals_single.reshape(shap_vals_single.shape[-2:])
    mean_abs_shap = np.abs(shap_vals_single).mean(axis=0)

importance_df = pd.DataFrame({
    "Feature": X_sample.columns.tolist(),
    "Mean |SHAP|": mean_abs_shap.tolist()
}).sort_values(by="Mean |SHAP|", ascending=False)

st.bar_chart(importance_df.set_index("Feature"))

# -----------------------------
# 3️⃣ Individual Prediction Force Plot
# -----------------------------
st.subheader("3️⃣ Explore Individual Predictions")

selected_index = st.slider("Select a sample index", 0, len(X_sample)-1, 0)
individual_data = X_sample.iloc[[selected_index]]

# Determine base value
if isinstance(explainer.expected_value, (list, np.ndarray)):
    base_value = explainer.expected_value[1] if isinstance(shap_values, list) else explainer.expected_value[0]
else:
    base_value = explainer.expected_value

try:
    shap.plots.force(
        base_value,
        shap_values[1][selected_index] if isinstance(shap_values, list) else shap_values[selected_index],
        individual_data,
        matplotlib=True
    )
    st.pyplot(clear_figure=True)
except Exception as e:
    st.error(f"⚠️ Could not render force plot: {e}")

# -----------------------------
# 4️⃣ Interactive HTML Force Plot
# -----------------------------
with st.expander("💡 View Interactive Force Plot (HTML)", expanded=False):
    try:
        force_html = shap.plots.force(
            base_value,
            shap_values[1][selected_index] if isinstance(shap_values, list) else shap_values[selected_index],
            individual_data
        )
        shap_html = f"<head>{shap.getjs()}</head><body>{force_html.html()}</body>"
        components.html(shap_html, height=300)
    except Exception as e:
        st.warning(f"Interactive SHAP plot unavailable: {e}")

# -----------------------------
# Footer
# -----------------------------
st.info("✅ All SHAP visualizations generated successfully!")
