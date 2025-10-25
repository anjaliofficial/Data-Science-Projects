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
st.set_page_config(page_title="🔍 SHAP Model Interpretation", layout="centered")
st.title("🧠 Stroke Prediction – SHAP Interpretation Dashboard")

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "../data/test_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "../models/stroke_model.pkl")

# -----------------------------
# Load Model & Data
# -----------------------------
try:
    model = joblib.load(MODEL_PATH)
    test_data = pd.read_csv(DATA_PATH)
    st.success("✅ Model and test dataset loaded successfully.")
except FileNotFoundError:
    st.error("❌ Missing file: Could not find model or test_data.csv. Check the /data and /models folders.")
    st.stop()

# -----------------------------
# Data Preparation
# -----------------------------
if "stroke" in test_data.columns:
    X_test = test_data.drop(columns=["stroke"])
else:
    X_test = test_data.copy()

# Auto-sample if dataset too large (for SHAP performance)
if len(X_test) > 1000:
    st.warning(f"⚠️ Dataset too large ({len(X_test)} rows). Sampling 1000 rows for SHAP to avoid lag.")
    X_sample = X_test.sample(1000, random_state=42)
else:
    X_sample = X_test.copy()

st.write("### 📋 Test Data Sample")
st.dataframe(X_test.head())

# -----------------------------
# Cached SHAP Computation
# -----------------------------
@st.cache_data(show_spinner=True, max_entries=5)
def compute_shap_values(model, X_sample):
    """Compute SHAP values with caching for speed."""
    # Auto-detect explainer type
    if hasattr(model, "predict_proba"):
        explainer = shap.Explainer(model, X_sample)
    else:
        explainer = shap.KernelExplainer(
            model.predict, 
            X_sample.sample(min(100, len(X_sample)), random_state=42)
        )
    shap_values = explainer(X_sample)
    return explainer, shap_values

# -----------------------------
# Compute SHAP Values
# -----------------------------
st.write("### ⚙️ Computing SHAP Values... (cached)")

try:
    explainer, shap_values = compute_shap_values(model, X_sample)
    st.success(f"✅ SHAP values computed on {len(X_sample)} samples (cached).")
except Exception as e:
    st.error(f"⚠️ Failed to compute SHAP values: {e}")
    st.stop()

# -----------------------------
# 1️⃣ SHAP Summary Plot
# -----------------------------
st.subheader("1️⃣ SHAP Summary Plot")
fig_summary, ax = plt.subplots(figsize=(8, 5))
shap.summary_plot(shap_values.values, X_sample, show=False)
st.pyplot(fig_summary, clear_figure=True)

# -----------------------------
# 2️⃣ Mean |SHAP| Feature Importance
# -----------------------------
st.subheader("2️⃣ Mean |SHAP| Feature Importance")

mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
importance_df = pd.DataFrame({
    "Feature": X_sample.columns,
    "Mean |SHAP|": mean_abs_shap
}).sort_values(by="Mean |SHAP|", ascending=False)

st.bar_chart(importance_df.set_index("Feature"))

# -----------------------------
# 3️⃣ Individual Prediction Force Plot
# -----------------------------
st.subheader("3️⃣ Explore Individual Predictions")

selected_index = st.slider("Select a sample index", 0, len(X_sample) - 1, 0)

# Determine correct base value for SHAP 0.20+
if isinstance(explainer.expected_value, (list, np.ndarray)):
    base_value = explainer.expected_value[0]
else:
    base_value = explainer.expected_value

try:
    shap.plots.force(
        base_value,
        shap_values.values[selected_index, :],
        X_sample.iloc[selected_index, :],
        matplotlib=True
    )
    st.pyplot(bbox_inches="tight", pad_inches=0.5, clear_figure=True)
except Exception as e:
    st.error(f"⚠️ Could not render force plot: {e}")

# -----------------------------
# 4️⃣ Interactive HTML Force Plot
# -----------------------------
with st.expander("💡 View Interactive Force Plot (HTML)", expanded=False):
    try:
        force_html = shap.plots.force(
            base_value,
            shap_values.values[selected_index, :],
            X_sample.iloc[selected_index, :]
        )
        shap_html = f"<head>{shap.getjs()}</head><body>{force_html.html()}</body>"
        components.html(shap_html, height=300)
    except Exception as e:
        st.warning(f"Interactive SHAP plot unavailable: {e}")

# -----------------------------
# Footer
# -----------------------------
st.info("✅ All SHAP visualizations generated successfully!")
