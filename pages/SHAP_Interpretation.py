import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import shap
import matplotlib.pyplot as plt
from streamlit_shap import st_shap

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

# -----------------------------
# Robust Feature Alignment (Ensures 16 features are present)
# -----------------------------
# 1. Add missing columns
missing_cols = [c for c in feature_order if c not in X_encoded.columns]
for c in missing_cols:
    X_encoded[c] = 0

# 2. Drop extra columns
extra_cols = [c for c in X_encoded.columns if c not in feature_order]
if extra_cols:
    X_encoded = X_encoded.drop(columns=extra_cols)
    st.warning(f"Dropped {len(extra_cols)} extra features not in model training data.")

# 3. Final Reorder
X_encoded = X_encoded[feature_order]

# -----------------------------
# DEBUGGING STEP: Check Column Counts
# -----------------------------
expected_count = len(feature_order)
current_count = len(X_encoded.columns)

if expected_count != current_count:
    st.error(f"FATAL: Feature counts DO NOT match after alignment. Expected: {expected_count}, Found: {current_count}")
    st.stop()

# -----------------------------
# SHAP Explainer & Values
# -----------------------------
explainer = shap.TreeExplainer(model)

if len(X_encoded) > 1000:
    shap_data = X_encoded.sample(1000, random_state=42)
else:
    shap_data = X_encoded

st.write("Calculating SHAP values... please wait ⏳")
shap_values = explainer.shap_values(shap_data)

# -----------------------------
# FIX FOR SHAP OUTPUT MISMATCH (32 vs 16)
# -----------------------------
if isinstance(shap_values, list) and len(shap_values) > 1:
    # This block handles the binary classification model output ([class_0, class_1])
    # We must explicitly use the SHAP values for the positive class (Stroke Risk, index 1)
    shap_values_class1 = np.array(shap_values[1])
    expected_value_class1 = explainer.expected_value[1] 
    
    # Critical Check: Ensures the selected array has the correct feature count (16)
    if shap_values_class1.shape[1] != len(shap_data.columns):
        st.error(
            f"❌ SHAP array feature count mismatch: SHAP output has {shap_values_class1.shape[1]} columns, "
            f"but data has {len(shap_data.columns)}."
        )
        st.stop()

elif isinstance(shap_values, np.ndarray):
    # Handles single-output models (regression/single-class)
    shap_values_class1 = shap_values
    expected_value_class1 = explainer.expected_value
else:
    st.error("❌ SHAP values are not in an expected format. Check your model type/SHAP library.")
    st.stop()

# -----------------------------
# 1️⃣ SHAP Summary Plot
# -----------------------------
st.markdown("### 📈 SHAP Summary Plot (Impact on Stroke Risk)")
# Plot using only the single array for class 1
st_shap(shap.summary_plot(shap_values_class1, shap_data, show=False), height=500)

# -----------------------------
# 2️⃣ Mean |SHAP| Feature Importance
# -----------------------------
st.markdown("### 🔍 Feature Importance (Mean |SHAP| Values)")

# Calculate mean absolute SHAP using the correct (N_SAMPLES, 16) array
mean_abs_shap = np.abs(shap_values_class1).mean(axis=0).flatten()

# Final assertion (Should pass if the logic above is correct)
feature_len = len(shap_data.columns)
shap_len = len(mean_abs_shap)

assert feature_len == shap_len, (
    f"Mismatch in feature and SHAP values length: Features={feature_len}, SHAP Output={shap_len}. "
    f"This indicates a deep model inconsistency requiring retraining."
)

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
    
    # Ensure data and SHAP values are extracted from the correct objects
    individual_data = shap_data.iloc[[index_choice]] 
    individual_shap_values = shap_values_class1[index_choice] 

    st.write("Selected sample data:")
    st.dataframe(individual_data)

    st.write("**SHAP Force Plot for selected prediction:**")
    try:
        force_plot = shap.force_plot(
            expected_value_class1,
            individual_shap_values,
            individual_data
        )
        st_shap(force_plot, height=300, width=800)
    except Exception as e:
        st.error(f"❌ Error generating force plot: {e}")
else:
    st.warning("Not enough data to show individual SHAP plots.")