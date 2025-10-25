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
    # Load the exact feature names and order the model was trained on
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
# Convert all to numeric and fill NaN (from coercing non-numeric values)
X_encoded = X_encoded.apply(pd.to_numeric, errors="coerce").fillna(0) 

# -----------------------------
# Robust Feature Alignment (VITAL FIX)
# -----------------------------
# 1. Add missing columns (features expected by the model but not in the current data)
missing_cols = [c for c in feature_order if c not in X_encoded.columns]
for c in missing_cols:
    X_encoded[c] = 0

# 2. Drop extra columns (features in the current data but NOT expected by the model)
extra_cols = [c for c in X_encoded.columns if c not in feature_order]
if extra_cols:
    X_encoded = X_encoded.drop(columns=extra_cols)
    st.warning(f"Dropped {len(extra_cols)} extra features not in model training data.")

# 3. Final Reorder (Ensures the columns are in the exact sequence the model expects)
X_encoded = X_encoded[feature_order]

# -----------------------------
# DEBUGGING STEP: Check Column Counts
# -----------------------------
expected_count = len(feature_order)
current_count = len(X_encoded.columns)
st.subheader("⚠️ Debug Check: Feature Alignment Status")
st.info(f"Expected Feature Count (from feature_names.pkl): **{expected_count}**")
st.info(f"Current Data Feature Count (after alignment): **{current_count}**")

# If this warning shows, the problem is in the saved feature_order or the data loading.
if expected_count != current_count:
    st.error("FATAL: Feature counts DO NOT match after alignment. Check your `feature_order` file and original data.")
    st.stop()
# -----------------------------

# -----------------------------
# SHAP Explainer & Values
# -----------------------------
explainer = shap.TreeExplainer(model)

# Use a subsample of data for faster SHAP calculation, maintaining column alignment
if len(X_encoded) > 1000:
    shap_data = X_encoded.sample(1000, random_state=42)
else:
    shap_data = X_encoded

st.write("Calculating SHAP values... please wait ⏳")
# The SHAP values calculation must use the data frame with the exact same columns as the model input
shap_values = explainer.shap_values(shap_data)

# --- SHAP Value Selection (Binary Classifier) ---
# We target the positive class (stroke risk), which is index 1.
if isinstance(shap_values, list) and len(shap_values) > 1:
    shap_values_class1 = shap_values[1]
    expected_value_class1 = explainer.expected_value[1] 
elif isinstance(shap_values, np.ndarray):
    # Handle single-output regression or single-class classification
    shap_values_class1 = shap_values
    expected_value_class1 = explainer.expected_value
else:
    st.error("❌ SHAP values are not in an expected format (list or numpy array). Check your model type.")
    st.stop()


# -----------------------------
# 1️⃣ SHAP Summary Plot
# -----------------------------
st.markdown("### 📈 SHAP Summary Plot (Impact on Stroke Risk)")
# Plot using the single array for class 1
st_shap(shap.summary_plot(shap_values_class1, shap_data, show=False), height=500)

# -----------------------------
# 2️⃣ Mean |SHAP| Feature Importance
# -----------------------------
st.markdown("### 🔍 Feature Importance (Mean |SHAP| Values)")

# Calculate mean absolute SHAP using the correct array
mean_abs_shap = np.abs(shap_values_class1).mean(axis=0).flatten()

# --- FINAL ASSERTION CHECK ---
# Check 1: Feature count from the DataFrame
feature_len = len(shap_data.columns)
# Check 2: SHAP output count (the cause of the original error)
shap_len = len(mean_abs_shap)

# If the code reaches this point, the counts from the data should match.
# If it fails, it means the model's SHAP output has a different number of columns 
# than the input data, indicating a model saving/loading inconsistency.
assert feature_len == shap_len, (
    f"Mismatch in feature and SHAP values length: Features={feature_len}, SHAP Output={shap_len}. "
    f"This usually means the saved model is inconsistent with the feature_order file."
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
    # The selected sample's feature data
    individual_data = shap_data.iloc[[index_choice]] 
    # The selected sample's SHAP values for class 1
    individual_shap_values = shap_values_class1[index_choice] 

    st.write("Selected sample data:")
    st.dataframe(individual_data)

    st.write("**SHAP Force Plot for selected prediction:**")
    try:
        # Use the pre-calculated Class 1 expected value and SHAP values
        force_plot = shap.force_plot(
            expected_value_class1,
            individual_shap_values,
            individual_data
        )
        # Use streamlit-shap to display force plot
        st_shap(force_plot, height=300, width=800)
    except Exception as e:
        st.error(f"❌ Error generating force plot: {e}")
else:
    st.warning("Not enough data to show individual SHAP plots.")