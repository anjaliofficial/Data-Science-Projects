import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import shap
import matplotlib.pyplot as plt
import streamlit.components.v1 as components 
import warnings
from streamlit_shap import st_shap # <<< NEW IMPORT for stable Force Plot rendering

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

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
# Load Data & Artifacts
# -----------------------------
try:
    df = pd.read_csv(DATA_PATH)
    st.subheader("📊 Data Preview")
    st.dataframe(df.head())
except FileNotFoundError:
    st.error("❌ stroke_cleaned.csv not found. Please check your /data folder.")
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_order = joblib.load(FEATURES_PATH) 
    st.success(f"Model and artifacts loaded successfully. Expected features: {len(feature_order)}")
except Exception as e:
    st.error(f"❌ Error loading model/scaler/features: {e}. Check your model directory.")
    st.stop()

# -----------------------------
# Prepare Data for SHAP
# -----------------------------
target_col = "stroke"
if target_col not in df.columns:
    st.error(f"Target column '{target_col}' not found in dataset.")
    st.stop()

X = df.drop(columns=[target_col])

# Detect categorical columns dynamically
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

# One-hot encode categorical features (matching training logic)
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
X_encoded = X_encoded.apply(pd.to_numeric, errors="coerce").fillna(0) 

# -----------------------------
# Robust Feature Alignment (Critical for SHAP)
# -----------------------------
st.markdown("### ⚙️ Feature Alignment Check")

# 1. Add missing columns
missing_cols = [c for c in feature_order if c not in X_encoded.columns]
for c in missing_cols:
    X_encoded[c] = 0

# 2. Drop extra columns
extra_cols = [c for c in X_encoded.columns if c not in feature_order]
if extra_cols:
    X_encoded = X_encoded.drop(columns=extra_cols)
    st.warning(f"Dropped {len(extra_cols)} extra features.")

# 3. Final Reorder
X_encoded = X_encoded[feature_order]

# Final check
current_feature_count = len(X_encoded.columns)
if current_feature_count != len(feature_order):
    st.error(f"FATAL: Feature counts DO NOT match after alignment. Expected {len(feature_order)}, got {current_feature_count}.")
    st.stop()
else:
    st.info(f"Feature alignment successful. Total features: {current_feature_count}")
    
# -----------------------------
# SHAP CALCULATION & SCALING FIX
# -----------------------------
# Sample a maximum of 1000 observations for performance
if len(X_encoded) > 1000:
    shap_data_unscaled = X_encoded.sample(1000, random_state=42)
else:
    shap_data_unscaled = X_encoded

# Apply the scaler (CRITICAL CORRECTION)
shap_data_scaled = scaler.transform(shap_data_unscaled)
shap_data_for_explainer = pd.DataFrame(shap_data_scaled, columns=shap_data_unscaled.columns)

st.write("Calculating SHAP values on **SCALED** data... please wait ⏳")

try:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(shap_data_for_explainer)
    
    # 🚨 DEBUG STEP (Kept for clarity)
    st.info(f"DEBUG: Type of shap_values: {type(shap_values)}")
    if isinstance(shap_values, list):
        st.info(f"DEBUG: List length: {len(shap_values)}. Shape of first element: {shap_values[0].shape}")
    elif isinstance(shap_values, np.ndarray):
        st.info(f"DEBUG: Array shape: {shap_values.shape}")
    
except Exception as e:
    st.error(f"❌ Error during SHAP calculation: {e}. Check your model type or scaling.")
    st.stop()

# -----------------------------
# 🌟 ROBUST SHAP OUTPUT SELECTION (Handles List and 3D Array)
# -----------------------------
expected_features = len(feature_order)

if isinstance(shap_values, list) and len(shap_values) > 1:
    # Case 1: Standard multi-class (list of arrays)
    shap_values_class1 = np.array(shap_values[1])
    expected_value_class1 = explainer.expected_value[1] 
    st.info("Using SHAP values for **Class 1 (Stroke)** from a list output.")
    
elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3 and shap_values.shape[2] == 2:
    # Case 2: CRITICAL FIX for (N, F, 2) array 
    shap_values_class1 = shap_values[:, :, 1] # Slice for positive class (index 1)
    
    if isinstance(explainer.expected_value, np.ndarray) and explainer.expected_value.ndim == 1:
        expected_value_class1 = explainer.expected_value[1]
    else:
        expected_value_class1 = explainer.expected_value
        
    st.success(f"Successfully extracted SHAP values for **Class 1 (Stroke)** from 3D array shape: {shap_values.shape}.")
    
elif isinstance(shap_values, np.ndarray) and shap_values.shape[-1] == expected_features:
    # Case 3: Single-output model with correct feature count (N, F)
    shap_values_class1 = shap_values
    expected_value_class1 = explainer.expected_value
    st.info("Using SHAP values for single-output model (N, F).")
    
# ... (omitted Case 4 for brevity, as Case 2 is the most likely format now)
    
else:
    st.error("❌ SHAP values are not in an expected format or shape. Cannot plot.")
    st.stop()


# -----------------------------
# 1️⃣ SHAP Summary Plot
# -----------------------------
st.markdown("---")
st.markdown("### 📈 SHAP Summary Plot (Impact on Stroke Risk)")
try:
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values_class1, shap_data_unscaled, show=False, max_display=15, plot_type="dot") 
    st.pyplot(fig)
    plt.close(fig)
except Exception as e:
    st.error(f"Error generating Summary Plot: {e}")


# -----------------------------
# 2️⃣ Mean |SHAP| Feature Importance
# -----------------------------
st.markdown("---")
st.markdown("### 🔍 Feature Importance (Mean |SHAP| Values)")

mean_abs_shap = np.abs(shap_values_class1).mean(axis=0).flatten()
plotting_feature_names = shap_data_unscaled.columns.tolist() 
feature_len = len(plotting_feature_names)
shap_len = len(mean_abs_shap)

# Create the Importance DataFrame
if feature_len != shap_len:
    st.error(f"Assertion Failed: Features={feature_len}, SHAP Output={shap_len}. Mismatch after output selection.")
    min_len = min(feature_len, shap_len)
    importance_df = pd.DataFrame({
        "Feature": plotting_feature_names[:min_len],
        "Mean |SHAP|": mean_abs_shap[:min_len]
    }).sort_values(by="Mean |SHAP|", ascending=False)
else:
    importance_df = pd.DataFrame({
        "Feature": plotting_feature_names,
        "Mean |SHAP|": mean_abs_shap
    }).sort_values(by="Mean |SHAP|", ascending=False)


st.bar_chart(importance_df.set_index("Feature"))

# -----------------------------
# 3️⃣ Individual Prediction Exploration
# -----------------------------
st.markdown("---")
st.markdown("### 🧩 Explore Individual Prediction")

if len(shap_data_unscaled) > 0:
    index_choice = st.slider("Select sample index:", 0, len(shap_data_unscaled) - 1, 0)
    
    individual_data_unscaled = shap_data_unscaled.iloc[[index_choice]] 
    individual_shap_values = shap_values_class1[index_choice] 

    st.write("**Selected sample data (Unscaled):**")
    st.dataframe(individual_data_unscaled.T)

    st.write("**SHAP Force Plot for selected prediction (Using `streamlit-shap`):**")
    try:
        # Generate the Force Plot. 
        # We must use shap.Explanation object for st_shap to work reliably
        single_instance_explanation = shap.Explanation(
            values=individual_shap_values,
            base_values=expected_value_class1,
            data=individual_data_unscaled.iloc[0].values,
            feature_names=individual_data_unscaled.columns.tolist()
        )
        
        # <<< CRITICAL FIX: Use st_shap for stable rendering >>>
        st_shap(
            shap.force_plot(
                expected_value_class1, 
                individual_shap_values, 
                individual_data_unscaled,
                matplotlib=False # Must be False
            )
        )
        
    except Exception as e:
        st.error(f"❌ Error generating force plot: {e}")
        st.info("Ensure `pip install streamlit-shap` and `pip install ipython` are complete.")
else:
    st.warning("Not enough data to show individual SHAP plots.")