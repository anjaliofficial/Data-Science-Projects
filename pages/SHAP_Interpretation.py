import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import shap
import matplotlib.pyplot as plt
import streamlit.components.v1 as components 
import warnings

# Suppress warnings related to nopython compilation which often appear with shap/numba
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="🔍 Predict Stroke Risk with SHAP Insights", layout="wide")
st.title("🔍 Predict Stroke Risk with SHAP Insights")

# -----------------------------
# Paths
# -----------------------------
# Adjust paths based on your project structure:
# Assuming this script is in 'pages/' and artifacts are in 'models/' and 'data/'
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
    # scaler is loaded but not used in this specific file, kept for completeness
    scaler = joblib.load(SCALER_PATH)
    # feature_order is crucial and must contain ALL features the model was TRAINED on
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
# SHAP CALCULATION
# -----------------------------
explainer = shap.TreeExplainer(model)

# Sample a maximum of 1000 observations for performance
if len(X_encoded) > 1000:
    shap_data = X_encoded.sample(1000, random_state=42)
else:
    shap_data = X_encoded

st.write("Calculating SHAP values... please wait ⏳")
shap_values = explainer.shap_values(shap_data)

# -----------------------------
# 🌟 ROBUST SHAP OUTPUT SELECTION (Fixes 16 vs 32 Mismatch)
# -----------------------------
expected_features = len(feature_order)

if isinstance(shap_values, list) and len(shap_values) > 1:
    # Standard multi-class (list of arrays, one for each class)
    shap_values_class1 = np.array(shap_values[1])
    expected_value_class1 = explainer.expected_value[1] 
    st.info("Using SHAP values for **Class 1 (Stroke)** from a list output.")
    
elif isinstance(shap_values, np.ndarray) and shap_values.shape[-1] == expected_features:
    # Single-output model with correct feature count
    shap_values_class1 = shap_values
    expected_value_class1 = explainer.expected_value
    st.info("Using SHAP values for single-output model (correct shape).")
    
elif isinstance(shap_values, np.ndarray) and shap_values.shape[-1] > expected_features and shap_values.shape[-1] % expected_features == 0:
    # 🚨 CRITICAL FIX for N x 32 array when N x 16 is expected
    # Assumes the model is binary (2 classes) and output is a flat array (N, features * 2)
    num_classes = shap_values.shape[-1] // expected_features
    
    # We select the last 'expected_features' columns, assuming they belong to the positive class (Stroke)
    shap_values_class1 = shap_values[:, -expected_features:]
    
    # Select the corresponding expected value (Base Value)
    if isinstance(explainer.expected_value, np.ndarray) and len(explainer.expected_value) == num_classes:
        expected_value_class1 = explainer.expected_value[-1]
    else:
        # Fallback if explainer only returned a single expected value
        expected_value_class1 = explainer.expected_value
        
    st.warning(f"Detected N x {shap_values.shape[-1]} SHAP array. Extracted **Class {num_classes-1} (Stroke)** values.")
    
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
    shap.summary_plot(shap_values_class1, shap_data, show=False, max_display=15, plot_type="dot")
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

# Ensure we use the aligned feature names
plotting_feature_names = shap_data.columns.tolist() 

feature_len = len(plotting_feature_names)
shap_len = len(mean_abs_shap)

# Create the Importance DataFrame
if feature_len != shap_len:
    # This block should now rarely be hit due to the SHAP output selection fix
    st.error(
        f"Assertion Failed: Features={feature_len}, SHAP Output={shap_len}. "
        f"Mismatch after output selection. Slicing for safety."
    )
    min_len = min(feature_len, shap_len)
    importance_df = pd.DataFrame({
        "Feature": plotting_feature_names[:min_len],
        "Mean |SHAP|": mean_abs_shap[:min_len]
    }).sort_values(by="Mean |SHAP|", ascending=False)
else:
    # Normal execution path
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

if len(shap_data) > 0:
    index_choice = st.slider("Select sample index:", 0, len(shap_data) - 1, 0)
    
    # Get the data and SHAP values for the selected sample
    individual_data = shap_data.iloc[[index_choice]] 
    individual_shap_values = shap_values_class1[index_choice] 

    st.write("**Selected sample data (Transposed):**")
    st.dataframe(individual_data.T)

    st.write("**SHAP Force Plot for selected prediction:**")
    try:
        # Generate the Force Plot. Using components.html for robust rendering (CRITICAL FIX)
        force_html = shap.force_plot(
            expected_value_class1,
            individual_shap_values,
            individual_data,
            matplotlib=False # Must be False to output HTML/JavaScript
        )
        
        # Display the HTML plot using Streamlit components
        # Use a high height to prevent cutoff, and allow scrolling
        components.html(force_html.html(), height=400, scrolling=True)
        
    except Exception as e:
        st.error(f"❌ Error generating force plot: {e}")
        st.info("Try running `pip install --upgrade shap` to resolve potential version conflicts.")
else:
    st.warning("Not enough data to show individual SHAP plots.")