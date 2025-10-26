import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import shap
import matplotlib.pyplot as plt
from streamlit_shap import st_shap
import warnings
import streamlit.components.v1 as components # Import for robust force plot rendering

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
y = df[target_col]

# Detect categorical columns dynamically
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

# One-hot encode categorical features (matching training logic)
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
# Coerce to numeric and fill NaN from potential dummy creation to match model input
X_encoded = X_encoded.apply(pd.to_numeric, errors="coerce").fillna(0) 

# -----------------------------
# Robust Feature Alignment (Critical for SHAP)
# -----------------------------
st.markdown("### ⚙️ Feature Alignment Check")

# 1. Add missing columns (features present in training but not in current data)
missing_cols = [c for c in feature_order if c not in X_encoded.columns]
for c in missing_cols:
    X_encoded[c] = 0

# 2. Drop extra columns (features present in current data but not in training)
extra_cols = [c for c in X_encoded.columns if c not in feature_order]
if extra_cols:
    X_encoded = X_encoded.drop(columns=extra_cols)
    st.warning(f"Dropped {len(extra_cols)} extra features: {extra_cols[:2]}...")

# 3. Final Reorder (ensures columns are in the exact order the model expects)
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
# The output is a list for multi-class/multi-output models, array otherwise
shap_values = explainer.shap_values(shap_data)

# -----------------------------
# SHAP OUTPUT SELECTION (Focus on Positive Class: Stroke=1)
# -----------------------------
if isinstance(shap_values, list) and len(shap_values) > 1:
    # Use the SHAP values for the positive class (Stroke Risk, index 1)
    shap_values_class1 = np.array(shap_values[1])
    expected_value_class1 = explainer.expected_value[1] 
    st.info("Using SHAP values for **Class 1 (Stroke)**.")
elif isinstance(shap_values, np.ndarray):
    # Handles single-output models (e.g., linear regression or binary classifier using a single output)
    shap_values_class1 = shap_values
    expected_value_class1 = explainer.expected_value
    st.info("Using SHAP values for single-output model.")
else:
    st.error("❌ SHAP values are not in an expected format.")
    st.stop()

# -----------------------------
# 1️⃣ SHAP Summary Plot
# -----------------------------
st.markdown("---")
st.markdown("### 📈 SHAP Summary Plot (Impact on Stroke Risk)")
# Use shap.summary_plot to display the Summary Plot
try:
    fig, ax = plt.subplots()
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

# Use the feature names from the SHAP data (which is aligned)
plotting_feature_names = shap_data.columns.tolist() 

feature_len = len(plotting_feature_names)
shap_len = len(mean_abs_shap)

# Create the Importance DataFrame
if feature_len != shap_len:
    st.error(
        f"Assertion Failed: Features={feature_len}, SHAP Output={shap_len}. "
        f"This indicates a feature length mismatch. Slicing to minimal length for plot."
    )
    # Use the shorter list to avoid indexing errors
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

    st.write("**Selected sample data:**")
    st.dataframe(individual_data.T) # Transpose for easier viewing

    st.write("**SHAP Force Plot for selected prediction:**")
    try:
        # Generate the Force Plot. Using st.components.v1.html for robust rendering
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