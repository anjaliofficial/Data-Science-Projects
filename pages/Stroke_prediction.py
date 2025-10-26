import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import shap
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import warnings

# Suppress warnings that commonly arise from SHAP/Numba
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

st.set_page_config(page_title="Predict Stroke Risk", layout="centered")
st.title("🔍 Predict Stroke Risk with SHAP Insights")

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "..", "models") 

MODEL_PATH = os.path.join(MODEL_DIR, "stroke_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "feature_names.pkl")

# -----------------------------
# Helper Function for SHAP Base Value
# -----------------------------
def get_expected_value(explainer):
    """Safely retrieves the base value (log-odds) for the positive class (index 1)."""
    ev = explainer.expected_value
    # For binary classification (most common TreeExplainer output)
    if isinstance(ev, np.ndarray) and ev.ndim >= 1 and ev.size >= 2:
        return ev[1]  # Return the base value for the positive class (stroke=1)
    # For single-output models or if ev is just a number
    if isinstance(ev, (float, np.ndarray)):
        return ev.item() if isinstance(ev, np.ndarray) else ev
    return ev


# -----------------------------
# Load Artifacts
# -----------------------------
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        feature_order = joblib.load(FEATURES_PATH)
        # TreeExplainer is fast and specific for Random Forest
        explainer = shap.TreeExplainer(model) 
        return model, scaler, feature_order, explainer
    except Exception as e:
        st.error(f"❌ Error loading artifacts. Please ensure 'train_model.py' has been run successfully and the model files exist in the 'models/' folder. Error: {e}")
        st.stop()

model, scaler, feature_order, explainer = load_artifacts()
base_value = get_expected_value(explainer)

# -----------------------------
# User Input
# -----------------------------
st.sidebar.header("Patient Information")

def user_input_features():
    gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
    age = st.sidebar.slider("Age", 0.0, 100.0, 50.0) 
    hypertension = st.sidebar.selectbox("Hypertension (0=No, 1=Yes)", [0, 1])
    heart_disease = st.sidebar.selectbox("Heart Disease (0=No, 1=Yes)", [0, 1])
    ever_married = st.sidebar.selectbox("Ever Married", ["Yes", "No"])
    work_type = st.sidebar.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
    residence_type = st.sidebar.selectbox("Residence Type", ["Urban", "Rural"])
    avg_glucose_level = st.sidebar.number_input("Average Glucose Level", 50.0, 300.0, 100.0)
    bmi = st.sidebar.number_input("BMI", 10.0, 60.0, 25.0)
    smoking_status = st.sidebar.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes", "Unknown"])
    
    data = {
        "gender": gender,
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "ever_married": ever_married,
        "work_type": work_type,
        "residence_type": residence_type,
        "avg_glucose_level": avg_glucose_level,
        "bmi": bmi,
        "smoking_status": smoking_status
    }
    return pd.DataFrame([data])

input_df = user_input_features()

# -----------------------------
# Preprocessing (Mimic Training Pipeline)
# -----------------------------

# 1. Handle 'Other' gender early
if input_df.iloc[0]['gender'] == 'Other':
    st.info("The model was trained excluding the 'Other' gender category. Please select 'Male' or 'Female' for prediction.")
    st.stop()

X_raw = input_df.copy()

# 2. One-Hot Encode (OHE)
categorical_cols = X_raw.select_dtypes(include=["object"]).columns.tolist()
X_encoded = pd.get_dummies(X_raw, columns=categorical_cols, drop_first=True)

# 3. Align Columns (Critical Step for OHE consistency)
X_aligned = pd.DataFrame(columns=feature_order)
for col in feature_order:
    # --- CORRECTED LINE (Fixes: AttributeError) ---
    # Assign the data. If the column doesn't exist, it returns the default [0], 
    # which pandas handles correctly during assignment.
    X_aligned[col] = X_encoded.get(col, [0]) 

# 4. Final Alignment and Type Conversion
# Convert all data to float and ensure final order.
X_aligned = X_aligned.fillna(0)[feature_order].astype(float)

# Check for empty frame after filtering (shouldn't happen here, but safe)
if X_aligned.empty:
    st.info("Input data is empty after processing.")
    st.stop()

# 5. Scale Input (CRITICAL: The model expects scaled data)
X_scaled = scaler.transform(X_aligned)

# -----------------------------
# Prediction & Visualization
# -----------------------------
if st.button("Predict Stroke Risk", type="primary"):
    
    # Calculate probability for class 1 (Stroke)
    pred_proba = model.predict_proba(X_scaled)[:, 1][0]
    st.subheader("Prediction Result")
    
    if pred_proba >= 0.5:
        st.error(f"🔴 High Risk: Probability of Stroke is **{pred_proba:.2%}**")
    else:
        st.success(f"🟢 Low Risk: Probability of Stroke is **{pred_proba:.2%}**")

    # -----------------------------
    # SHAP explanation
    # -----------------------------
    st.markdown("### 🧪 SHAP Feature Interpretation")

    # Use the UN-SCALED, aligned DataFrame X_aligned for SHAP calculation for readable values
    shap_values_list = explainer.shap_values(X_aligned)

    # Extract SHAP values for the positive class (index 1)
    if isinstance(shap_values_list, list) and len(shap_values_list) > 1:
        shap_values_pos_class = shap_values_list[1] # Class 1 (stroke) - SHAPE: (1, N)
    else:
        # Handle the case where the explainer returns a single array (e.g., (1, N))
        shap_values_pos_class = shap_values_list 

    # -----------------------------
    # SHAP Explanation Object 
    # -----------------------------
    # Create the Explanation object using 1D numpy arrays
    single_instance_explanation = shap.Explanation(
        values=shap_values_pos_class.flatten(), # 1D SHAP values
        base_values=base_value,
        data=X_aligned.iloc[0].values, # 1D feature values
        feature_names=feature_order
    )

    # -----------------------------
    # Force Plot (Requires IPython to be installed: pip install ipython)
    # -----------------------------
    st.subheader("Individual Force Plot")
    shap.initjs()

    # Pass the complete Explanation object to the force plot
    force_plot = shap.plots.force(
        single_instance_explanation,
        matplotlib=False
    )
    # Render the plot in Streamlit
    components.html(force_plot.html(), height=400)

    # -----------------------------
    # Waterfall Plot
    # -----------------------------
    st.subheader("Feature Contribution (Waterfall Plot)")
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Use the existing Explanation object
    shap.plots.waterfall(single_instance_explanation, show=False)
    
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)
    
# -----------------------------
# Show Input DataFrame
# -----------------------------
with st.expander("Show Encoded and Aligned Input Features (What the model sees)"):
    # Transpose for easier viewingpip install plotly
    st.dataframe(X_aligned.T.style.set_properties(**{'font-size': '10pt'}))