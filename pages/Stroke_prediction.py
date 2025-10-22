import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import shap
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

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
    if isinstance(ev, np.ndarray):
        if ev.ndim == 1 and ev.size >= 2:
            return ev[1]
        return ev[0]
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
        explainer = shap.TreeExplainer(model)
        return model, scaler, feature_order, explainer
    except Exception as e:
        st.error(f"❌ Error loading artifacts. Please ensure 'scripts/train_model.py' has been run successfully: {e}")
        st.stop()

model, scaler, feature_order, explainer = load_artifacts()
base_value = get_expected_value(explainer)

# -----------------------------
# User Input
# -----------------------------
st.sidebar.header("Patient Information")
def user_input_features():
    gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
    age = st.sidebar.slider("Age", 0, 100, 50)
    hypertension = st.sidebar.selectbox("Hypertension (0=No, 1=Yes)", [0, 1])
    heart_disease = st.sidebar.selectbox("Heart Disease (0=No, 1=Yes)", [0, 1])
    ever_married = st.sidebar.selectbox("Ever Married", ["Yes", "No"])
    work_type = st.sidebar.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "Children", "Never_worked"])
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
# Preprocessing
# -----------------------------
# 1. One-hot encode the input
X = pd.get_dummies(input_df, drop_first=True)

# 2. Align features to the exact order the model expects
for col in feature_order:
    if col not in X.columns:
        X[col] = 0
X = X[feature_order]

# 3. Scale for model prediction
X_scaled = scaler.transform(X)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Stroke Risk", type="primary"):
    pred_proba = model.predict_proba(X_scaled)[:, 1][0]
    st.subheader("Prediction Result")
    
    if pred_proba >= 0.5:
        st.error(f"🔴 High Risk: Probability of Stroke is **{pred_proba:.2%}**")
    else:
        st.success(f"🟢 Low Risk: Probability of Stroke is **{pred_proba:.2%}**")

    # -----------------------------
    # SHAP explanation for this input
    # -----------------------------
    st.markdown("### 🧪 SHAP Feature Interpretation")

    shap_values_input = explainer.shap_values(X) 
    if isinstance(shap_values_input, list):
        shap_values_input = shap_values_input[1] # class 1 = stroke

    # Individual force plot (FIX IS HERE)
    st.subheader("Individual Force Plot")
    shap.initjs()
    # FIX: Convert the single row DataFrame to a 2D NumPy array for reliability
    force_plot = shap.plots.force(
        base_value, 
        shap_values_input[0],
        X.iloc[0].values.reshape(1, -1), # FIX: Ensure 2D NumPy array
        feature_names=feature_order, # Provide names since we are using a NumPy array
        matplotlib=False
    )
    components.html(force_plot.html(), height=400)
    
    # Feature importance waterfall plot
    st.subheader("Feature Contribution (Waterfall Plot)")
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Create SHAP Explanation object
    shap_explanation = shap.Explanation(
        values=shap_values_input[0],
        base_values=base_value,
        data=X.iloc[0].values, 
        feature_names=feature_order 
    )
    
    shap.plots.waterfall(
        shap_explanation, 
        show=False
    )
    st.pyplot(fig, clear_figure=True)

# -----------------------------
# Show input dataframe
# -----------------------------
with st.expander("Show Input Features"):
    st.dataframe(input_df.style.set_properties(**{'font-size': '10pt'}))