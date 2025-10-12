import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

st.title("🧪 SHAP Feature Interpretation")

# Load model artifacts
model = joblib.load("models/xgb_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_order = joblib.load("models/feature_names.pkl")

# Load dataset for SHAP analysis
csv_path = "stroke_cleaned.csv"
df = pd.read_csv(csv_path)
df['bmi'].fillna(df['bmi'].median(), inplace=True)

X = pd.get_dummies(df[['gender','age','hypertension','heart_disease','ever_married',
                       'work_type','Residence_type','avg_glucose_level','bmi','smoking_status']], drop_first=True)

# Align columns
for col in feature_order:
    if col not in X.columns:
        X[col] = 0
X = X[feature_order]

X_scaled = scaler.transform(X)

# SHAP values
explainer = shap.Explainer(model, X_scaled)
shap_values = explainer(X_scaled)

st.subheader("Feature Importance (SHAP Summary)")
fig, ax = plt.subplots(figsize=(10,6))
shap.summary_plot(shap_values, X_scaled, feature_names=feature_order, show=False)
st.pyplot(fig)
