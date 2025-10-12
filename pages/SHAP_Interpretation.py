import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

st.title("🧪 SHAP Feature Interpretation")

# Load artifacts
model = joblib.load("models/xgb_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_order = joblib.load("models/feature_names.pkl")

df = pd.read_csv("data/stroke_data.csv")
df['bmi'].fillna(df['bmi'].median(), inplace=True)

# Encode categorical features
df['gender'] = df['gender'].map({"Male":0,"Female":1,"Other":2})
df['ever_married'] = df['ever_married'].map({"No":0,"Yes":1})
df['work_type'] = df['work_type'].map({"Private":0, "Self-employed":1,"Govt_job":2,"Children":3,"Never_worked":4})
df['Residence_type'] = df['Residence_type'].map({"Urban":0,"Rural":1})
df['smoking_status'] = df['smoking_status'].map({"formerly smoked":0,"never smoked":1,"smokes":2,"Unknown":3})

X_scaled = scaler.transform(df[feature_order])
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_scaled)

st.subheader("Feature Importance (SHAP Summary Plot)")
plt.figure(figsize=(10,6))
shap.summary_plot(shap_values, X_scaled, feature_names=feature_order, show=False)
st.pyplot(plt)
