import streamlit as st

st.set_page_config(page_title="Stroke Prediction System", page_icon="🧠", layout="wide")
st.sidebar.success("Select a page above")

st.title("🧠 Stroke Risk Prediction & Insights")
st.markdown("""
Welcome to the **Stroke Risk Prediction System**.

This system uses a **Random Forest Classifier** to predict stroke risk and leverages **SHAP** to explain the underlying factors driving those predictions.

Use the sidebar to navigate to the core functions:

* **🔍 Predict Stroke Risk:** Get a personalized prediction based on patient features.
* **📊 Explore Visual Insights:** View data distributions and statistical insights using **Plotly**.
* **🧪 Interpret Model Features with SHAP:** See the feature importance and model fairness analysis.
""")

# Note: Using a generic health/brain icon as the original link might not load universally.
st.image("https://cdn-icons-png.flaticon.com/512/4380/4380791.png", width=200) 
st.markdown("---")
st.caption("Developed by **Anjali Bista** · Data Science Project")