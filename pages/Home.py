import streamlit as st

st.set_page_config(page_title="Stroke Prediction System", page_icon="🧠", layout="wide")
st.sidebar.success("Select a page above 👆")

st.title("🧠 Stroke Risk Prediction & Insights")
st.markdown("""
Welcome to the **Stroke Risk Prediction System** built with **XGBoost + Streamlit + SHAP**.

Use the sidebar to navigate to the core functions:
- 🔍 Predict Stroke Risk  
- 📊 Explore Visual Insights  
- 🧪 Interpret Model Features with SHAP
""")

st.image("https://cdn-icons-png.flaticon.com/512/4380/4380791.png", width=200)
st.markdown("---")
st.caption("Developed by **Anjali Bista** · Data Science Project")