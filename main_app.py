import streamlit as st

st.set_page_config(
    page_title="Stroke Prediction System",
    page_icon="🧠",
    layout="wide"
)

st.sidebar.title("Stroke Risk System")
st.sidebar.info("Select a page from the sidebar.")

st.title("🧠 Stroke Risk Prediction & Insights")
st.markdown("""
Welcome to the **Stroke Risk Prediction System** built with **XGBoost + Streamlit**.

Use the sidebar to navigate:
- 🏠 Home  
- 🔍 Predict Stroke Risk  
- 📊 Visual Insights  
- 🧪 SHAP Feature Interpretation
""")

st.image("https://cdn-icons-png.flaticon.com/512/4380/4380791.png", width=200)
st.caption("Developed by **Anjali Bista** · Advanced Data Science Project")
