import streamlit as st

# ==========================
# Page Configuration
# ==========================
st.set_page_config(
    page_title="Stroke Prediction System",
    page_icon="🧠",
    layout="wide"
)

# ==========================
# Sidebar Navigation Tip
# ==========================
st.sidebar.success("Select a page above 👆")

# ==========================
# Title & Introduction
# ==========================
st.title("🧠 Stroke Risk Prediction & Insights")
st.markdown("""
Welcome to the **Stroke Risk Prediction System**, powered by **XGBoost + Streamlit**.

Use the sidebar to navigate through the system:
- 🏠 **Home** – Overview of the project  
- 🔍 **Predict Stroke Risk** – Enter patient data and get prediction  
- 📊 **Visual Insights** – Explore interactive visualizations  
- 🧪 **SHAP Interpretation** – Understand feature impact on predictions  
- 📁 **About Project** – Learn more about this project
""")

# ==========================
# Image / Visual
# ==========================
st.image(
    "https://cdn-icons-png.flaticon.com/512/4380/4380791.png",
    width=200,
    caption="Stroke Risk Prediction"
)

# ==========================
# Divider & Footer
# ==========================
st.markdown("---")
st.markdown("""
💡 **Tip:** Navigate via the sidebar to access all pages of the system.
""")
st.caption("Developed by **Anjali Bista** · Data Science Project")
