# main_app.py

import streamlit as st

st.set_page_config(
    page_title="Stroke Risk Prediction System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded" # Ensure sidebar is open by default
)

# --- Sidebar Setup ---
st.sidebar.title("🧠 Stroke Risk System")
st.sidebar.markdown("---") # Visual separator
st.sidebar.info("Navigate between prediction, data analysis, and model interpretation using the pages below.")

# --- Main Page Content ---
st.header("An Advanced Data Science Project")
st.title("🩺 Predictive Modeling for Stroke Risk")
st.markdown("---")

col1, col2 = st.columns([1, 2])

with col1:
    # Use a local or more stylized image if possible, but the current one works.
    st.image("https://cdn-icons-png.flaticon.com/512/4380/4380791.png", width=180)
    st.markdown("""
    Developed by **Anjali Bista**
    
    _Leveraging Machine Learning to enhance proactive health assessment._
    """)

with col2:
    st.markdown("""
    Welcome to the **Stroke Risk Prediction System** built on a robust Machine Learning model (likely **XGBoost** or **Random Forest**).
    
    This application offers a comprehensive approach to health assessment:
    
    ### Key Features:
    
    1.  **🔍 Personalized Prediction:** Input a patient's health metrics to receive an instant, calculated stroke risk probability.
    2.  **🧪 Model Transparency (SHAP):** Understand *why* the model made a specific prediction by exploring the contribution of each individual feature.
    3.  **📊 Data Insights:** Visualize key trends, distributions, and relationships within the dataset used to train the model.
    """)

st.markdown("---")

st.info("""
**Get Started:** Select a page from the sidebar to begin.
""")