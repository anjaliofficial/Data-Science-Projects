import streamlit as st

st.set_page_config(
    page_title="Stroke Risk Prediction System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded" 
)

# --- Sidebar Setup ---
st.sidebar.title(" Stroke Risk System")
st.sidebar.markdown("---")
st.sidebar.info("Navigate to the **Prediction** page to use the model, or the **Data Analysis** page for insights.")

# --- Main Page Content ---
st.header("A Data Science Project")
st.title("Predictive Modeling for Stroke Risk")
st.markdown("---")

col1, col2 = st.columns([1, 2])

with col1:
    # Using a relevant icon or placeholder image
    st.markdown("")
    st.markdown("""
    Developed by **Anjali Bista**
    
    _Leveraging Machine Learning to enhance proactive health assessment._
    """)

with col2:
    st.markdown("""
    Welcome to the **Stroke Risk Prediction System**. This application is built around a trained **Random Forest Classifier** model designed to assess a patient's probability of having a stroke based on various health and lifestyle factors.
    
    ### Key Features:
    
    1.  **🔍 Personalized Prediction:** Input a patient's health metrics to receive an instant, calculated stroke risk probability.
    2.  **🧪 Model Transparency (SHAP):** Understand *why* the model made a specific prediction by exploring the contribution of each individual feature using **SHAP (SHapley Additive exPlanations)**.
    3.  **📊 Data Insights:** (Requires a separate `data_analysis.py` page) Visualize key trends, distributions, and relationships within the dataset used to train the model.
    """)

st.markdown("---")

st.info("""
**Get Started:** Check the sidebar and navigate to the **Stroke Prediction** page (usually located under `pages/stroke_prediction.py`) to begin the assessment.
""")
