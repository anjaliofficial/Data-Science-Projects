import streamlit as st

st.title("About This Project")

st.markdown("""
### 🧠 Stroke Prediction System
This project predicts the **probability of stroke** using health and lifestyle data.
The prediction model uses a **Random Forest Classifier** algorithm, trained on a publicly available dataset.

Crucially, the system utilizes **SHAP (SHapley Additive exPlanations)** to provide **individualized feature importance** and explain *why* the prediction was made.

---

### ⚙️ Core Tech Stack

* **Python** 
* **Streamlit** (For the interactive web application)
* **Scikit-learn + Random Forest**  (For the machine learning model)
* **SHAP**  (For model interpretability and explanations)
* **Plotly**  (For comprehensive data visualization in the 'Visual Insights' page)
* **Matplotlib + Seaborn**  (For static visualizations)

---

### 👩‍💻 Developed by
**Anjali Bista**  
Data Science Enthusiast · 2025  
""")

st.success("Built with ❤️ and Streamlit.")