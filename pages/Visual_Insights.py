import streamlit as st
import pandas as pd
import plotly.express as px
import os
import numpy as np

st.set_page_config(page_title="Visual Insights", layout="wide")
st.title("📊 Stroke Data Visual Insights")
st.markdown("Explore key trends and risk factor distributions in the dataset.")

# -----------------------------
# Paths (Assumes 'data' folder is one level up from 'pages')
# -----------------------------
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
# It's generally better practice to visualize the cleaned data used for training
CLEANED_CSV_PATH = os.path.join(DATA_DIR, "stroke_cleaned.csv") 

# -----------------------------
# Load data
# -----------------------------
@st.cache_data
def load_data(path):
    """Loads cleaned data. If not found, attempts to load raw data and clean BMI for visualization."""
    if not os.path.exists(path):
        st.warning(f"⚠️ Cleaned data file not found at: {path}. Attempting to load raw file.")
        # Fallback to raw file if cleaned is missing, assuming raw is named 'stroke_data.csv'
        raw_path = os.path.join(DATA_DIR, "stroke_data.csv")
        if os.path.exists(raw_path):
            df = pd.read_csv(raw_path)
        else:
            st.error(f"❌ Raw data file not found at: {raw_path}. Please place your stroke data CSV file in the 'data/' folder.")
            return pd.DataFrame()
    else:
        df = pd.read_csv(path)
        
    # Ensure consistency for visualization by handling any residual NaN in 'bmi' (e.g., if using raw data)
    if 'bmi' in df.columns:
        df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
        df['bmi'] = df['bmi'].fillna(df['bmi'].median())
        
    # Drop the 'Other' gender for cleaner visualization if present, as it was dropped in training
    if 'gender' in df.columns:
        df = df[df['gender'].str.lower() != 'other']

    return df

df = load_data(CLEANED_CSV_PATH)
if df.empty:
    st.stop()


# -----------------------------
# Age bins for visualization
# -----------------------------
age_bins = [0, 20, 40, 60, 80, 100]
age_labels = ['0-19', '20-39', '40-59', '60-79', '80+']
df['age_group'] = pd.cut(df['age'], bins=age_bins, right=False, labels=age_labels)


# -----------------------------
# Visualizations
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    # Stroke rate by age group
    st.subheader("1. Stroke Rate by Age Group")
    stroke_rate = df.groupby('age_group', observed=True)['stroke'].mean().reset_index()
    
    # Sort for better visualization
    stroke_rate = stroke_rate.sort_values('age_group')

    fig1 = px.bar(
        stroke_rate, x='age_group', y='stroke',
        labels={'stroke': 'Stroke Rate', 'age_group': 'Age Group'},
        title='Stroke Rate increases significantly with Age',
    )
    fig1.update_traces(texttemplate='%{y:.2%}', textposition='outside')
    fig1.update_layout(yaxis_tickformat=".2%")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    # Stroke count by gender
    st.subheader("2. Stroke Count by Gender")
    if 'gender' in df.columns:
        # Group by stroke=1 and count
        gender_count = df[df['stroke'] == 1].groupby('gender', observed=True).size().reset_index(name='count')
        
        fig2 = px.pie(
            gender_count, names='gender', values='count',
            title='Distribution of Stroke Cases by Gender'
        )
        # Ensure the 'Other' category is not displayed if it has 0 count
        fig2.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")
st.subheader("3. Relationship between Glucose, BMI, and Stroke")
if 'avg_glucose_level' in df.columns and 'bmi' in df.columns:
    # Scatter: Glucose vs BMI
    fig3 = px.scatter(
        df, x='avg_glucose_level', y='bmi', color=df['stroke'].map({0:'No Stroke', 1:'Stroke'}),
        labels={'color':'Stroke', 'avg_glucose_level': 'Avg. Glucose Level', 'bmi': 'BMI'}, 
        title='Glucose vs BMI colored by Stroke Outcome',
        hover_data=['age', 'hypertension', 'heart_disease'],
        color_discrete_map={'No Stroke': 'blue', 'Stroke': 'red'}
    )
    st.plotly_chart(fig3, use_container_width=True)


# -----------------------------
# Additional statistics
# -----------------------------
st.markdown("---")
st.subheader("📌 General Dataset Statistics")
col_stats1, col_stats2, col_stats3 = st.columns(3)

col_stats1.metric("Total Records", df.shape[0])
col_stats2.metric("Total Stroke Cases (1)", df['stroke'].sum())
col_stats3.metric("Overall Stroke Rate", f"{(df['stroke'].mean() * 100):.2f}%")