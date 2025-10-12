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
CSV_PATH = os.path.join(DATA_DIR, "stroke_data.csv") # Use 'stroke_data.csv' or 'stroke_cleaned.csv'

# -----------------------------
# Load data
# -----------------------------
@st.cache_data
def load_data(path):
    if not os.path.exists(path):
        st.error(f"Data file not found at: {path}. Please place your stroke data CSV file there.")
        return pd.DataFrame()
        
    df = pd.read_csv(path)
    # Fill missing BMI values with median for visualization consistency
    if 'bmi' in df.columns:
        df['bmi'] = df['bmi'].fillna(df['bmi'].median())
    return df

df = load_data(CSV_PATH)
if df.empty:
    st.stop()


# -----------------------------
# Age bins for visualization
# -----------------------------
age_bins = [0, 20, 40, 60, 80, 100]
df['age_group'] = pd.cut(df['age'], bins=age_bins, right=False, labels=[f"{a}-{b-1}" for a, b in zip(age_bins[:-1], age_bins[1:])])


# -----------------------------
# Visualizations
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    # Stroke rate by age group
    st.subheader("1. Stroke Rate by Age Group")
    stroke_rate = df.groupby('age_group', observed=True)['stroke'].mean().reset_index()
    
    fig1 = px.bar(
        stroke_rate, x='age_group', y='stroke',
        labels={'stroke': 'Stroke Rate', 'age_group': 'Age Group'},
        title='Stroke Rate increases significantly with Age',
        text=stroke_rate['stroke'].round(4) * 100 # Display percentage
    )
    fig1.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    fig1.update_layout(yaxis_tickformat=".2%")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    # Stroke count by gender
    st.subheader("2. Stroke Count by Gender")
    if 'gender' in df.columns:
        gender_count = df.groupby('gender', observed=True)['stroke'].sum().reset_index()
        fig2 = px.pie(
            gender_count, names='gender', values='stroke',
            title='Distribution of Stroke Cases by Gender'
        )
        st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")
st.subheader("3. Relationship between Glucose, BMI, and Stroke")
if 'avg_glucose_level' in df.columns and 'bmi' in df.columns:
    # Scatter: Glucose vs BMI
    fig3 = px.scatter(
        df, x='avg_glucose_level', y='bmi', color=df['stroke'].map({0:'No Stroke', 1:'Stroke'}),
        labels={'color':'Stroke', 'avg_glucose_level': 'Avg. Glucose Level', 'bmi': 'BMI'}, 
        title='Glucose vs BMI colored by Stroke Outcome',
        hover_data=['age', 'hypertension', 'heart_disease']
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