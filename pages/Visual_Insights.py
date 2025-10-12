# pages/Visual_Insights.py
import streamlit as st
import pandas as pd
import plotly.express as px
import os
import numpy as np

st.set_page_config(page_title="Visual Insights", layout="wide")
st.title("📊 Stroke Data Visual Insights")

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
CSV_PATH = os.path.join(DATA_DIR, "stroke_data.csv")  # Ensure cleaned CSV is here

# -----------------------------
# Load data
# -----------------------------
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    # Fill missing BMI values with median
    if 'bmi' in df.columns:
        df['bmi'] = df['bmi'].fillna(df['bmi'].median())
    return df

df = load_data(CSV_PATH)

# -----------------------------
# Age bins for visualization
# -----------------------------
age_bins = [0, 20, 40, 60, 80, 100]
df['age_group'] = pd.cut(df['age'], bins=age_bins, right=False)

# -----------------------------
# Stroke rate by age group
# -----------------------------
stroke_rate = df.groupby('age_group', observed=True)['stroke'].mean().reset_index()
stroke_rate['age_group'] = stroke_rate['age_group'].astype(str)  # Avoid Interval issues

fig1 = px.bar(
    stroke_rate, x='age_group', y='stroke',
    labels={'stroke': 'Stroke Rate', 'age_group': 'Age Group'},
    title='Stroke Rate by Age Group',
    text=stroke_rate['stroke'].round(2)
)
fig1.update_layout(yaxis_tickformat=".2%")
st.plotly_chart(fig1, use_container_width=True)

# -----------------------------
# Stroke count by gender
# -----------------------------
if 'gender' in df.columns:
    gender_count = df.groupby('gender', observed=True)['stroke'].sum().reset_index()
    fig2 = px.pie(
        gender_count, names='gender', values='stroke',
        title='Stroke Count by Gender'
    )
    st.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# Scatter: Glucose vs BMI
# -----------------------------
if 'avg_glucose_level' in df.columns and 'bmi' in df.columns:
    fig3 = px.scatter(
        df, x='avg_glucose_level', y='bmi', color=df['stroke'].map({0:'No Stroke', 1:'Stroke'}),
        labels={'color':'Stroke'}, title='Glucose vs BMI'
    )
    st.plotly_chart(fig3, use_container_width=True)

# -----------------------------
# Additional insights
# -----------------------------
st.subheader("📌 Additional Statistics")
st.write("Total records:", df.shape[0])
st.write("Stroke cases:", df['stroke'].sum())
st.write("Non-stroke cases:", df.shape[0] - df['stroke'].sum())
