import streamlit as st
import pandas as pd
import plotly.express as px

st.title("📊 Visual Health Insights")

df = pd.read_csv("data/stroke_data.csv")
df['bmi'].fillna(df['bmi'].median(), inplace=True)
df['age_group'] = pd.cut(df['age'], bins=[0,20,40,60,80,120], labels=['0-20','21-40','41-60','61-80','81+']).astype(str)

# Stroke rate by age
fig1 = px.histogram(df, x='age_group', color='stroke', barmode='group', title="Stroke Rate by Age Group")
st.plotly_chart(fig1, use_container_width=True)

# Glucose level distribution
fig2 = px.box(df, x='stroke', y='avg_glucose_level', color='stroke', title="Glucose Level vs Stroke")
st.plotly_chart(fig2, use_container_width=True)
