import streamlit as st
import pandas as pd
import plotly.express as px

st.title("📊 Visual Insights")

# Load data
df = pd.read_csv("scripts/stroke_cleaned.csv")
df.fillna(df.median(numeric_only=True), inplace=True)

# Age groups
age_bins = pd.cut(df['age'], bins=[0, 20, 40, 60, 80, 100])
df['age_group'] = age_bins.astype(str)

stroke_rate = df.groupby('age_group')['stroke'].mean().reset_index()

fig = px.bar(stroke_rate, x='age_group', y='stroke', labels={'stroke':'Stroke Rate'}, title="Stroke Rate by Age Group")
st.plotly_chart(fig, use_container_width=True)
