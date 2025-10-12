import streamlit as st
import pandas as pd
import plotly.express as px

st.title("📊 Visual Insights / EDA")

# Load CSV
csv_path = "stroke_cleaned.csv"
df = pd.read_csv(csv_path)
df['bmi'].fillna(df['bmi'].median(), inplace=True)

# Stroke rate by age group
age_bins = pd.cut(df['age'], bins=[0,18,35,50,65,80,100], labels=["0-18","19-35","36-50","51-65","66-80","81-100"])
stroke_rate = df.groupby(age_bins)['stroke'].mean().reset_index()
stroke_rate['stroke'] *= 100

fig1 = px.bar(stroke_rate, x='age', y='stroke', labels={'stroke':'Stroke Rate (%)','age':'Age Group'},
              title="Stroke Rate by Age Group")
st.plotly_chart(fig1, use_container_width=True)

# Stroke vs BMI
fig2 = px.scatter(df, x='bmi', y='stroke', color='gender', title="Stroke vs BMI")
st.plotly_chart(fig2, use_container_width=True)
