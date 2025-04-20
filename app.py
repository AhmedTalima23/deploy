import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# Load the trained model, scaler, and encoder
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
encoder = joblib.load('encoder.pkl')  # if used, or replicate encoding here

st.title("Employee Attrition Prediction")

# Input fields for the user
st.header("Enter Employee Details")

# Input: Age, Years at Company, Monthly Income, etc.
age = st.slider("Age", 18, 65, 30)
years_at_company = st.number_input("Years at Company", min_value=0, step=1, value=2)
monthly_income = st.number_input("Monthly Income", min_value=1000, step=100)

# Categorical Inputs
work_life_balance = st.selectbox("Work-Life Balance", ["Excellent", "Good", "Fair", "Poor"])
job_satisfaction = st.selectbox("Job Satisfaction", ["Very High", "High", "Medium", "Low"])
number_of_promotions = st.number_input("Number of Promotions", min_value=0, step=1)
overtime = st.selectbox("Overtime", ["Yes", "No"])
distance_from_home = st.number_input("Distance from Home (km)", min_value=0, step=1)
education_level = st.selectbox("Education Level", ["High School", "Associate Degree", "Bachelor's Degree", "Masters Degree", "PhD"])
number_of_dependents = st.number_input("Number of Dependents", min_value=0, step=1)
company_tenure = st.number_input("Company Tenure", min_value=0, step=1)
leadership_opportunities = st.selectbox("Leadership Opportunities", ["Yes", "No"])
innovation_opportunities = st.selectbox("Innovation Opportunities", ["Yes", "No"])
employee_recognition = st.selectbox("Employee Recognition", ["Very High", "High", "Medium", "Low"])

# One-hot encoded inputs (already encoded during model training)
gender = st.selectbox("Gender", ["Male", "Female"])
job_role = st.selectbox("Job Role", ["Finance", "Healthcare", "Media", "Technology"])
company_size = st.selectbox("Company Size", ["Medium", "Small"])
remote_work = st.selectbox("Remote Work", ["Yes", "No"])
company_reputation = st.selectbox("Company Reputation", ["Excellent", "Good", "Fair", "Poor"])
marital_status = st.selectbox("Marital Status", ["Married", "Single", "Divorced"])
job_level = st.selectbox("Job Level", ["Mid", "Senior"])

# Create a DataFrame from the user input
input_data = pd.DataFrame([{
    "Age": age,
    "Years at Company": years_at_company,
    "Monthly Income": monthly_income,
    "Work-Life Balance": work_life_balance,
    "Job Satisfaction": job_satisfaction,
    "Number of Promotions": number_of_promotions,
    "Overtime": overtime,
    "Distance from Home": distance_from_home,
    "Education Level": education_level,
    "Number of Dependents": number_of_dependents,
    "Company Tenure": company_tenure,
    "Leadership Opportunities": leadership_opportunities,
    "Innovation Opportunities": innovation_opportunities,
    "Employee Recognition": employee_recognition,
    "Gender": gender,
    "Job Role": job_role,
    "Company Size": company_size,
    "Remote Work": remote_work,
    "Company Reputation": company_reputation,
    "Marital Status": marital_status,
    "Job Level": job_level
}])

# Apply the same preprocessing steps (encoding, scaling)
# One-hot encoding manually (example for 'Gender')
input_data = pd.get_dummies(input_data, columns=["Gender", "Job Role", "Company Size", "Remote Work", "Company Reputation", "Marital Status", "Job Level"], drop_first=True)

# Now apply scaling
numeric_columns = ["Age", "Years at Company", "Monthly Income", "Number of Promotions", "Distance from Home", "Number of Dependents", "Company Tenure"]
input_data[numeric_columns] = scaler.transform(input_data[numeric_columns])

# Prediction button
if st.button("Predict"):
    try:
        prediction = model.predict(input_data)
        result = "🟢 Stayed" if prediction == 0 else "🔴 Left"
        st.success(f"Prediction: {result}")
    except Exception as e:
        st.error(f"Something went wrong during prediction: {e}")
