import os
from dotenv import load_dotenv

import requests
import streamlit as st
import pandas as pd

load_dotenv()
API_URL = os.environ.get("CHURN_API_URL", "http://localhost:8000") # read env variable or fall back to local dev

st.title("Telco Customer Churn Predictor")
st.caption(f"Calling API at: {API_URL}")

with st.form("customer form"):
    first_col, second_col = st.columns(2)

    with first_col:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        tenure_months = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

    
    with second_col:
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        payment_method = st.selectbox("Payment Method",["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.0, step=10.0)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=840.0, step=10.0)

    include_shap = st.checkbox("Show feature explanations with SHAP", value=True)
    submitted = st.form_submit_button("Predict Churn")

if submitted:
    payload = {
        "Gender": gender,
        "Senior Citizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "Tenure Months": tenure_months,
        "Phone Service": phone_service,
        "Multiple Lines": multiple_lines,
        "Internet Service": internet_service,
        "Online Security": online_security,
        "Online Backup": online_backup,
        "Device Protection": device_protection,
        "Tech Support": tech_support,
        "Streaming TV": streaming_tv,
        "Streaming Movies": streaming_movies,
        "Contract": contract,
        "Paperless Billing": paperless_billing,
        "Payment Method": payment_method,
        "Monthly Charges": monthly_charges,
        "Total Charges": total_charges,
        "include_shap": include_shap,
    }

    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()

        st.subheader(f"Result with Decision Threshold: {result['threshold_used']}")
        
        if result["prediction"] == 1:
            st.error("**Churn**")
        else:
            st.success("**Retain**")

        st.metric("Probability", f"{result['probability'] * 100:.2f}%")

        st.subheader("Top factors driving this prediction")
        if result["top_factors"]:
            factors_df = pd.DataFrame(result["top_factors"])
            st.dataframe(factors_df, use_container_width=True)
        else:
            st.caption("Enable the SHAP checkbox above to see feature explanations.")

    except requests.exceptions.RequestException as e:
        st.error(f"Request to API failed: {e}")


# az container stop --resource-group churn --name churn-api