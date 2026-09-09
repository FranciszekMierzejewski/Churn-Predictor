import os
from dotenv import load_dotenv

import requests
import streamlit as st
import pandas as pd

load_dotenv()
API_URL = os.environ.get("CHURN_API_URL", "http://localhost:8000") # read env variable or fall back to local dev


st.set_page_config(
    page_title = "Telco Customer Churn Predictor",
)

st.title("Telco Customer Churn Predictor")
st.caption("Adjust decision threshold to control sensitivity of prediction")
#st.caption(f"Calling API at: {API_URL}")

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
        with st.spinner("Calculating churn probability..."):
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=30)

        response.raise_for_status()
        result = response.json()
        
        probability = result['probability']
        prediction = result['prediction']
        threshold_used = result['threshold_used']

        st.divider()
        st.subheader("Prediction result")
        st.caption(
            f"Production decision threshold: `{threshold_used:.2f}`"
        )

        first_metric, second_metric = st.columns(2)

        with first_metric:
            st.metric(
                "Churn probability",
                f"{probability * 100:.2f}%",
            )

        with second_metric:
            st.metric(
                "Prediction",
                "Churn" if prediction == 1 else "Retain",
            )

        if prediction == 1:
            st.error(
                "Predicted outcome: **Churn**. "
                "Consider proactive retention action."
            )
        else:
            st.success(
                "Predicted outcome: **Retain**. "
                "The customer is not classified as likely to churn "
                "at the model's tuned threshold."
            )

        st.subheader("Top factors driving this prediction")

        if result.get("top_factors"):
            factors_df = pd.DataFrame(result["top_factors"])

            factors_df = factors_df.rename(
                columns={
                    "feature": "Feature",
                    "shap_value": "SHAP impact",
                    "feature_value": "Scaled feature value",
                }
            )

            factors_df.index = factors_df.index + 1

            st.dataframe(
                factors_df,
                width="stretch",
            )

            st.caption(
                "Positive SHAP impact generally pushes the prediction "
                "toward churn; negative impact generally pushes it "
                "toward retention."
            )
        elif not include_shap:
            st.caption(
                "Enable “Show feature explanations with SHAP” to view "
                "the factors behind the prediction."
            )
        else:
            st.info(
                "The API returned no feature explanations for this prediction."
            )

    except requests.exceptions.Timeout:
        st.error(
            "The API request timed out. SHAP calculations can take longer "
            "than a basic prediction; try again or temporarily disable SHAP."
        )

    except requests.exceptions.HTTPError:
        detail = "The API returned an error."

        try:
            detail = response.json().get("detail", detail)
        except ValueError:
            pass

        st.error(f"Prediction request failed: {detail}")

    except requests.exceptions.RequestException as exception:
        st.error(f"Could not reach the prediction API: {exception}")

# az container stop --resource-group churn --name churn-api
# az container logs --resource-group churn --name churn-api
# python -m streamlit run streamlit_app.py