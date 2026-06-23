# Roadmap

## Completed
 1. Exploratory data analysis with business-driven hypotheses
 2. Preprocessing pipeline with feature engineering
 3. Model comparison - Logistic Regression, Random Forest, XGBoost
 4. SHAP explainability - global and per-customer
 5. Threshold tuning and calibration analysis
 6. Clean src/ module with preprocess.py, train.py, predict.py
 7. End-to-end demo notebook


## Planned
 ### Streamlit Application
   An interactive web app with three tabs:
    1. Single Prediction - input a customer profile via sliders and dropdowns, get a churn probability and a SHAP waterfall chart explaining the prediction
    2. Batch Prediction - upload a CSV of customer records, get churn probabilities and risk levels for all customers, download results
    3. Sensitivity Analysis - simulate the effect of business decisions (e.g. a £10 price increase on certain services) on churn probability in real time, with before/after SHAP comparisons

   The app is designed to mirror the kind of tool a retention analyst at a telecoms company would actually use — translating model outputs into actionable business insight.

   ### Business Intelligence Dashboard (Power BI / Tableau)
    An executive-facing dashboard connected to the model outputs, covering:
    - Churn rate by contract type, internet service, and tenure band
    - Geographic distribution of churn risk across California zip codes
    - Top churn reason categories with trend over time
    - High-value customer (top CLTV quartile) churn monitoring
    - Model performance summary - AUC-ROC, recall, precision at chosen threshold