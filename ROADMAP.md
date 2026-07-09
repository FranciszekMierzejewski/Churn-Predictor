# Roadmap
 
## Completed
 1. Exploratory data analysis with business-driven hypotheses
 2. Preprocessing pipeline with feature engineering
 3. Model comparison - Logistic Regression, Random Forest, XGBoost
 4. SHAP explainability - global and per-customer
 5. Threshold tuning and calibration analysis
 6. Clean `app/` module with `preprocess.py`, `train.py`, `predict.py`, `schema.py`, `main.py`
 7. FastAPI service wrapping the model, with optional on-demand SHAP explanations
 8. Containerised the API with Docker and deployed to Azure Container Instances
 9. Streamlit frontend — Single Prediction tab: input a customer profile via dropdowns and number inputs, get a churn probability, an adjustable decision threshold slider, and a ranked SHAP feature-impact table explaining the prediction
 10. Deployed the frontend to Streamlit Community Cloud, connected to the live API
## Planned
 ### Streamlit Application — remaining tabs
   1. **Batch Prediction** - upload a CSV of customer records, get churn probabilities and risk levels for all customers, download results
   2. **Sensitivity Analysis** - simulate the effect of business decisions (e.g. a £10 price increase on certain services) on churn probability in real time, with before/after SHAP comparisons
   The app is designed to mirror the kind of tool a retention analyst at a telecoms company would actually use — translating model outputs into actionable business insight.
 
 ### Business Intelligence Dashboard (Power BI / Tableau)
   An executive-facing dashboard connected to the model outputs, covering:
   - Churn rate by contract type, internet service, and tenure band
   - Geographic distribution of churn risk across California zip codes
   - Top churn reason categories with trend over time
   - High-value customer (top CLTV quartile) churn monitoring
   - Model performance summary - AUC-ROC, recall, precision at chosen threshold
 ### Infrastructure improvements
   - CI/CD (GitHub Actions) to auto-build and push the Docker image on merge to `main`
   - Automated tests for `preprocess.py`/`predict.py` to catch schema drift before deployment
   - Consider AWS Lambda + API Gateway as a lower-cost alternative to always-on ACI, since Lambda scales to zero between requests