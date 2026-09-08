# Telco Customer Churn Predictor
 
An end-to-end machine learning pipeline that predicts customer churn for a telecommunications company, deployed as a live web application with SHAP-powered explainability to identify the key drivers of churn at an individual customer level.
 
**Live demo:** [churn-predictor-fm.streamlit.app](https://churn-predictor-fm.streamlit.app/)
**API docs:** [Swagger UI](http://churnpredictorapi.swedencentral.azurecontainer.io:8000/docs)
 
Built on the [IBM Telco Customer Churn dataset](https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset) (7,043 customers, 33 features).
 
---
 
## Project Structure
 
```
Churn-Predictor/
├── app/
│   ├── __pycache__/
│   ├── main.py                     # FastAPI app (predict + health endpoints)
│   ├── predict.py                   # Inference + optional SHAP explanation
│   ├── preprocess.py                 # Raw -> model-ready feature pipeline
│   ├── schemas.py                    # Pydantic request/response models
│   └── train.py                      # Model training and persistence
├── data/                          # Not tracked by git
│   ├── Telco_Customer_Churn.xlsx
│   ├── X_train.csv                  # Training features (SHAP background fallback)
│   ├── X_test.csv
│   ├── y_train.csv
│   └── y_test.csv
├── models/                        # Not tracked by git
│   ├── logistic_regression.pkl
│   ├── shap_background.csv         # Precomputed SHAP background sample
│   └── thresholds.pkl               # recall_threshold, C, l1_ratio
├── notebooks/
│   ├── .ipynb_checkpoints/
│   ├── 01_eda.ipynb                # Exploratory data analysis
│   ├── 02_preprocessing.ipynb      # Cleaning, encoding, feature engineering
│   ├── 03_modelling.ipynb          # Model comparison (LR, RF, XGBoost)
│   ├── 04_model_tuning.ipynb       # Elasticnet regularisation, threshold tuning, calibration
│   ├── 05_shap.ipynb               # SHAP explainability analysis on the tuned model
│   └── 06_demo.ipynb               # End-to-end demo on sample customer profiles
├── telco_env/                     # Local virtual environment (not tracked by git)
├── streamlit_app.py                # Interactive frontend calling the deployed API
├── Dockerfile                     # Container image for the FastAPI service
├── requirements.txt                # Minimal deps for running the API/Streamlit app
├── requirements-dev.txt            # Adds notebook-only deps (xgboost, matplotlib, seaborn, jupyter)
├── README.md
├── ROADMAP.md
├── LICENSE
├── .env                            # CHURN_API_URL (not tracked by git)
├── .gitignore
└── .dockerignore
```
 
---
 
## Architecture
 
```
Streamlit Cloud (frontend)  --HTTP-->  Azure Container Instance (FastAPI + model)
```
 
- **Model & API**: a Logistic Regression pipeline served via FastAPI, containerised with Docker, and deployed on Azure Container Instances (ACI).
- **Frontend**: a Streamlit app hosted on Streamlit Community Cloud, calling the API over HTTP for predictions and SHAP explanations.
- **Explainability**: SHAP values are computed on demand (toggleable in the UI) using a precomputed 200-row background sample, keeping default requests fast.
---
 
## Key Findings
 
### Model benchmark (5-fold CV, default hyperparameters, used to pick the model family)
 
| Model | AUC-ROC | Recall | Precision | F1 |
|---|---|---|---|---|
| Logistic Regression | **0.858** | **0.809** | 0.535 | 0.644 |
| Random Forest | 0.840 | 0.662 | 0.583 | 0.620 |
| XGBoost | 0.836 | 0.676 | 0.558 | 0.611 |
 
Logistic Regression wins on both AUC-ROC and recall, so it was carried forward for tuning.
 
### Final tuned model (elasticnet LR, C=0.1, l1_ratio=1.0, decision threshold=0.32)
 
| Metric | Default threshold (0.5) | Tuned threshold (0.32) |
|---|---|---|
| Recall | 78.3% | **90.4%** |
| Precision | 52.4% | 45.0% |
| F1 | 0.628 | 0.601 |
| AUC-ROC | 0.846 (test set; threshold-independent) | |
 
The threshold was chosen via out-of-fold F-beta (β=2) analysis on the training set, not the test set, to avoid leaking test data into a modelling decision. The test set above was touched exactly once, to report final numbers.
 
L1 regularisation (l1_ratio=1.0, i.e. pure Lasso) turned out to outperform ridge/mixed elasticnet at this C, and zeroed out **7 of 29 engineered features entirely** (`gender`, `device protection`, `monthly charges`, `internet service_DSL`, `number of subscriptions`, `payment method_Mailed check`, `tenure month type_24 to 48`) - the final model relies on the remaining 22.
 
**Top churn drivers identified by SHAP (on the final tuned model):**
1. **Contract type** — month-to-month customers churn at ~43% vs ~3% for two-year contracts
2. **Dependents** — customers without dependents are significantly more likely to churn
3. **Fiber optic internet** — Fiber Optic customers churn at a disproportionately high rate
4. **Tenure (0–12 months)** — customers in their first year churn at a much higher rate than longer-tenured customers
5. **No internet service** — customers without internet service are notably less likely to churn
These agree closely with the features L1 regularisation kept and weighted most heavily, which is a useful cross-check: two independent methods (coefficient magnitude and SHAP) converge on the same drivers.
 
---
 
## Methodology
 
### Class Imbalance
The dataset is imbalanced (~73.5% retained, ~26.5% churned). `class_weight="balanced"` was used to penalise misclassification of the minority class during training. This was chosen because:
- It avoids introducing synthetic data noise
- AUC-ROC and F1 were used as primary metrics, not accuracy
### Model Selection
All models were evaluated using **5-fold Stratified Cross-Validation** to preserve class balance across folds. AUC-ROC was the primary metric given the class imbalance.
 
### Threshold Tuning
The default decision threshold of 0.5 was tuned to **0.32**, chosen to maximise an F-beta(β=2) score - weighting recall above precision, since in a churn use case the cost of missing a churner (false negative) is higher than the cost of a wasted retention offer (false positive). Threshold selection was done on out-of-fold predictions from the training set, then evaluated once on the held-out test set, to avoid tuning against the same data used for final reporting. The live app also exposes this as an adjustable slider, so the sensitivity/precision trade-off can be explored interactively rather than fixed at build time.
 
### Regularisation
`GridSearchCV` over `C ∈ {0.001, 0.01, 0.1, 1, 10, 100}` and `l1_ratio ∈ {0, 0.25, 0.5, 0.75, 1}` (true elasticnet - both parameters tuned, not just C with an unset l1_ratio) identified **C=0.1, l1_ratio=1.0** as optimal, with mean CV AUC-ROC of 0.858. AUC-ROC plateaus beyond C=0.1 with negligible overfitting.
 
### Calibration
A calibration curve was used to check whether predicted probabilities reflect true churn rates. Isotonic calibration modestly improved the reliability curve's fit to the diagonal without materially changing AUC-ROC (0.8461 → 0.8458), consistent with AUC being threshold/calibration-invariant - calibration improves probability *quality* for risk-scoring use cases, not ranking ability.
 
---
 
## Explainability
 
SHAP (SHapley Additive exPlanations) values are computed using `shap.LinearExplainer` against a fixed 200-row background sample, run against the final tuned model (not an intermediate default-hyperparameter version). This produces:
 
- **Global summary plot** - which features drive churn across the entire customer base
- **Feature importance ranking** - by mean absolute SHAP value
- **Dependence plots** - how churn risk changes across the range of each top feature
- **Per-prediction feature ranking** - exactly which features pushed a given customer toward or away from churning, surfaced live in the app when the SHAP toggle is enabled
---
 
## API
 
`POST /predict` accepts a customer profile and returns a churn probability, prediction, and (optionally) per-feature SHAP impact. See the [Swagger docs](http://churnpredictorapi.swedencentral.azurecontainer.io:8000/docs) for the full schema and to try it directly.
 
```json
{
  "probability": 0.927,
  "prediction": 1,
  "threshold_used": 0.32,
  "top_factors": [
    {"feature": "tenure month type_0 to 12", "shap_value": 0.754, "feature_value": 1.503},
    {"feature": "contract", "shap_value": 0.491, "feature_value": -0.828}
  ]
}
```
 
---
 
## Running Locally
 
### API
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```
Visit `http://localhost:8000/docs`.
 
### Frontend
```bash
export CHURN_API_URL=http://localhost:8000   # or your deployed API URL
streamlit run streamlit_app.py
```
 
### Reproducing the model from scratch
```bash
pip install -r requirements-dev.txt
```
Download the dataset from [Kaggle](https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset), place `Telco_Customer_Churn.xlsx` in `data/`, then run the notebooks in order:
```
01_eda.ipynb → 02_preprocessing.ipynb → 03_modelling.ipynb → 04_model_tuning.ipynb → 05_shap.ipynb → 06_demo.ipynb
```
`04_model_tuning.ipynb` saves `models/logistic_regression.pkl` and `models/thresholds.pkl`; `05_shap.ipynb` reads those and additionally saves `models/shap_background.csv`.
 
Then, to retrain standalone using the exact hyperparameters found by the tuning notebook (reads `C`, `l1_ratio`, and `recall_threshold` from `models/thresholds.pkl` rather than any hardcoded value):
```bash
python -m app.train
```
 
### Deploying your own copy
```bash
docker build -t churn-predictor-api .
docker tag churn-predictor-api <your-dockerhub-username>/churn-predictor-api:latest
docker push <your-dockerhub-username>/churn-predictor-api:latest
 
az container create \
    --resource-group <your-resource-group> \
    --name churn-api \
    --image <your-dockerhub-username>/churn-predictor-api:latest \
    --dns-name-label <your-dns-label> \
    --ports 8000 \
    --cpu 1 \
    --memory 1.5 \
    --location <your-region> \
    --os-type Linux
```
 
---
 
## Requirements
 
**`requirements.txt`** (API + Streamlit deployment):
```
fastapi==0.120.0
pydantic==2.9.2
pandas==2.2.3
numpy==1.26.4
scikit-learn==1.9.0
shap==0.46.0
joblib==1.4.2
uvicorn==0.34.0
streamlit==1.58.0
python-dotenv==1.2.2
requests==2.32.3
```
 
**`requirements-dev.txt`** (adds notebook-only dependencies - install this on top of the above requirements to reproduce the notebooks):
```
-r requirements.txt
xgboost
matplotlib
seaborn
openpyxl
jupyter
```
 
---
 
## Licence
 
MIT