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
│   └── thresholds.pkl
├── notebooks/
│   ├── .ipynb_checkpoints/
│   ├── 01_eda.ipynb                # Exploratory data analysis
│   ├── 02_preprocessing.ipynb      # Cleaning, encoding, feature engineering
│   ├── 03_modelling.ipynb          # Model comparison (LR, RF, XGBoost)
│   ├── 04_shap.ipynb               # SHAP explainability analysis
│   ├── 05_model_tuning.ipynb       # Regularisation, threshold tuning, calibration
│   └── 06_demo.ipynb               # End-to-end demo on sample customer profiles
├── telco_env/                     # Local virtual environment (not tracked by git)
├── streamlit_app.py                # Interactive frontend calling the deployed API
├── Dockerfile                     # Container image for the FastAPI service
├── requirements.txt
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
- **Explainability**: SHAP values are computed on demand (toggleable in the UI) using a precomputed 100-row background sample, keeping default requests fast.
---
 
## Key Findings
 
| Model | AUC-ROC | Recall | Precision | F1 |
|---|---|---|---|---|
| Logistic Regression | **0.859** | **0.809** | 0.531 | 0.641 |
| Random Forest | 0.847 | 0.502 | 0.670 | 0.574 |
| XGBoost | 0.836 | 0.659 | 0.569 | 0.610 |
 
**Logistic Regression outperforms both ensemble methods** on AUC-ROC and recall — suggesting the churn signal in this dataset is largely linear in nature. This is a meaningful result: the simplest model wins, and is also the most interpretable.
 
**Top churn drivers identified by SHAP:**
1. **Tenure months** — shorter tenure strongly predicts churn
2. **Monthly charges** — higher charges increase churn risk
3. **Contract type** — month-to-month customers churn at 43% vs 3% for two-year contracts
4. **Fiber optic internet** — Fiber Optic customers churn at a disproportionately high rate
5. **Dependents** — customers without dependents are significantly more likely to churn
---
 
## Methodology
 
### Class Imbalance
The dataset is imbalanced (~73.5% retained, ~26.5% churned). `class_weight="balanced"` was used to penalise misclassification of the minority class during training. This was chosen because:
- It avoids introducing synthetic data noise
- AUC-ROC and F1 were used as primary metrics, not accuracy

### Model Selection
All models were evaluated using **5-fold Stratified Cross-Validation** to preserve class balance across folds. AUC-ROC was the primary metric given the class imbalance.
 
### Threshold Tuning
The default decision threshold of 0.5 was tuned to **0.4** to maximise recall whilst preserving precision at an acceptable level. In a churn use case, the cost of missing a churner (false negative) is higher than the cost of a wasted retention offer (false positive). The live app also exposes this as an adjustable slider, so the sensitivity/precision trade-off can be explored interactively rather than fixed at build time.
 
### Regularisation
GridSearchCV over `C ∈ {0.001, 0.01, 0.1, 1, 10, 100}` with ElasticNet penalty identified **C=0.1** as optimal. AUC-ROC plateaus beyond this value with negligible overfitting.
 
### Calibration
A calibration curve confirmed that predicted probabilities closely reflect true churn rates, making the model suitable for risk scoring - not just binary classification.
 
---
 
## Explainability
 
SHAP (SHapley Additive exPlanations) values are computed using `shap.LinearExplainer` against a fixed 100-row background sample. This produces:
 
- **Global summary plot** - which features drive churn across the entire customer base
- **Feature importance ranking** - by mean absolute SHAP value
- **Dependence plots** - how churn risk changes across the range of each top feature
- **Per-prediction feature ranking** - exactly which features pushed a given customer toward or away from churning, surfaced live in the app when the SHAP toggle is enabled
---
 
## API
 
`POST /predict` accepts a customer profile and returns a churn probability, prediction, and (optionally) per-feature SHAP impact. See the [Swagger docs](http://churnpredictorapi.swedencentral.azurecontainer.io:8000/docs) for the full schema and to try it directly.
 
```json
{
  "probability": 0.62,
  "prediction": 1,
  "threshold_used": 0.4,
  "top_factors": [
    {"feature": "contract", "shap_value": 0.59, "feature_value": -0.83}
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
Download the dataset from [Kaggle](https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset), place `Telco_Customer_Churn.xlsx` in `data/`, then run the notebooks in order:
```
01_eda.ipynb → 02_preprocessing.ipynb → 03_modelling.ipynb → 04_shap.ipynb → 05_model_tuning.ipynb
```
Then train and save the model:
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
 
```
fastapi
uvicorn
pydantic
pandas
numpy
scikit-learn
xgboost
imbalanced-learn
shap
joblib
openpyxl
streamlit
requests
python-dotenv
```
 
---
 
## Licence
 
MIT
 