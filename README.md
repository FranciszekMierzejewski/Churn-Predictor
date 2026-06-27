# Telco Customer Churn Predictor

An end-to-end machine learning pipeline that predicts customer churn for a telecommunications company, with SHAP-powered explainability to identify the key drivers of churn at an individual customer level.

Built on the [IBM Telco Customer Churn dataset](https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset) (7,043 customers, 33 features).

---

## Project Structure

```
Churn-Predictor/
├── data/                          # Raw data (not tracked by git)
│   └── Telco_Customer_Churn.xlsx
├── notebooks/
│   ├── 01_eda.ipynb               # Exploratory data analysis
│   ├── 02_preprocessing.ipynb     # Cleaning, encoding, feature engineering
│   ├── 03_modelling.ipynb         # Model comparison (LR, RF, XGBoost)
│   ├── 04_shap.ipynb              # SHAP explainability analysis
│   ├── 05_model_tuning.ipynb      # Regularisation, threshold tuning, calibration
│   └── 06_demo.ipynb              # End-to-end demo on two customer profiles
├── src/
│   ├── preprocess.py              # Preprocessing pipeline
│   ├── train.py                   # Model training and persistence
│   └── predict.py                 # Inference and SHAP explanation
├── models/                        # Saved model artifacts (not tracked by git)
│   ├── logistic_regression.pkl
│   └── thresholds.pkl
├── requirements.txt
├── README.md
└── ROADMAP.md
```

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
The dataset is imbalanced (~73.5% retained, ~26.5% churned). Rather than SMOTE, `class_weight="balanced"` was used to penalise misclassification of the minority class during training. This was chosen because:
- It avoids introducing synthetic data noise
- AUC-ROC and F1 were used as primary metrics, not accuracy
- Results were comparable to SMOTE in cross-validation experiments

### Model Selection
All models were evaluated using **5-fold Stratified Cross-Validation** to preserve class balance across folds. AUC-ROC was the primary metric given the class imbalance.

### Threshold Tuning
The default decision threshold of 0.5 was tuned to **0.4** to maximise recall whilst preserving precision at an acceptable level. In a churn use case, the cost of missing a churner (false negative) is higher than the cost of a wasted retention offer (false positive).

### Regularisation
GridSearchCV over `C ∈ {0.001, 0.01, 0.1, 1, 10, 100}` with ElasticNet penalty identified **C=0.1** as optimal. AUC-ROC plateaus beyond this value with negligible overfitting.

### Calibration
A calibration curve confirmed that predicted probabilities closely reflect true churn rates, making the model suitable for risk scoring - not just binary classification.

---

## Explainability

SHAP (SHapley Additive exPlanations) values are computed for every prediction using `shap.LinearExplainer`. This produces:

- **Global summary plot** - which features drive churn across the entire customer base
- **Feature importance ranking** - by mean absolute SHAP value
- **Dependence plots** - how churn risk changes across the range of each top feature
- **Waterfall plots** - why the model made a specific prediction for an individual customer

The waterfall plot is the key output: it shows exactly which features pushed a customer toward or away from churning, enabling targeted retention interventions.

---

## Reproducing the Project

### 1. Clone and install dependencies

```bash
git clone https://github.com/your-username/Churn-Predictor.git
cd Churn-Predictor
pip install -r requirements.txt
```

### 2. Add the data

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset) and place `Telco_Customer_Churn.xlsx` in the `data/` folder.

### 3. Run the notebooks in order

```
01_eda.ipynb → 02_preprocessing.ipynb → 03_modelling.ipynb → 04_shap.ipynb → 05_model_tuning.ipynb
```

### 4. Train and save the model

```bash
cd src
python train.py
```

### 5. Run the demo

Open `notebooks/06_demo.ipynb` and run all cells.

---

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
imbalanced-learn
shap
joblib
openpyxl
```

---

## Licence

MIT
