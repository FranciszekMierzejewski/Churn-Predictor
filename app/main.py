import os
import pandas as pd
import logging
from fastapi import FastAPI 
from mangum import Mangum # runs FastAPI under AWS Lambda 

from .schemas import Customer, FeatureImpact, PredictionResponse
from .preprocess import preprocess
from .train import load_model
from .predict import predict


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO) # log variable

app = FastAPI(
    title="Telco Churn Prediction API",
    description="Predicts customer churn probability from a trained logistic regression model.",
)

# load artifacts on first request for AWS Lambda to reuse the container for subsequent invocations
_pipeline = None
_threshold = None
_feature_columns = None
_shap_background = None


def get_model_artifacts():
    global _pipeline, _threshold, _feature_columns, _shap_background

    if _pipeline is None:
        pipeline, thresholds = load_model()
        threshold = thresholds.get("recall_threshold", 0.4) # get threshold, default to 0.4 if empty

    if os.path.exists("models/shap_background.csv"):
        shap_background = pd.read_csv("models/shap_background.csv")
        feature_columns = shap_background.columns.tolist()
        logger.info("Loaded SHAP background")
    elif os.path.exists("data/X_train.csv"):
        shap_background = pd.read_csv("data/X_train.csv")
        feature_columns = shap_background.columns.tolist()
        logger.warning("Shap background not found. Using full X_train")
    else:
        raise FileNotFoundError(
            "Neither shap_background.csv nor X_train.csv could be found."
        )

    _pipeline = pipeline
    _threshold = threshold
    _feature_columns = feature_columns
    _shap_background = shap_background

    logger.info("Model artifacts loaded successfully")

    return _pipeline, _threshold, _feature_columns, _shap_background


@app.get("/health") # pings health automatically to keep container alive every few seconds
def health():
    return {"status": "ok"}