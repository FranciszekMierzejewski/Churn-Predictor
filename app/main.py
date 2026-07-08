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
        threshold = thresholds.get("recall_threshold", 0.4)
