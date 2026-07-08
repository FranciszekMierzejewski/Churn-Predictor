import pandas as pd
import numpy as np
import shap
import joblib
import os
from sklearn.pipeline import Pipeline
from .train import load_model

def get_churn_probability(pipeline: Pipeline, customer_row: pd.DataFrame) -> float:
    """
    Assuming same features in X_train and customer_row, get probability that customer will churn.
    """
    return float(pipeline.predict_proba(customer_row)[:, 1][0])


def get_shap_explanation(pipeline: Pipeline, X_train: pd.DataFrame, customer_row: pd.DataFrame) -> dict:
    """
    Returns SHAP values and base value for a single customer.
    """
    logistic_regression = pipeline["model"]
    scaler = pipeline["scaler"]

    X_train_scaled = pd.DataFrame(
        scaler.transform(X_train),
        columns = X_train.columns
    )

    customer_row_scaled = pd.DataFrame(
        scaler.transform(customer_row),
        columns = customer_row.columns
    )

    masker = shap.maskers.Independent(X_train_scaled) # background dataset to simulate missing features, assumes features are independent and replaces with values sampled from X_train
    explainer = shap.LinearExplainer(logistic_regression, masker=masker) # compute SHAP values
    explanation = explainer(customer_row_scaled) # compute contribution from each feature in classification
    # phi_i(row) = weight_i * (x_i(row) - mean_i), for shap values with linear model

    return {
        "shap values" : explanation.values[0],
        "base value" : float(explainer.expected_value),
        "feature names" : list(X_train.columns),
        "feature values" : customer_row_scaled.iloc[0].tolist()
    }

def predict(pipeline: Pipeline, X_train: pd.DataFrame, customer_row: pd.DataFrame, threshold: float = 0.4, include_shap: bool = False) -> dict:
    """
    Prediction for a customer. Returns probability, churn prediction and SHAP explanation.
    """

    probability = get_churn_probability(pipeline, customer_row)
    prediction = int(probability >= threshold)

    result = {
        "probability": probability,
        "prediction": prediction
    }

    if include_shap:
        result["explanation"] = get_shap_explanation(pipeline, X_train, customer_row)

    return result