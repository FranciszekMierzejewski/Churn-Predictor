import os
import pandas as pd
import logging
from fastapi import FastAPI, HTTPException
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

        if os.path.exists("models/shap_background.csv"): # indent if for repeat execution
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


@app.post("/predict", response_model=PredictionResponse)
def predict_churn(customer: Customer):
    try:
        pipeline, threshold, feature_columns, shap_background = get_model_artifacts()

        raw_row = pd.DataFrame([customer.model_dump(by_alias=True)])
        processed_row = preprocess(raw_row)
        processed_row = processed_row.reindex(columns=feature_columns, fill_value=0)

        include_shap = customer.include_shap

        result = predict(
            pipeline,
            shap_background,
            processed_row,
            threshold=threshold,
            include_shap=include_shap
        )

        response = PredictionResponse(
            probability=float(result["probability"]),
            prediction=int(result["prediction"]),
            threshold_used=float(threshold),
            top_factors=[],
        )

        if include_shap and "explanation" in result:
            explanation = result["explanation"]
            shap_values = explanation.get("shap values", [])
            feature_names = explanation.get("feature names", [])
            feature_values = explanation.get("feature values", [])

            impact_order = sorted(
                range(len(shap_values)),
                key=lambda i: abs(shap_values[i]),
                reverse=True,
            )[:10]

            response.top_factors = [
                FeatureImpact(
                    feature=str(feature_names[i]),
                    shap_value=float(shap_values[i]),
                    feature_value=float(feature_values[i]),
                )
                for i in impact_order
            ]

        return response
    
    except ValueError as exception:
        logger.exception("Value error during prediction")
        raise HTTPException(status_code=422, detail=f"Invalid input: {str(exception)}") # 422 = Unprocessible content
    except KeyError as exception:
        logger.exception("Missing expected feature during preprocessing")
        raise HTTPException(status_code=422, detail=f"Missing feature: {str(exception)}")
    except FileNotFoundError as exception:
        logger.exception("Model artifact file missing")
        raise HTTPException(status_code=500, detail=str(exception))
    except HTTPException: # keep original error, not 500
        raise
    except Exception: # catch all left
        logger.exception("Unexpected error during prediction")
        raise HTTPException(status_code=500, detail="Internal server error")