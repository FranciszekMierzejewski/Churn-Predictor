import pandas as pd
from fastapi import FastAPI 
from mangum import Mangum # runs FastAPI under AWS Lambda 

from app.schemas import Customer, FeatureImpact, PredictionResponse
from app.preprocess import preprocess
from app.train import load_model
from app.predict import predict

app = FastAPI(
    title="Telco Churn Prediction API",
    description="Predicts customer churn probability from a trained logistic regression model.",
)

# defining api endpoint
@app.get("/")
def first_example():
    return {"message": "Hello, FastAPI!"}