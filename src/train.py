import pandas as pd
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def train_model(X_train: pd.DataFrame, y_train: pd.DataFrame, C: float = 0.1) -> Pipeline:
    """
    Trains logistic regression pipeline on preprocessed training data. C value from grid search found in 05_model_tuning.ipynb
    """
    pipeline = Pipeline([
            ("scaler", StandardScaler()), # standardises features using z = (x - mean)/std to centre data around mean 0
            ("model", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42, C=C, penalty="elasticnet", solver="saga"))
    ])
    
    pipeline.fit(X_train, y_train)
    return pipeline


def save_model(pipeline: Pipeline, threshold_recall: float = 0.4) -> None:
    """
    Save fitted pipeline and thresholds
    """
    os.makedirs("../models", exist_ok=True)

    joblib.dump(pipeline, "../models/logistic_regression.pkl")
    joblib.dump(
        {"recall_threshold": float(threshold_recall)},
        "../models/thresholds.pkl"
    )

    print("Model and thresholds saved.")


def load_model() -> tuple:
    """
    Load pipeline and thresholds
    """
    pipeline = joblib.load("../models/logistic_regression.pkl")
    thresholds = joblib.load("../models/thresholds.pkl")

    return pipeline, thresholds


if __name__ == "__main__":
    X_train = pd.read_csv("../data/X_train.csv")
    y_train = pd.read_csv("../data/y_train.csv").squeeze()

    pipeline = train_model(X_train, y_train, C = 0.1)

    save_model(pipeline, 0.4)
