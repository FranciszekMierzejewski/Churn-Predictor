import pandas as pd
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def load_tuned_params() -> dict:
    """
    Loads C, l1_ratio, and recall_threshold from models/thresholds.pkl
    """
    path = BASE_DIR / "models" / "thresholds.pkl"

    if not path.exists():
        raise FileNotFoundError(
            "'models/thresholds.pkl' not found, run 'notebooks/05_model_tuning' first since train.py gets its hyperparameters from there"
        )

    params = joblib.load(path)

    for required_params in ("C", "l1_ratio", "recall_threshold"):
        if required_params not in params:
            raise KeyError(
                "Missing f'{required_params}', try rerunning 'notebooks/05_model_tuning' again"
            ) 

    return params


def train_model(X_train: pd.DataFrame, y_train: pd.DataFrame, C: float, l1_ratio: float) -> Pipeline:
    """
    Trains logistic regression pipeline on preprocessed training data. C value from grid search found in 05_model_tuning.ipynb
    """
    pipeline = Pipeline([
            ("scaler", StandardScaler()), # standardises features using z = (x - mean)/std to centre data around mean 0
            ("model", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42, C=C, l1_ratio=l1_ratio, penalty="elasticnet", solver="saga"))
    ])
    
    pipeline.fit(X_train, y_train)
    return pipeline


def save_model(pipeline: Pipeline, recall_threshold: float, C: float, l1_ratio: float) -> None:
    """
    Save fitted pipeline, thresholds, C and l1_ratio
    """
    models_dir = BASE_DIR / "models"
    models_dir.mkdir(exist_ok=True)

    joblib.dump(pipeline, models_dir / "logistic_regression.pkl")
    joblib.dump(
        {
            "recall_threshold": float(recall_threshold),
            "C" : float(C),
            "l1_ratio" : float(l1_ratio)

        },
        models_dir / "thresholds.pkl"
    )

    print(f"Model and thresholds saved. C = {C}, l1_ratio = {l1_ratio}, threshold = {recall_threshold}")


def load_model() -> tuple:
    """
    Load pipeline and thresholds
    """
    models_dir = BASE_DIR / "models"
    pipeline = joblib.load(models_dir / "logistic_regression.pkl")
    thresholds = joblib.load(models_dir / "thresholds.pkl")

    return pipeline, thresholds


if __name__ == "__main__":
    X_train = pd.read_csv("data/X_train.csv")
    y_train = pd.read_csv("data/y_train.csv").squeeze()

    # previously we used C that did not match grid search winner and the 0.4 threshold did not match the one 05_model_tuning computed or saved
    params = load_tuned_params()
    pipeline = train_model(X_train, y_train, C=params["C"], l1_ratio=params["l1_ratio"])
    save_model(pipeline, params["recall_threshold"], params["C"], params["l1_ratio"])
