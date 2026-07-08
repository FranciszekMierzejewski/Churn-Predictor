from pydantic import BaseModel, Field
from typing import Literal

class Customer(BaseModel):
    """
    Input features for churn prediction, apart from excluded in preprocessing
    """

    # examples = [] shows up in Swagger UI as schema details, hint for that field
    gender: Literal["Male", "Female"] = Field(..., alias="Gender", examples=["Male"]) # literal to not have manual validation later
    senior_citizen: Literal["Yes", "No"] = Field(..., alias="Senior Citizen", examples=["No"])
    partner: Literal["Yes", "No"] = Field(..., alias="Partner", examples=["Yes"])
    dependents: Literal["Yes", "No"] = Field(..., alias="Dependents", examples=["No"])
    tenure_months: int = Field(..., alias="Tenure Months", ge=0, le=100, examples=[12]) # >= 0, <= 100
    phone_service: Literal["Yes", "No"] = Field(..., alias="Phone Service", examples=["Yes"])
    multiple_lines: Literal["Yes", "No", "No phone service"] = Field(
        ..., alias="Multiple Lines", examples=["No"]
    )
    internet_service: Literal["DSL", "Fiber optic", "No"] = Field(
        ..., alias="Internet Service", examples=["Fiber optic"]
    )
    online_security: Literal["Yes", "No", "No internet service"] = Field(
        ..., alias="Online Security", examples=["No"]
    )
    online_backup: Literal["Yes", "No", "No internet service"] = Field(
        ..., alias="Online Backup", examples=["Yes"]
    )
    device_protection: Literal["Yes", "No", "No internet service"] = Field(
        ..., alias="Device Protection", examples=["No"]
    )
    tech_support: Literal["Yes", "No", "No internet service"] = Field(
        ..., alias="Tech Support", examples=["No"]
    )
    streaming_tv: Literal["Yes", "No", "No internet service"] = Field(
        ..., alias="Streaming TV", examples=["Yes"]
    )
    streaming_movies: Literal["Yes", "No", "No internet service"] = Field(
        ..., alias="Streaming Movies", examples=["Yes"]
    )
    contract: Literal["Month-to-month", "One year", "Two year"] = Field(
        ..., alias="Contract", examples=["Month-to-month"]
    )
    paperless_billing: Literal["Yes", "No"] = Field(
        ..., alias="Paperless Billing", examples=["Yes"]
    )
    payment_method: Literal[
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ] = Field(..., alias="Payment Method", examples=["Electronic check"])
    monthly_charges: float = Field(..., alias="Monthly Charges", ge=0, examples=[85.5])
    total_charges: float = Field(..., alias="Total Charges", ge=0, examples=[1024.5])
    include_shap: bool = Field(default=False, alias="include_shap") # compute SHAP explanation if true

    model_config = {
        "populate_by_name": True,  # allows snake_case OR alias (e.g. "Senior Citizen")
        "json_schema_extra": {
            "example": {
                "Gender": "Male",
                "Senior Citizen": "No",
                "Partner": "Yes",
                "Dependents": "No",
                "Tenure Months": 12,
                "Phone Service": "Yes",
                "Multiple Lines": "No",
                "Internet Service": "Fiber optic",
                "Online Security": "No",
                "Online Backup": "Yes",
                "Device Protection": "No",
                "Tech Support": "No",
                "Streaming TV": "Yes",
                "Streaming Movies": "Yes",
                "Contract": "Month-to-month",
                "Paperless Billing": "Yes",
                "Payment Method": "Electronic check",
                "Monthly Charges": 85.5,
                "Total Charges": 1024.5,
                "include_shap": False
            }
        },
    }


class FeatureImpact(BaseModel):
    feature: str
    shap_value: float
    feature_value: float


class PredictionResponse(BaseModel):
    probability: float
    prediction: int
    threshold_used: float
    top_factors: list[FeatureImpact]
