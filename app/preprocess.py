import pandas as pd
import numpy as np

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Receoves raw Telco Customer Churn dataframe, returns cleaned, encoded and feature engineered dataframe prepared for modelling.
    """

    df = df.copy() # copy so edits won't alter original dataframe

    # Lower each column name
    df.columns = df.columns.str.lower()

    # Drop irrelevant information
    df = df.drop(columns=["customerid", "count", "lat long", "latitude", "longitude", "churn label", "churn score", "churn reason", "cltv", "city", "zip code", "country", "state"], errors = "ignore")

    df["total charges"] = pd.to_numeric(df["total charges"], errors = "coerce") # convert invalid to NaN
    df["total charges"] = df["total charges"].fillna(0) # convert NaN to to 0

    columns_to_change_to_binary = ["gender", "senior citizen", "partner", "dependents", "phone service", "multiple lines", "online security", "online backup", "device protection", "tech support", "streaming tv", "streaming movies", "paperless billing"]

    for column in columns_to_change_to_binary:
        df[column] = df[column].map({
            "Female":0, 
            "Male":1,
            "No":0,
            "No phone service":0,
            "No internet service":0,
            "Yes":1,
            })
        
    subscription_services = ["phone service", "multiple lines", "online security", "online backup", "device protection", "tech support", "streaming tv", "streaming movies"]

    df["number of subscriptions"] = df[subscription_services].apply(lambda row: (row == 1).sum(), axis=1)

    df["average charge per subscription"] = df.apply(lambda row: row["monthly charges"] / row["number of subscriptions"] if row["number of subscriptions"] > 0 else 0,
                                                    axis = 1)

    df["tenure month type"] = pd.cut( # group tenure months into bins
        df["tenure months"],
        bins = [0, 12, 24, 48, 72],
        labels = ["0 to 12", "12 to 24", "24 to 48", "48 to 72"] # 0<=m<12, 12<=m<24, 24<=m<48, 48<=m<=72
    )
    
    df = df.drop(columns=["tenure months"])

    df["contract"] = df["contract"].map({
    "Month-to-month":0,
    "One year":1,
    "Two year":2
    })

    # One hot encoding categorical variables
    df = pd.get_dummies(df, columns=["internet service", "payment method", "tenure month type"], dtype=int) 

    return df