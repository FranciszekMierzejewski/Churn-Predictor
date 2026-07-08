import os

import requests
import streamlit as st
import pandas as pd

API_URL = os.environ.get("CHURN_API_URL", "http://localhost:8000") # read env variable or fall back to local dev

st.title("Telco Customer Churn Predictor")
st.caption(f"Calling API at: {API_URL}")