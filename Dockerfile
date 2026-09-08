FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY models/ ./models/
COPY data/X_train.csv ./data/X_train.csv
# main.py falls back to data/X_train.csv if shap_background.csv is missing

EXPOSE 8000

# Run uvicorn from inside the app/ subdirectory so bare imports resolve
WORKDIR /app/app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]