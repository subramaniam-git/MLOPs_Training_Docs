#!/bin/bash

###############################################################################
# PROJECT CONFIGURATION (CHANGE ONLY THESE IF NEEDED)
###############################################################################

# Root project directory name
PROJECT_NAME="mlops-e2e"

# S3 bucket where FINAL trained models are stored
# ⚠️ YOU MUST CREATE THIS BUCKET MANUALLY IN AWS
S3_MODEL_BUCKET="mlops-models"

# S3 key where the latest production model is stored
S3_MODEL_KEY="latest/model.pkl"

###############################################################################
# DO NOT MODIFY BELOW UNLESS YOU KNOW WHAT YOU ARE DOING
###############################################################################

echo "Creating MLOps project structure: $PROJECT_NAME"

mkdir -p $PROJECT_NAME
cd $PROJECT_NAME || exit 1

###############################################################################
# DIRECTORY STRUCTURE
###############################################################################

# Core application folders
mkdir -p api ui src

# Data folders (Git ignored, DVC managed)
mkdir -p data/raw data/processed

# Model folder (runtime only, populated dynamically)
mkdir -p models

# Container & Kubernetes
mkdir -p docker k8s

###############################################################################
# DATA VALIDATION SCRIPT
###############################################################################
# Purpose:
# - Validate dataset before preprocessing/training
# - Catch missing values & class imbalance early
###############################################################################

cat <<EOF > src/data_validation.py
import pandas as pd

def main():
    df = pd.read_csv("data/raw/reviews.csv")

    print("Total rows:", len(df))
    print("\\nMissing values:")
    print(df.isnull().sum())

    print("\\nClass distribution:")
    print(df["sentiment"].value_counts())

if __name__ == "__main__":
    main()
EOF

###############################################################################
# DATA PREPROCESSING SCRIPT
###############################################################################
# Purpose:
# - Clean raw data
# - Create deterministic processed dataset
# - Output tracked by DVC
###############################################################################

cat <<EOF > src/preprocessing.py
import pandas as pd

def main():
    df = pd.read_csv("data/raw/reviews.csv")

    # Basic text normalization (can be extended later)
    df["review_text"] = df["review_text"].str.lower()

    # Save processed data
    df.to_csv("data/processed/clean.csv", index=False)

    print("✅ Preprocessing completed")

if __name__ == "__main__":
    main()
EOF

###############################################################################
# TRAINING SCRIPT (VERY IMPORTANT)
###############################################################################
# Responsibilities:
# 1. Train model using latest processed data
# 2. Save model locally (for DVC versioning)
# 3. Upload FINAL model to S3 (for serving)
#
# 🔑 DVC handles VERSIONING
# 🔑 S3 handles SERVING
###############################################################################

cat <<EOF > src/train.py
import pandas as pd
import joblib
import boto3
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ---------------------------
# CHANGE THESE IF NEEDED
# ---------------------------
S3_BUCKET = "$S3_MODEL_BUCKET"
S3_KEY = "$S3_MODEL_KEY"
# ---------------------------

def main():
    df = pd.read_csv("data/processed/clean.csv")

    X = df["review_text"]
    y = df["sentiment"]

    vectorizer = TfidfVectorizer()
    X_vec = vectorizer.fit_transform(X)

    model = LogisticRegression(max_iter=200)
    model.fit(X_vec, y)

    # Save locally (tracked by DVC)
    os.makedirs("models", exist_ok=True)
    joblib.dump((model, vectorizer), "models/model.pkl")

    # Upload to S3 for serving
    s3 = boto3.client("s3")
    s3.upload_file("models/model.pkl", S3_BUCKET, S3_KEY)

    print("✅ Model trained and uploaded to S3")

if __name__ == "__main__":
    main()
EOF

###############################################################################
# FASTAPI APPLICATION
###############################################################################
# Purpose:
# - Stateless container
# - Downloads latest model from S3 at startup
# - Never bundles model inside image
###############################################################################

cat <<EOF > api/main.py
from fastapi import FastAPI
import joblib
import boto3
import os

# ---------------------------
# CHANGE THESE IF NEEDED
# ---------------------------
S3_BUCKET = "$S3_MODEL_BUCKET"
S3_KEY = "$S3_MODEL_KEY"
LOCAL_MODEL_PATH = "models/model.pkl"
# ---------------------------

app = FastAPI()

def download_model():
    if not os.path.exists(LOCAL_MODEL_PATH):
        os.makedirs("models", exist_ok=True)
        s3 = boto3.client("s3")
        s3.download_file(S3_BUCKET, S3_KEY, LOCAL_MODEL_PATH)

# Download model ONCE at startup
download_model()

model, vectorizer = joblib.load(LOCAL_MODEL_PATH)

@app.post("/predict")
def predict(text: str):
    vec = vectorizer.transform([text])
    prediction = model.predict(vec)[0]
    return {"sentiment": prediction}
EOF

###############################################################################
# STREAMLIT UI
###############################################################################
# Purpose:
# - Simple frontend
# - Talks ONLY to FastAPI
###############################################################################

cat <<EOF > ui/app.py
import streamlit as st
import requests

# ⚠️ When running locally
API_URL = "http://localhost:8000/predict"

st.title("Feedback Sentiment Analyzer")

text = st.text_area("Enter feedback")

if st.button("Predict"):
    response = requests.post(API_URL, params={"text": text})
    st.json(response.json())
EOF

###############################################################################
# DOCKERFILES (STATELESS BY DESIGN)
###############################################################################
# ❌ No model inside image
# ✅ Model fetched at runtime
###############################################################################

cat <<EOF > docker/Dockerfile.api
FROM python:3.10-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api api

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
EOF

cat <<EOF > docker/Dockerfile.ui
FROM python:3.10-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ui ui

CMD ["streamlit", "run", "ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
EOF

###############################################################################
# REQUIREMENTS (MINIMAL & PROJECT-SPECIFIC)
###############################################################################

cat <<EOF > requirements.txt
fastapi
uvicorn
streamlit
pandas
scikit-learn
joblib
requests
boto3
EOF

###############################################################################
# DVC PIPELINE
###############################################################################
# Purpose:
# - Reproducible ML workflow
# - Automatic dependency tracking
###############################################################################

cat <<EOF > dvc.yaml
stages:
  preprocess:
    cmd: python src/preprocessing.py
    deps:
      - data/raw/reviews.csv
    outs:
      - data/processed/clean.csv

  train:
    cmd: python src/train.py
    deps:
      - data/processed/clean.csv
    outs:
      - models/model.pkl
EOF

###############################################################################
# GITIGNORE (STRICT SEPARATION)
###############################################################################

cat <<EOF > .gitignore
data/raw/
data/processed/
models/
mlruns/
__pycache__/
.env
EOF

echo "✅ Self-documented MLOps project created successfully"
