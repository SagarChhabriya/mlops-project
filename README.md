# Customer Churn Prediction – End-to-End MLOps Project

This project is an end-to-end machine learning application for predicting customer churn using Logistic Regression. It covers the complete workflow from model training and experiment tracking to deployment as a production-ready web application.

The purpose of this project is to practice and demonstrate real-world MLOps concepts such as reproducibility, model versioning, containerization, CI/CD, and cloud deployment.

---

## Project Overview

Customer churn prediction helps telecom companies identify customers who are likely to leave based on their behavior and usage patterns.
This application predicts churn using the following features:

* Call Failure
* Complains
* Subscription Length
* Charge Amount
* Seconds of Use
* Frequency of SMS
* Distinct Called Numbers
* Age

The trained model is served using a Streamlit application that allows users to enter customer details and receive churn probability in real time.

---

## Model Details

* Algorithm: Logistic Regression
* Models:

  * Baseline model
  * Tuned model
* Preprocessing:

  * Feature scaling using StandardScaler

Model artifacts are stored in:

```
models/
├── feature_names.pkl  
├── logistic_regression_baseline.pkl  
├── logistic_regression_tuned.pkl  
```

---

## Tech Stack

* Machine Learning: scikit-learn, pandas, numpy
* Experiment Tracking: Weights & Biases (W&B)
* Web App: Streamlit
* Containerization: Docker, Docker Hub
* CI/CD: GitHub Actions
* Cloud Deployment: Google Cloud Run, Google Artifact Registry

---

## Workflow

```
Data → Model Training → W&B Tracking & Artifacts  
                   ↓  
               Saved Models  
                   ↓  
            Streamlit Application  
                   ↓  
               Docker Image  
                   ↓  
           GitHub Actions Pipeline  
                   ↓  
        Google Artifact Registry  
                   ↓  
              Google Cloud Run  
```

---

## Features

* Interactive UI for churn prediction
* Real-time churn probability
* Risk categorization (Low, Medium, High)
* Model insights and interpretation
* Error handling for missing or mismatched features

---

## Running Locally

1. Clone the repository

```bash
git clone https://github.com/SagarChhabriya/mlops-project.git
cd mlops-project
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Make sure the following files exist in the `models/` directory:

* logistic_regression_tuned.pkl
* scaler.pkl
* feature_names.pkl

4. Run the app

```bash
streamlit run app.py
```

---

## Running with Docker

Build the image:

```bash
docker build -t churn-prediction-app .
```

Run the container:

```bash
docker run -p 8501:8501 churn-prediction-app
```

Open:

```
http://localhost:8501
```

- Docker Image: https://hub.docker.com/repository/docker/sagarchhabriya/my-mlops-project

---

## CI/CD Pipeline

GitHub Actions is used to automate:

* Docker image build
* Push to Google Artifact Registry
* Deployment to Google Cloud Run

This ensures that any update to the main branch is automatically deployed.

---

## Experiment Tracking

Weights & Biases is used for:

* Logging experiments
* Tracking hyperparameters and metrics
* Managing trained models as artifacts

This makes the training process reproducible and easy to compare.

---

## Author

Sagar Chhabriya

This project was built as a personal learning project to understand how machine learning systems are trained, deployed, and maintained in a production environment.
****


