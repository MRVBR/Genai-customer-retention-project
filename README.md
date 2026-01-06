📊 GenAI-Powered Customer Churn Analysis

A Streamlit-based web application that predicts customer churn using Machine Learning, provides explainable insights, and generates AI-driven retention emails.

🚀 Features

Churn prediction using Random Forest

High / Low churn risk classification

Human-readable churn reasons

Generative AI retention email (with safe fallback)

Interactive Streamlit dashboard

🛠 Tech Stack

Python

Streamlit

Pandas, NumPy

Scikit-learn

SHAP (conceptual explainability)

OpenAI API (optional)

🧠 ML Overview

Problem: Binary classification (Churn / No Churn)

Algorithm: Random Forest

Metrics: Accuracy, Precision, Recall, F1-score

▶️ Run Locally
pip install -r requirements.txt
python train_model.py
streamlit run app.py

🌐 Deployment

Deployed on Streamlit Community Cloud via GitHub.

🧪 Demo Inputs

High churn: Low tenure, high monthly charges, no tech support
Low churn: Long tenure, low charges, tech support enabled

🔐 API Key

OpenAI API key is optional.
If unavailable, the app shows a fallback email.
