# SmartLoan-AI-classification-

End-to-end ML application for loan approval prediction and credit risk assessment using Random Forest, SHAP, and Streamlit.




#  SmartLoan AI

### Loan Approval & Credit Risk Prediction System

SmartLoan AI is an end-to-end Machine Learning application that predicts loan approval and assesses customer credit risk based on financial, employment, and loan-related information.

The project uses Machine Learning models to support loan decision-making and provides an interactive web interface built with Streamlit.

---

## Project Overview

The objective of SmartLoan AI is to help financial institutions analyze loan applications and predict whether a loan is likely to be approved.

The system provides:

- Loan Approval Prediction
- Approval Probability
- Credit Risk Classification
- Feature Engineering
- Model Comparison
- Explainable AI using SHAP
- Interactive Streamlit Interface

> **Note:** This project is intended for educational and decision-support purposes. It should not be used as the sole basis for real-world lending decisions.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Random Forest
- Logistic Regression
- SHAP
- Matplotlib
- Streamlit
- Joblib

---

##  Dataset

The project uses a synthetic dataset containing **10,000 loan applications**.

### Main Features

- Age
- Gender
- Marital Status
- Dependents
- Education
- Employment Type
- Years Employed
- Monthly Income
- Credit Score
- Existing Loans
- Loan Amount
- Loan Term
- Debt-to-Income Ratio
- Late Payments
- Savings
- Loan Purpose
- Property Ownership

### Target Variable

`Loan_Approved`

The target contains:

- Approved
- Rejected

---

##  Machine Learning Workflow

```text
Raw Dataset
     ↓
Data Cleaning
     ↓
Missing Value Handling
     ↓
Categorical Data Cleaning
     ↓
Exploratory Data Analysis
     ↓
Feature Engineering
     ↓
Train-Test Split
     ↓
Data Preprocessing
     ↓
Model Training
     ↓
Model Evaluation
     ↓
Hyperparameter Tuning
     ↓
SHAP Explainability
     ↓
Model Saving
     ↓
Streamlit Deployment
