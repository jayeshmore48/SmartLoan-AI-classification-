import streamlit as st
import joblib
import pandas as pd   

st.set_page_config(
    page_title="SmartLoan AI",
    page_icon="🏦",
    layout="wide"
)
st.markdown("""
<style>

    /* Main background */
    .stApp {
        background-color: #f5f7fb;
    }

    /* Main title */
    h1 {
        color: #0f172a;
        font-weight: 700;
    }

    /* Section headers */
    h2, h3 {
        color: #1e3a8a;
    }

    /* Input labels */
    label {
        font-weight: 600;
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        background-color: #2563eb;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px;
        font-size: 16px;
        font-weight: 600;
    }

    .stButton > button:hover {
        background-color: #1d4ed8;
        color: white;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background-color: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
    }

</style>
""", unsafe_allow_html=True)

# Load trained model and preprocessor
model = joblib.load("loan_approval_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

st.title("🏦 SmartLoan AI")
st.subheader("Loan Approval & Credit Risk Prediction System")

st.write(
    "Enter customer details below to predict loan approval "
    "and assess credit risk."
)

st.success("Model and Preprocessor loaded successfully!")
st.header("👤 Customer Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age",
        min_value=18,
        max_value=80,
        value=25
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    education = st.selectbox(
        "Education",
        ["Graduate", "Not Graduate"]
    )

    years_employed = st.number_input(
        "Years Employed",
        min_value=0,
        max_value=50,
        value=5
    )


with col2:
    marital_status = st.selectbox(
        "Marital Status",
        ["Single", "Married"]
    )

    dependents = st.number_input(
        "Dependents",
        min_value=0,
        max_value=10,
        value=0
    )

    employment_type = st.selectbox(
        "Employment Type",
        ["Salaried", "Self-Employed", "Business", "Unemployed"]
    )

    monthly_income = st.number_input(
        "Monthly Income (₹)",
        min_value=0,
        value=0,
        step=5000
    )

st.header("💰 Financial Information")

col1, col2 = st.columns(2)

with col1:
    credit_score = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=850,
        value=750
    )

    loan_amount = st.number_input(
        "Loan Amount (₹)",
        min_value=0,
        value=0,
        step=10000
    )

    monthly_debt = st.number_input(
        "Monthly Debt Payment (₹)",
        min_value=0,
        value=0,
        step=1000
    )
    late_payments = st.number_input(
    "Late Payments",
    min_value=0,
    max_value=20,
    value=0
)


with col2:
    existing_loans = st.number_input(
        "Existing Loans",
        min_value=0,
        max_value=10,
        value=1
    )

    loan_term = st.number_input(
        "Loan Term (Months)",
        min_value=6,
        max_value=360,
        value=60
    )

    savings = st.number_input(
        "Savings (₹)",
        min_value=0,
        value=0,
        step=10000
    )


# Calculate DTI automatically
debt_to_income = (
    monthly_debt / monthly_income
    if monthly_income > 0 else 0
)

st.info(f"📊 Calculated Debt-to-Income Ratio: **{debt_to_income:.2f}**")

st.header("📋 Loan Information")

col1, col2 = st.columns(2)

with col1:
    loan_purpose = st.selectbox(
        "Loan Purpose",
        ["Home", "Education", "Business", "Personal", "Vehicle"]
    )

with col2:
    property_ownership = st.selectbox(
        "Property Ownership",
        ["Owned", "Rented", "Mortgaged"]
    )

# Feature Engineering
loan_to_income = loan_amount / monthly_income if monthly_income > 0 else 0
savings_to_loan = savings / loan_amount if loan_amount > 0 else 0
annual_income = monthly_income * 12
loan_to_annual_income = loan_amount / annual_income if annual_income > 0 else 0


# Prediction Button
if st.button("🔍 Predict Loan Approval"):

    customer_data = pd.DataFrame({
        "Age": [age],
        "Gender": [gender],
        "Marital_Status": [marital_status],
        "Dependents": [dependents],
        "Education": [education],
        "Employment_Type": [employment_type],
        "Years_Employed": [years_employed],
        "Monthly_Income": [monthly_income],
        "Credit_Score": [credit_score],
        "Existing_Loans": [existing_loans],
        "Loan_Amount": [loan_amount],
        "Loan_Term_Months": [loan_term],
        "Debt_to_Income": [debt_to_income],
        "Late_Payments": [late_payments],
        "Savings": [savings],
        "Loan_Purpose": [loan_purpose],
        "Property_Ownership": [property_ownership],
        "Loan_to_Income": [loan_to_income],
        "Savings_to_Loan": [savings_to_loan],
        "Annual_Income": [annual_income],
        "Loan_to_Annual_Income": [loan_to_annual_income]
    })

    # Preprocessing
    customer_processed = preprocessor.transform(customer_data)

    # Prediction
    prediction = model.predict(customer_processed)

    # Probability
    probability = model.predict_proba(customer_processed)[0]

    approved_probability = probability[
        list(model.classes_).index("Approved")
    ]

    # Risk
    if credit_score <= 600:
        risk_category = "High Risk"
    elif credit_score <= 700:
        risk_category = "Medium Risk"
    elif credit_score >= 700:
        risk_category = "Low Risk"
       

    # Result
    st.divider()
    st.header("📊 Loan Prediction Result")

    if prediction[0] == "Approved":
        st.success("✅ LOAN APPROVED")
    else:
        st.error("❌ LOAN REJECTED")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Approval Probability",
            f"{approved_probability * 100:.2f}%"
        )

    with col2:
        st.metric(
            "Credit Risk",
            risk_category
        )


        