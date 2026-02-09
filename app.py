import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="wide"
)

# -------------------------
# LOAD MODEL
# -------------------------
model = joblib.load("loan_approval_model.pkl")

# -------------------------
# CUSTOM STYLES
# -------------------------
st.markdown(
    """
    <style>
    .card {
        background-color: #f5f7fa;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------
# HEADER
# -------------------------
st.markdown(
    """
    <h1 style='text-align: center;'>🏦 Loan Approval Predictor</h1>
    <p style='text-align: center; color: gray;'>
    Enter applicant details in the sidebar to check loan approval status
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

# =========================
# SIDEBAR INPUTS
# =========================
with st.sidebar:
    st.header("📝 Applicant Details")

    no_of_dependents = st.number_input(
        "Number of Dependents",
        min_value=0,
        step=1,
        help="Number of people financially dependent on the applicant"
    )

    education = st.selectbox(
        "Education Level",
        ["Graduate", "Not Graduate"],
        help="Highest education qualification"
    )

    self_employed = st.selectbox(
        "Employment Type",
        ["No", "Yes"],
        help="Is the applicant self-employed?"
    )

    income_annum = st.number_input(
        "Annual Income (₹)",
        min_value=0,
        help="Total yearly income before tax"
    )

    loan_amount = st.number_input(
        "Loan Amount (₹)",
        min_value=0,
        help="Total loan amount requested"
    )

    loan_term = st.number_input(
        "Loan Term (months)",
        min_value=0,
        help="Loan repayment duration in months"
    )

    cibil_score = st.slider(
        "CIBIL Score",
        300,
        900,
        700,
        help="Credit score indicating repayment history"
    )

    st.subheader("💰 Asset Details")

    residential_assets_value = st.number_input(
        "Residential Assets (₹)",
        min_value=0,
        help="Value of owned residential property"
    )

    commercial_assets_value = st.number_input(
        "Commercial Assets (₹)",
        min_value=0,
        help="Value of owned commercial property"
    )

    luxury_assets_value = st.number_input(
        "Luxury Assets (₹)",
        min_value=0,
        help="Value of luxury items like cars or jewelry"
    )

    bank_asset_value = st.number_input(
        "Bank Assets (₹)",
        min_value=0,
        help="Savings, fixed deposits, and bank balances"
    )

    predict_btn = st.button("🔍 Predict Loan Status", use_container_width=True)

# -------------------------
# ENCODE INPUTS
# -------------------------
education = 1 if education == "Graduate" else 0
self_employed = 1 if self_employed == "Yes" else 0

# =========================
# MAIN LOGIC
# =========================
if predict_btn:

    input_data = pd.DataFrame({
        "no_of_dependents": [no_of_dependents],
        "education": [education],
        "self_employed": [self_employed],
        "income_annum": [income_annum],
        "loan_amount": [loan_amount],
        "loan_term": [loan_term],
        "cibil_score": [cibil_score],
        "residential_assets_value": [residential_assets_value],
        "commercial_assets_value": [commercial_assets_value],
        "luxury_assets_value": [luxury_assets_value],
        "bank_asset_value": [bank_asset_value]
    })

    # -------------------------
    # BUSINESS RULES (BANK LOGIC)
    # -------------------------
    if cibil_score < 600:
        st.markdown(
            """
            <div class="card">
                <h2 style="color: red;">❌ Loan Rejected</h2>
                <p>Reason: Credit score is too low</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.stop()

    if income_annum < loan_amount * 0.3:
        st.markdown(
            """
            <div class="card">
                <h2 style="color: red;">❌ Loan Rejected</h2>
                <p>Reason: Income is insufficient for requested loan</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.stop()

    # -------------------------
    # ML MODEL PREDICTION
    # -------------------------
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    # -------------------------
    # RESULT CARD
    # -------------------------
    if prediction == 1:
        st.markdown(
            f"""
            <div class="card">
                <h2 style="color: green;">✅ Loan Approved</h2>
                <h4>Confidence: {probability*100:.2f}%</h4>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div class="card">
                <h2 style="color: red;">❌ Loan Rejected</h2>
                <h4>Confidence: {(1-probability)*100:.2f}%</h4>
            </div>
            """,
            unsafe_allow_html=True
        )

    # -------------------------
    # FINANCIAL OVERVIEW CHART
    # -------------------------
    st.subheader("📊 Financial Overview")

    fig, ax = plt.subplots()
    ax.bar(
        ["Income", "Loan Amount", "Total Assets"],
        [
            income_annum,
            loan_amount,
            residential_assets_value + commercial_assets_value +
            luxury_assets_value + bank_asset_value
        ]
    )
    ax.set_ylabel("Amount (₹)")
    st.pyplot(fig)

    # -------------------------
    # CREDIT SCORE INSIGHT
    # -------------------------
    st.subheader("📈 Credit Score Insight")

    if cibil_score < 600:
        st.warning("Low credit score — high risk")
    elif cibil_score < 750:
        st.info("Average credit score — moderate risk")
    else:
        st.success("Excellent credit score — low risk")
