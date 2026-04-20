import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load('loan_model.pkl')

# Page config
st.set_page_config(page_title="Loan Approval Prediction", page_icon="🏦")
st.title("🏦 Loan Approval Prediction System")
st.markdown("Fill in the applicant details below to predict loan approval.")

# Input fields
col1, col2 = st.columns(2)

with col1:
    no_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=2)
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    income_annum = st.number_input("Annual Income (₹)", min_value=0, value=500000, step=10000)
    loan_amount = st.number_input("Loan Amount (₹)", min_value=0, value=1000000, step=10000)

with col2:
    loan_term = st.number_input("Loan Term (months)", min_value=1, max_value=360, value=12)
    cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900, value=650)
    residential_assets_value = st.number_input("Residential Assets Value (₹)", min_value=0, value=1000000, step=10000)
    commercial_assets_value = st.number_input("Commercial Assets Value (₹)", min_value=0, value=500000, step=10000)
    luxury_assets_value = st.number_input("Luxury Assets Value (₹)", min_value=0, value=500000, step=10000)

bank_asset_value = st.number_input("Bank Asset Value (₹)", min_value=0, value=500000, step=10000)

# Encode inputs
education_enc    = 0 if education == "Graduate" else 1
self_employed_enc = 1 if self_employed == "Yes" else 0

# Predict button
if st.button("🔍 Predict Loan Status"):
    input_data = np.array([[
        no_of_dependents,
        education_enc,
        self_employed_enc,
        income_annum,
        loan_amount,
        loan_term,
        cibil_score,
        residential_assets_value,
        commercial_assets_value,
        luxury_assets_value,
        bank_asset_value
    ]])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    st.markdown("---")
    if prediction == 0:
        st.success("✅ Loan APPROVED")
        st.info(f"Approval Confidence: {round(probability[0] * 100, 2)}%")
    else:
        st.error("❌ Loan REJECTED")
        st.info(f"Rejection Confidence: {round(probability[1] * 100, 2)}%")