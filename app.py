
import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("loan_default_model.pkl")

st.title("💸 Loan Default Prediction App")

# Input fields
st.subheader("Enter Applicant Details:")
income = st.number_input("Annual Income", min_value=0)
loan_amnt = st.number_input("Loan Amount", min_value=0)
int_rate = st.slider("Interest Rate (%)", 0.0, 40.0, step=0.1)
dti = st.slider("Debt-to-Income Ratio", 0.0, 50.0, step=0.1)

if st.button("Predict Loan Default Risk"):
    input_data = pd.DataFrame([[income, loan_amnt, int_rate, dti]],
                              columns=['annual_inc', 'loan_amnt', 'int_rate', 'dti'])
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result:")
    if prediction == 1:
        st.error(f"⚠️ Loan Likely to Default (Risk Score: {prob:.2f})")
    else:
        st.success(f"✅ Loan Likely to Be Repaid (Confidence: {1 - prob:.2f})")
