import streamlit as st
import pandas as pd
import pickle
import os

st.set_page_config(page_title="BNPL Credit Risk Scoring", layout="wide")

st.title("BNPL Credit Risk Scoring Platform")
st.markdown("Predict default probability for Buy Now Pay Later customers using XGBoost.")

# Load model
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
    with open(model_path, "rb") as f:
        return pickle.load(f)

try:
    model = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False
    st.error("Model not found. Please run `python train_model.py` first.")

# Load and display sample data
st.subheader("Sample BNPL Dataset")
data_path = os.path.join(os.path.dirname(__file__), "..", "data", "bnpl_sample_500.csv")
df = pd.read_csv(data_path)
st.dataframe(df.head(10), use_container_width=True)

st.markdown("---")

# Input form
st.subheader("Risk Assessment")
st.markdown("Enter customer details to predict default probability:")

col1, col2 = st.columns(2)

with col1:
    transaction_amount = st.slider("Transaction Amount ($)", 100, 5000, 1500)
    transaction_count = st.slider("Transaction Count (6 months)", 1, 50, 10)
    age = st.slider("Age", 18, 70, 30)
    annual_income = st.slider("Annual Income ($)", 20000, 200000, 60000)

with col2:
    num_credit_cards = st.slider("Number of Credit Cards", 0, 10, 2)
    payment_history_days_late = st.slider("Payment History Days Late", 0, 90, 0)
    spend_score = st.slider("Spend Score", 0.0, 1.0, 0.5)

# Predict button
if st.button("Calculate Risk", type="primary"):
    if model_loaded:
        input_data = pd.DataFrame([{
            "TransactionAmount": transaction_amount,
            "TransactionCount": transaction_count,
            "Age": age,
            "AnnualIncome": annual_income,
            "NumCreditCards": num_credit_cards,
            "PaymentHistoryDaysLate": payment_history_days_late,
            "SpendScore": spend_score
        }])

        prob = model.predict_proba(input_data)[0][1]
        risk_level = "HIGH" if prob > 0.5 else "LOW"

        st.markdown("---")
        st.subheader("Prediction Results")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Default Probability", f"{prob:.2%}")
        with col2:
            if risk_level == "HIGH":
                st.error(f"Risk Level: {risk_level}")
            else:
                st.success(f"Risk Level: {risk_level}")
    else:
        st.error("Model not loaded. Cannot make predictions.")
