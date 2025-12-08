import streamlit as st
import pandas as pd
import pickle
import json

st.title("🏦 BNPL Credit Risk Scoring Platform")
st.caption("Built with XGBoost • Adapted from tedoaba/KAIM-W6 • Local demo")

# Load model & data
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)
model = load_model()

df = pd.read_csv("../data/bnpl_sample_500.csv")
st.write("### Sample dataset (500 BNPL customers)", df.head(10))

st.divider()

st.subheader("Live Risk Scoring")
with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        amount = st.number_input("Transaction Amount", 100, 10000, 1500)
        count = st.number_input("Transaction Count (last 6 mo)", 1, 100, 14)
        age = st.number_input("Age", 18, 80, 29)
    with col2:
        income = st.number_input("Annual Income", 20000, 200000, 68000)
        cards = st.number_input("Number of Credit Cards", 0, 15, 3)
        late = st.number_input("Days Late (worst)", 0, 90, 0)
        score = st.slider("Spend Score", 0.0, 1.0, 0.75, 0.01)

    submitted = st.form_submit_button("Calculate Default Risk")
    if submitted:
        input_data = pd.DataFrame([{
            "TransactionAmount": amount,
            "TransactionCount": count,
            "Age": age,
            "AnnualIncome": income,
            "NumCreditCards": cards,
            "PaymentHistoryDaysLate": late,
            "SpendScore": score
        }])
        prob = model.predict_proba(input_data)[0][1]
        st.metric("Default Probability", f"{prob:.1%}")
        st.write("Risk Level:", "🔴 HIGH RISK" if prob > 0.5 else "🟢 LOW RISK")
