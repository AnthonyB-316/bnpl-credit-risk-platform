import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="BNPL Credit Risk", layout="wide")

st.title("BNPL Credit Risk Scoring")
st.caption("XGBoost model trained on 30,000 real credit card customers (UCI Dataset)")

# Load model
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    return pd.read_csv("bnpl_credit_30k.csv")

try:
    model = load_model()
    model_loaded = True
except:
    model_loaded = False

tab1, tab2 = st.tabs(["Risk Calculator", "Dataset"])

with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Customer Info")

        avg_purchase = st.slider(
            "Average Monthly Bill ($)", 0, 100000, 20000, step=1000,
            help="Average monthly credit card bill amount. Higher balances = more exposure."
        )
        num_purchases = st.slider(
            "Months with Payments (last 6 mo)", 0, 12, 6,
            help="Number of months where customer made payments. More payments = better history."
        )
        age = st.slider(
            "Age", 21, 75, 35,
            help="Customer age. Younger borrowers statistically have higher default rates."
        )
        annual_income = st.slider(
            "Credit Limit ($)", 10000, 500000, 100000, step=10000,
            help="Customer's approved credit limit. Higher limits indicate better creditworthiness."
        )
        num_credit_cards = st.slider(
            "Education Level", 1, 4, 2,
            help="1=Graduate school, 2=University, 3=High school, 4=Other"
        )
        days_late = st.slider(
            "Worst Payment Delay (days)", 0, 90, 0, step=10,
            help="Longest payment delay in past 6 months. THIS IS THE #1 PREDICTOR - even 10 days late significantly increases risk."
        )
        debt_to_income = st.slider(
            "Credit Utilization", 0.0, 1.0, 0.3, step=0.05,
            help="Current balance / credit limit. Example: 0.30 means using 30% of available credit. Above 0.50 is high utilization."
        )

        calculate = st.button("Calculate Risk", type="primary", use_container_width=True)

    with col2:
        st.subheader("Risk Score")

        if calculate and model_loaded:
            input_data = pd.DataFrame([{
                "TransactionAmount": avg_purchase,
                "TransactionCount": num_purchases,
                "Age": age,
                "AnnualIncome": annual_income,
                "NumCreditCards": num_credit_cards,
                "PaymentHistoryDaysLate": days_late,
                "SpendScore": debt_to_income
            }])

            prob = model.predict_proba(input_data)[0][1]

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                number={'suffix': "%"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 25], 'color': '#2ecc71'},
                        {'range': [25, 50], 'color': '#f1c40f'},
                        {'range': [50, 100], 'color': '#e74c3c'}
                    ],
                }
            ))
            fig.update_layout(height=280, margin=dict(t=30, b=30))
            st.plotly_chart(fig, use_container_width=True)

            if prob > 0.5:
                st.error(f"**HIGH RISK** - {prob:.1%} default probability")
                st.markdown("This customer would likely be **declined** or require additional verification.")
            elif prob > 0.25:
                st.warning(f"**MEDIUM RISK** - {prob:.1%} default probability")
                st.markdown("This customer might be approved with **lower credit limits** or shorter terms.")
            else:
                st.success(f"**LOW RISK** - {prob:.1%} default probability")
                st.markdown("This customer would likely be **approved** for standard BNPL terms.")

            # Show key risk factors
            st.markdown("---")
            st.markdown("**Key Risk Factors:**")
            factors = []
            if days_late > 0:
                factors.append(f"- **Payment delay history ({days_late} days)** - strongest predictor!")
            if debt_to_income > 0.5:
                factors.append(f"- High credit utilization ({debt_to_income:.0%})")
            if age < 30:
                factors.append(f"- Younger age bracket ({age})")
            if annual_income < 50000:
                factors.append(f"- Lower credit limit (${annual_income:,})")
            if num_purchases < 4:
                factors.append(f"- Few payment months ({num_purchases}/12)")
            if factors:
                for f in factors:
                    st.markdown(f)
            else:
                st.markdown("- No significant risk factors identified")

        elif not model_loaded:
            st.error("Model not loaded")
        else:
            st.info("Adjust inputs and click Calculate")

with tab2:
    df = load_data()

    st.subheader("Training Dataset Overview")
    st.caption("Model trained on UCI Credit Card Default dataset (30,000 Taiwan credit card customers, 2005)")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Customers", f"{len(df):,}")
    col2.metric("Default Rate", f"{df['Default'].mean():.1%}")
    col3.metric("Avg Credit Limit", f"${df['AnnualIncome'].mean():,.0f}")
    col4.metric("Avg Age", f"{df['Age'].mean():.0f}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Default Rate by Credit Limit**")
        df_temp = df.copy()
        df_temp['Credit Bracket'] = pd.cut(
            df_temp['AnnualIncome'],
            bins=[0, 50000, 100000, 200000, 600000],
            labels=['<$50k', '$50-100k', '$100-200k', '>$200k']
        )
        rates = df_temp.groupby('Credit Bracket')['Default'].mean() * 100
        fig = px.bar(x=rates.index, y=rates.values, labels={'x': '', 'y': 'Default Rate (%)'})
        fig.update_traces(marker_color=['#e74c3c', '#f1c40f', '#2ecc71', '#2ecc71'])
        fig.update_layout(height=250, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Default Rate by Payment History**")
        df_temp = df.copy()
        df_temp['Payment History'] = df_temp['PaymentHistoryDaysLate'].apply(
            lambda x: 'Never Late' if x == 0 else '1-30 Days' if x <= 30 else '30+ Days'
        )
        rates = df_temp.groupby('Payment History')['Default'].mean() * 100
        rates = rates.reindex(['Never Late', '1-30 Days', '30+ Days'])
        fig = px.bar(x=rates.index, y=rates.values, labels={'x': '', 'y': 'Default Rate (%)'})
        fig.update_traces(marker_color=['#2ecc71', '#f1c40f', '#e74c3c'])
        fig.update_layout(height=250, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.markdown("**Feature Importance** (what the model learned)")
    importance_df = pd.DataFrame({
        'Feature': ['Payment Delay', 'Transaction Count', 'Credit Utilization', 'Bill Amount', 'Credit Limit', 'Education', 'Age'],
        'Importance': [0.805, 0.041, 0.040, 0.037, 0.033, 0.025, 0.020]
    })
    fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h')
    fig.update_traces(marker_color='#3498db')
    fig.update_layout(height=250, margin=dict(t=10, b=10), yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Payment history dominates - a single late payment is the strongest predictor of future default.")

    st.markdown("---")
    with st.expander("View Raw Data (30,000 records)"):
        st.dataframe(df, use_container_width=True, hide_index=True)

st.divider()
st.caption("XGBoost model trained on UCI Credit Card Default dataset | [GitHub](https://github.com/AnthonyB-316/bnpl-credit-risk-platform)")
