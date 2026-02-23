import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="BNPL Credit Risk", layout="wide")

st.title("BNPL Credit Risk Scoring")
st.caption("Predict default probability for Buy Now Pay Later customers")

# Load model
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    return pd.read_csv("bnpl_sample_500.csv")

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
            "Average Purchase ($)", 100, 5000, 1500,
            help="Typical dollar amount per BNPL transaction. Higher amounts = more exposure per purchase."
        )
        num_purchases = st.slider(
            "Purchases (6 months)", 1, 50, 10,
            help="Number of BNPL transactions in the last 6 months. Frequent users may indicate reliance on credit."
        )
        age = st.slider(
            "Age", 18, 70, 30,
            help="Customer age. Younger borrowers statistically have higher default rates."
        )
        annual_income = st.slider(
            "Annual Income ($)", 20000, 200000, 60000, step=5000,
            help="Yearly income before taxes. Higher income generally means better ability to repay."
        )
        num_credit_cards = st.slider(
            "Credit Cards", 0, 10, 2,
            help="Total number of credit cards. Many cards can indicate either good credit history or over-reliance on credit."
        )
        days_late = st.slider(
            "Worst Payment Delay (days)", 0, 90, 0,
            help="Longest time past due on any payment. Even one late payment is a strong predictor of future default."
        )
        debt_to_income = st.slider(
            "Debt-to-Income Ratio", 0.0, 1.0, 0.3, step=0.05,
            help="Monthly debt payments divided by monthly income. Example: 0.30 means 30% of income goes to debt. Above 0.40 is considered risky by most lenders."
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
                        {'range': [0, 30], 'color': '#2ecc71'},
                        {'range': [30, 60], 'color': '#f1c40f'},
                        {'range': [60, 100], 'color': '#e74c3c'}
                    ],
                }
            ))
            fig.update_layout(height=280, margin=dict(t=30, b=30))
            st.plotly_chart(fig, use_container_width=True)
            
            if prob > 0.5:
                st.error(f"**HIGH RISK** - {prob:.1%} default probability")
                st.markdown("This customer would likely be **declined** or require additional verification.")
            elif prob > 0.3:
                st.warning(f"**MEDIUM RISK** - {prob:.1%} default probability")
                st.markdown("This customer might be approved with **lower credit limits** or shorter terms.")
            else:
                st.success(f"**LOW RISK** - {prob:.1%} default probability")
                st.markdown("This customer would likely be **approved** for standard BNPL terms.")

            # Show key risk factors
            st.markdown("---")
            st.markdown("**Key Risk Factors:**")
            factors = []
            if debt_to_income > 0.4:
                factors.append(f"- High debt-to-income ({debt_to_income:.0%})")
            if days_late > 0:
                factors.append(f"- Payment history issues ({days_late} days late)")
            if age < 25:
                factors.append(f"- Younger age bracket ({age})")
            if annual_income < 40000:
                factors.append(f"- Lower income (${annual_income:,})")
            if num_purchases > 20:
                factors.append(f"- Heavy BNPL usage ({num_purchases} purchases)")
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
    st.caption("The model was trained on this synthetic dataset of 500 BNPL customers")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Customers", len(df))
    col2.metric("Default Rate", f"{df['Default'].mean():.1%}")
    col3.metric("Avg Income", f"${df['AnnualIncome'].mean():,.0f}")
    col4.metric("Avg Age", f"{df['Age'].mean():.0f}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Default Rate by Income**")
        df_temp = df.copy()
        df_temp['Income Bracket'] = pd.cut(
            df_temp['AnnualIncome'],
            bins=[0, 40000, 70000, 100000, 200000],
            labels=['<$40k', '$40-70k', '$70-100k', '>$100k']
        )
        rates = df_temp.groupby('Income Bracket')['Default'].mean() * 100
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
    with st.expander("View Raw Data"):
        st.dataframe(df, use_container_width=True, hide_index=True)

st.divider()
st.caption("XGBoost model trained on 500 synthetic BNPL applications | [GitHub](https://github.com/AnthonyB-316/bnpl-credit-risk-platform)")
