import streamlit as st
import pandas as pd
import pickle
import os
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="BNPL Credit Risk Scoring", layout="wide")

st.title("BNPL Credit Risk Scoring Platform")
st.markdown("Predict default probability for Buy Now Pay Later customers using XGBoost.")

# Load model
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
    with open(model_path, "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "bnpl_sample_500.csv")
    return pd.read_csv(data_path)

try:
    model = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False
    st.error("Model not found. Please run `python train_model.py` first.")

df = load_data()

# Tabs
tab1, tab2 = st.tabs(["Risk Calculator", "Data Overview"])

with tab1:
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Customer Details")

        transaction_amount = st.slider("Transaction Amount ($)", 100, 5000, 1500)
        transaction_count = st.slider("Transaction Count (6 months)", 1, 50, 10)
        age = st.slider("Age", 18, 70, 30)
        annual_income = st.slider("Annual Income ($)", 20000, 200000, 60000)
        num_credit_cards = st.slider("Number of Credit Cards", 0, 10, 2)
        payment_history_days_late = st.slider("Days Late (worst)", 0, 90, 0)
        spend_score = st.slider("Spend Score", 0.0, 1.0, 0.5)

        calculate = st.button("Calculate Risk", type="primary", use_container_width=True)

    with col_right:
        st.subheader("Risk Assessment")

        if calculate and model_loaded:
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

            # Risk Gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                number={'suffix': "%", 'font': {'size': 48}},
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
            fig.update_layout(height=250, margin=dict(t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)

            # Result
            if risk_level == "HIGH":
                st.error(f"**HIGH RISK** - {prob:.1%} chance of default")
            else:
                st.success(f"**LOW RISK** - {prob:.1%} chance of default")

            # Quick comparison
            avg_rate = df['Default'].mean()
            if prob > avg_rate:
                st.caption(f"This is {prob/avg_rate:.1f}x higher than the average default rate ({avg_rate:.1%})")
            else:
                st.caption(f"This is {avg_rate/prob:.1f}x lower than the average default rate ({avg_rate:.1%})")

        elif not calculate:
            st.info("Adjust the sliders and click **Calculate Risk**")
        else:
            st.error("Model not loaded")

with tab2:
    st.subheader("Dataset Summary")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Customers", f"{len(df):,}")
    col2.metric("Default Rate", f"{df['Default'].mean():.1%}")
    col3.metric("Avg Income", f"${df['AnnualIncome'].mean():,.0f}")
    col4.metric("Avg Age", f"{df['Age'].mean():.0f}")

    st.markdown("---")

    # Two main charts
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Default Rate by Income Level**")
        df_temp = df.copy()
        df_temp['Income Level'] = pd.cut(df_temp['AnnualIncome'],
                                          bins=[0, 40000, 70000, 100000, 200000],
                                          labels=['<$40k', '$40-70k', '$70-100k', '>$100k'])
        rates = df_temp.groupby('Income Level')['Default'].mean() * 100
        fig = px.bar(x=rates.index, y=rates.values,
                    labels={'x': '', 'y': 'Default Rate (%)'},
                    color=rates.values,
                    color_continuous_scale=['#2ecc71', '#f1c40f', '#e74c3c'])
        fig.update_layout(showlegend=False, coloraxis_showscale=False, height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Feature Importance**")
        feature_names = ["TransactionAmount", "TransactionCount", "Age",
                        "AnnualIncome", "NumCreditCards", "PaymentHistoryDaysLate", "SpendScore"]
        importance = model.feature_importances_

        imp_df = pd.DataFrame({
            'Feature': [f.replace('PaymentHistoryDaysLate', 'DaysLate') for f in feature_names],
            'Importance': importance
        }).sort_values('Importance', ascending=True)

        fig = px.bar(imp_df, x='Importance', y='Feature', orientation='h',
                    color='Importance', color_continuous_scale='Blues')
        fig.update_layout(showlegend=False, coloraxis_showscale=False, height=300)
        st.plotly_chart(fig, use_container_width=True)

    # Data preview
    st.markdown("**Sample Data** (500 BNPL customers)")
    st.dataframe(df.head(10), use_container_width=True, hide_index=True)

# Footer
st.markdown("---")
st.caption("Built with XGBoost, Streamlit & AWS Lambda | [GitHub](https://github.com/AnthonyB-316/bnpl-credit-risk-platform)")
