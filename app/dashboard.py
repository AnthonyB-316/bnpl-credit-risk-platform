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
def load_default_data():
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "bnpl_sample_500.csv")
    return pd.read_csv(data_path)

try:
    model = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False
    st.error("Model not found. Please run `python train_model.py` first.")

# Tabs
tab1, tab2, tab3 = st.tabs(["Risk Calculator", "Data Overview", "Upload Your Data"])

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
            df_default = load_default_data()
            avg_rate = df_default['Default'].mean()
            if prob > avg_rate:
                st.caption(f"This is {prob/avg_rate:.1f}x higher than the average default rate ({avg_rate:.1%})")
            else:
                st.caption(f"This is {avg_rate/prob:.1f}x lower than the average default rate ({avg_rate:.1%})")

        elif not calculate:
            st.info("Adjust the sliders and click **Calculate Risk**")
        else:
            st.error("Model not loaded")

with tab2:
    df = load_default_data()

    st.subheader("Dataset Summary")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Customers", f"{len(df):,}")
    col2.metric("Default Rate", f"{df['Default'].mean():.1%}")
    col3.metric("Avg Income", f"${df['AnnualIncome'].mean():,.0f}")
    col4.metric("Avg Age", f"{df['Age'].mean():.0f}")

    st.markdown("---")

    # Two charts side by side
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
        st.markdown("**Default Rate by Age Group**")
        df_temp = df.copy()
        df_temp['Age Group'] = pd.cut(df_temp['Age'],
                                       bins=[18, 25, 35, 45, 55, 70],
                                       labels=['18-25', '26-35', '36-45', '46-55', '56-70'])
        rates = df_temp.groupby('Age Group')['Default'].mean() * 100
        fig = px.bar(x=rates.index, y=rates.values,
                    labels={'x': '', 'y': 'Default Rate (%)'},
                    color=rates.values,
                    color_continuous_scale=['#2ecc71', '#f1c40f', '#e74c3c'])
        fig.update_layout(showlegend=False, coloraxis_showscale=False, height=300)
        st.plotly_chart(fig, use_container_width=True)

    # Full scrollable data table
    st.markdown("---")
    st.markdown(f"**Full Dataset** ({len(df)} customers)")
    st.dataframe(df, use_container_width=True, hide_index=True, height=400)

with tab3:
    st.subheader("Upload Your Own Data")
    st.markdown("Upload a CSV file to score multiple customers at once.")

    # Show required format
    with st.expander("Required CSV Format"):
        st.markdown("""
        Your CSV must have these columns:
        - `TransactionAmount` - Dollar amount (e.g., 1500)
        - `TransactionCount` - Number of transactions in 6 months (e.g., 10)
        - `Age` - Customer age (e.g., 30)
        - `AnnualIncome` - Yearly income (e.g., 60000)
        - `NumCreditCards` - Number of credit cards (e.g., 2)
        - `PaymentHistoryDaysLate` - Worst payment delay in days (e.g., 0)
        - `SpendScore` - Spending behavior score 0-1 (e.g., 0.5)
        """)

        # Sample download
        sample = pd.DataFrame([
            {"TransactionAmount": 1500, "TransactionCount": 10, "Age": 30,
             "AnnualIncome": 60000, "NumCreditCards": 2, "PaymentHistoryDaysLate": 0, "SpendScore": 0.5},
            {"TransactionAmount": 3000, "TransactionCount": 25, "Age": 22,
             "AnnualIncome": 35000, "NumCreditCards": 4, "PaymentHistoryDaysLate": 15, "SpendScore": 0.8},
        ])
        st.download_button(
            "Download Sample CSV",
            sample.to_csv(index=False),
            "sample_customers.csv",
            "text/csv"
        )

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            user_df = pd.read_csv(uploaded_file)

            required_cols = ["TransactionAmount", "TransactionCount", "Age",
                           "AnnualIncome", "NumCreditCards", "PaymentHistoryDaysLate", "SpendScore"]

            missing = [col for col in required_cols if col not in user_df.columns]

            if missing:
                st.error(f"Missing columns: {', '.join(missing)}")
            else:
                st.success(f"Loaded {len(user_df)} customers")

                if st.button("Score All Customers", type="primary"):
                    if model_loaded:
                        # Make predictions
                        probs = model.predict_proba(user_df[required_cols])[:, 1]

                        # Add results to dataframe
                        results = user_df.copy()
                        results['Default_Probability'] = (probs * 100).round(2)
                        results['Risk_Level'] = ['HIGH' if p > 0.5 else 'LOW' for p in probs]

                        # Summary
                        st.markdown("### Results")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Total Scored", len(results))
                        col2.metric("High Risk", f"{(results['Risk_Level'] == 'HIGH').sum()}")
                        col3.metric("Avg Default Prob", f"{probs.mean():.1%}")

                        # Show results table
                        st.dataframe(results, use_container_width=True, hide_index=True, height=400)

                        # Download results
                        st.download_button(
                            "Download Results CSV",
                            results.to_csv(index=False),
                            "risk_scores.csv",
                            "text/csv"
                        )
                    else:
                        st.error("Model not loaded")

        except Exception as e:
            st.error(f"Error reading file: {e}")

# Footer
st.markdown("---")
st.caption("Built with XGBoost, Streamlit & AWS Lambda | [GitHub](https://github.com/AnthonyB-316/bnpl-credit-risk-platform)")
