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
        st.caption("Use sliders or type exact values - they stay in sync!")

        def sync_slider(key):
            st.session_state[key] = st.session_state[f"{key}_slider"]

        def sync_input(key):
            st.session_state[key] = st.session_state[f"{key}_input"]

        # Initialize defaults
        defaults = {
            'avg_purchase': 1500, 'num_purchases': 10, 'age': 30,
            'annual_income': 60000, 'num_credit_cards': 2, 'days_late': 0, 'debt_to_income': 0.3
        }
        for key, val in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = val

        # Average Purchase Amount
        col_a, col_b = st.columns([3, 1])
        with col_a:
            st.slider("Average Purchase Amount ($)", 100, 5000,
                      value=st.session_state.avg_purchase,
                      key="avg_purchase_slider", on_change=sync_slider, args=("avg_purchase",),
                      help="Typical dollar amount per BNPL purchase")
        with col_b:
            st.number_input("$", 100, 5000,
                           value=st.session_state.avg_purchase,
                           key="avg_purchase_input", on_change=sync_input, args=("avg_purchase",),
                           label_visibility="collapsed")

        # Number of Purchases
        col_a, col_b = st.columns([3, 1])
        with col_a:
            st.slider("BNPL Purchases (6 months)", 1, 50,
                      value=st.session_state.num_purchases,
                      key="num_purchases_slider", on_change=sync_slider, args=("num_purchases",),
                      help="How many BNPL transactions in the last 6 months")
        with col_b:
            st.number_input("#", 1, 50,
                           value=st.session_state.num_purchases,
                           key="num_purchases_input", on_change=sync_input, args=("num_purchases",),
                           label_visibility="collapsed")

        # Age
        col_a, col_b = st.columns([3, 1])
        with col_a:
            st.slider("Age", 18, 70,
                      value=st.session_state.age,
                      key="age_slider", on_change=sync_slider, args=("age",))
        with col_b:
            st.number_input("yrs", 18, 70,
                           value=st.session_state.age,
                           key="age_input", on_change=sync_input, args=("age",),
                           label_visibility="collapsed")

        # Annual Income
        col_a, col_b = st.columns([3, 1])
        with col_a:
            st.slider("Annual Income ($)", 20000, 200000, step=1000,
                      value=st.session_state.annual_income,
                      key="annual_income_slider", on_change=sync_slider, args=("annual_income",),
                      help="Yearly income before taxes")
        with col_b:
            st.number_input("$", 20000, 200000, step=1000,
                           value=st.session_state.annual_income,
                           key="annual_income_input", on_change=sync_input, args=("annual_income",),
                           label_visibility="collapsed")

        # Credit Cards
        col_a, col_b = st.columns([3, 1])
        with col_a:
            st.slider("Number of Credit Cards", 0, 10,
                      value=st.session_state.num_credit_cards,
                      key="num_credit_cards_slider", on_change=sync_slider, args=("num_credit_cards",))
        with col_b:
            st.number_input("#", 0, 10,
                           value=st.session_state.num_credit_cards,
                           key="num_credit_cards_input", on_change=sync_input, args=("num_credit_cards",),
                           label_visibility="collapsed")

        # Days Late
        col_a, col_b = st.columns([3, 1])
        with col_a:
            st.slider("Worst Payment Delay (days)", 0, 90,
                      value=st.session_state.days_late,
                      key="days_late_slider", on_change=sync_slider, args=("days_late",),
                      help="Longest time past due date on any payment")
        with col_b:
            st.number_input("days", 0, 90,
                           value=st.session_state.days_late,
                           key="days_late_input", on_change=sync_input, args=("days_late",),
                           label_visibility="collapsed")

        # Debt-to-Income
        col_a, col_b = st.columns([3, 1])
        with col_a:
            st.slider("Debt-to-Income Ratio", 0.0, 1.0, step=0.05,
                      value=st.session_state.debt_to_income,
                      key="debt_to_income_slider", on_change=sync_slider, args=("debt_to_income",),
                      help="Monthly debt payments divided by monthly income (0.3 = 30%)")
        with col_b:
            st.number_input("ratio", 0.0, 1.0, step=0.05,
                           value=st.session_state.debt_to_income,
                           key="debt_to_income_input", on_change=sync_input, args=("debt_to_income",),
                           label_visibility="collapsed")

        # Get values from session state
        avg_purchase = st.session_state.avg_purchase
        num_purchases = st.session_state.num_purchases
        age = st.session_state.age
        annual_income = st.session_state.annual_income
        num_credit_cards = st.session_state.num_credit_cards
        days_late = st.session_state.days_late
        debt_to_income = st.session_state.debt_to_income

        calculate = st.button("Calculate Risk", type="primary", use_container_width=True)

    with col_right:
        st.subheader("Risk Assessment")

        if calculate and model_loaded:
            # Map to model features (model was trained with original names)
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
    st.caption("Demo dataset of 500 simulated BNPL customers")

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

    # Full scrollable data table with better column names
    st.markdown("---")
    st.markdown(f"**Full Dataset** ({len(df)} customers)")

    # Rename columns for display
    df_display = df.copy()
    df_display.columns = [
        'Avg Purchase ($)',
        'Num Purchases (6mo)',
        'Age',
        'Annual Income ($)',
        'Credit Cards',
        'Days Late',
        'Debt-to-Income',
        'Defaulted'
    ]
    # Convert Default to Yes/No
    df_display['Defaulted'] = df_display['Defaulted'].map({0: 'No', 1: 'Yes'})

    st.dataframe(df_display, use_container_width=True, hide_index=True, height=400)

    # Legend
    st.caption("**Defaulted**: Whether the customer failed to repay (Yes = defaulted, No = paid back)")

with tab3:
    st.subheader("Upload Your Own Data")
    st.markdown("Upload a CSV file to score multiple customers at once.")

    # Show required format
    with st.expander("Required CSV Format"):
        st.markdown("""
        Your CSV must have these columns (use exact names):

        | Column | Description | Example |
        |--------|-------------|---------|
        | `TransactionAmount` | Average purchase amount ($) | 1500 |
        | `TransactionCount` | Number of BNPL purchases (6 months) | 10 |
        | `Age` | Customer age | 30 |
        | `AnnualIncome` | Yearly income ($) | 60000 |
        | `NumCreditCards` | Number of credit cards | 2 |
        | `PaymentHistoryDaysLate` | Worst payment delay (days) | 0 |
        | `SpendScore` | Debt-to-income ratio (0-1) | 0.3 |
        """)

        # Sample download
        sample = pd.DataFrame([
            {"TransactionAmount": 1500, "TransactionCount": 10, "Age": 30,
             "AnnualIncome": 60000, "NumCreditCards": 2, "PaymentHistoryDaysLate": 0, "SpendScore": 0.3},
            {"TransactionAmount": 3000, "TransactionCount": 25, "Age": 22,
             "AnnualIncome": 35000, "NumCreditCards": 4, "PaymentHistoryDaysLate": 15, "SpendScore": 0.6},
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
