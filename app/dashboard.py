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

# Tabs for different views
tab1, tab2, tab3 = st.tabs(["Risk Assessment", "Data Explorer", "Model Insights"])

with tab1:
    st.subheader("Customer Risk Assessment")
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

            col1, col2, col3 = st.columns([1, 2, 1])

            with col2:
                # Risk Gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Default Risk Score", 'font': {'size': 24}},
                    number={'suffix': "%", 'font': {'size': 40}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1},
                        'bar': {'color': "darkblue"},
                        'bgcolor': "white",
                        'steps': [
                            {'range': [0, 25], 'color': 'green'},
                            {'range': [25, 50], 'color': 'yellow'},
                            {'range': [50, 75], 'color': 'orange'},
                            {'range': [75, 100], 'color': 'red'}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': prob * 100
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Default Probability", f"{prob:.2%}")
            with col2:
                if risk_level == "HIGH":
                    st.error(f"Risk Level: {risk_level}")
                else:
                    st.success(f"Risk Level: {risk_level}")

            # Customer comparison
            st.markdown("### How does this compare?")
            col1, col2 = st.columns(2)

            with col1:
                avg_default_rate = df['Default'].mean() * 100
                st.metric("Dataset Average Default Rate", f"{avg_default_rate:.1f}%",
                         delta=f"{(prob*100 - avg_default_rate):.1f}% vs avg",
                         delta_color="inverse")

            with col2:
                percentile = (df['AnnualIncome'] < annual_income).mean() * 100
                st.metric("Income Percentile", f"{percentile:.0f}%",
                         help="Percentage of customers with lower income")

        else:
            st.error("Model not loaded. Cannot make predictions.")

with tab2:
    st.subheader("Dataset Overview")

    # Summary stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Customers", len(df))
    with col2:
        st.metric("Default Rate", f"{df['Default'].mean():.1%}")
    with col3:
        st.metric("Avg Transaction", f"${df['TransactionAmount'].mean():,.0f}")
    with col4:
        st.metric("Avg Income", f"${df['AnnualIncome'].mean():,.0f}")

    st.markdown("---")

    # Distribution charts
    st.subheader("Feature Distributions")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(df, x="AnnualIncome", color="Default",
                          title="Income Distribution by Default Status",
                          labels={"Default": "Defaulted"},
                          color_discrete_map={0: "green", 1: "red"})
        st.plotly_chart(fig, use_container_width=True)

        fig = px.histogram(df, x="Age", color="Default",
                          title="Age Distribution by Default Status",
                          color_discrete_map={0: "green", 1: "red"})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(df, x="PaymentHistoryDaysLate", color="Default",
                          title="Payment History by Default Status",
                          color_discrete_map={0: "green", 1: "red"})
        st.plotly_chart(fig, use_container_width=True)

        fig = px.box(df, x="Default", y="TransactionAmount",
                    title="Transaction Amount by Default Status",
                    color="Default",
                    color_discrete_map={0: "green", 1: "red"})
        st.plotly_chart(fig, use_container_width=True)

    # Scatter plot
    st.subheader("Income vs Transaction Amount")
    fig = px.scatter(df, x="AnnualIncome", y="TransactionAmount",
                    color="Default", size="SpendScore",
                    title="Customer Segments",
                    color_discrete_map={0: "green", 1: "red"},
                    hover_data=["Age", "NumCreditCards"])
    st.plotly_chart(fig, use_container_width=True)

    # Raw data
    st.subheader("Raw Data")
    st.dataframe(df, use_container_width=True)

with tab3:
    st.subheader("Model Insights")

    if model_loaded:
        # Feature importance
        feature_names = ["TransactionAmount", "TransactionCount", "Age",
                        "AnnualIncome", "NumCreditCards", "PaymentHistoryDaysLate", "SpendScore"]
        importance = model.feature_importances_

        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=True)

        fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                    title="Feature Importance (XGBoost)",
                    color='Importance', color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)

        # Correlation matrix
        st.subheader("Feature Correlations")
        corr = df.corr()
        fig = px.imshow(corr, text_auto='.2f', aspect="auto",
                       title="Correlation Heatmap",
                       color_continuous_scale='RdBu_r')
        st.plotly_chart(fig, use_container_width=True)

        # Default rate by feature bins
        st.subheader("Default Rate Analysis")

        col1, col2 = st.columns(2)

        with col1:
            df['Income_Bin'] = pd.cut(df['AnnualIncome'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
            default_by_income = df.groupby('Income_Bin')['Default'].mean() * 100
            fig = px.bar(x=default_by_income.index, y=default_by_income.values,
                        title="Default Rate by Income Level",
                        labels={'x': 'Income Level', 'y': 'Default Rate (%)'})
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            df['Age_Bin'] = pd.cut(df['Age'], bins=[18, 25, 35, 45, 55, 70], labels=['18-25', '26-35', '36-45', '46-55', '56-70'])
            default_by_age = df.groupby('Age_Bin')['Default'].mean() * 100
            fig = px.bar(x=default_by_age.index, y=default_by_age.values,
                        title="Default Rate by Age Group",
                        labels={'x': 'Age Group', 'y': 'Default Rate (%)'})
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Model not loaded. Train the model first to see insights.")
