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
        
        avg_purchase = st.slider("Average Purchase ($)", 100, 5000, 1500)
        num_purchases = st.slider("Purchases (6 months)", 1, 50, 10)
        age = st.slider("Age", 18, 70, 30)
        annual_income = st.slider("Annual Income ($)", 20000, 200000, 60000, step=5000)
        num_credit_cards = st.slider("Credit Cards", 0, 10, 2)
        days_late = st.slider("Worst Payment Delay (days)", 0, 90, 0)
        debt_to_income = st.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.3, step=0.05)
        
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
            else:
                st.success(f"**LOW RISK** - {prob:.1%} default probability")
        elif not model_loaded:
            st.error("Model not loaded")
        else:
            st.info("Adjust inputs and click Calculate")

with tab2:
    df = load_data()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Customers", len(df))
    col2.metric("Default Rate", f"{df['Default'].mean():.1%}")
    col3.metric("Avg Income", f"${df['AnnualIncome'].mean():,.0f}")
    
    st.dataframe(df, use_container_width=True, hide_index=True)

st.divider()
st.caption("XGBoost model trained on 500 synthetic BNPL applications")
