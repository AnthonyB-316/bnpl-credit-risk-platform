# BNPL Credit Risk Scoring

Credit scoring model for Buy Now Pay Later applications. Predicts default risk based on income, credit history, and transaction patterns.

## Run it

```bash
pip install -r requirements.txt
streamlit run app/dashboard.py
```

The dashboard lets you input applicant data and see the risk score + feature breakdown.

## How it works

XGBoost classifier trained on 500 synthetic BNPL applications. Features include debt-to-income ratio, payment history, account age, etc.

## AWS deployment

There's an AWS SAM template in `infrastructure/` if you want to deploy it as a Lambda function. Just run `./deploy.sh` with your AWS credentials configured.

## Stack

Python, XGBoost, Streamlit, AWS SAM
