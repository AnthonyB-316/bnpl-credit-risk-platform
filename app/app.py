import os
import boto3
import pickle
import pandas as pd
from flask import Flask, request, jsonify
from mangum import Mangum

app = Flask(__name__)

# Fixed bucket name — this is the one CloudFormation creates
BUCKET = "bnpl-credit-risk-model-bucket-anthonyb"
MODEL_KEY = "model.pkl"

s3 = boto3.client('s3')

def load_model():
    print(f"Downloading model from s3://{BUCKET}/{MODEL_KEY}")
    obj = s3.get_object(Bucket=BUCKET, Key=MODEL_KEY)
    return pickle.loads(obj['Body'].read())

# Load once at cold start
try:
    model = load_model()
    print("Model loaded successfully")
except Exception as e:
    print(f"Model load failed: {e}")
    model = None

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.get_json(force=True)
        df = pd.DataFrame([data])
        prob = float(model.predict_proba(df)[0][1])
        return jsonify({
            "default_probability": round(prob, 4),
            "risk_level": "HIGH" if prob > 0.5 else "LOW"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

handler = Mangum(app)