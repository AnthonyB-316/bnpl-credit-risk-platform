import os
import json
import pickle
import pandas as pd
from flask import Flask, request, jsonify
from mangum import Mangum

app = Flask(__name__)

# Load model at startup
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame([data])
    prob = model.predict_proba(df)[0][1]
    return jsonify({
        "default_probability": round(float(prob), 4),
        "risk_level": "HIGH" if prob > 0.5 else "LOW"
    })

handler = Mangum(app)