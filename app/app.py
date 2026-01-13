import os
import pickle
import pandas as pd
from flask import Flask, request, jsonify
from mangum import Mangum

app = Flask(__name__)

# S3 configuration from environment variable
BUCKET = os.environ.get("S3_BUCKET", "bnpl-credit-risk-model-bucket-anthonyb")
MODEL_KEY = "model.pkl"
LOCAL_MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")


def load_model():
    """Load model from local file first, fall back to S3 for Lambda."""
    # Try local file first (for local development)
    if os.path.exists(LOCAL_MODEL_PATH):
        print(f"Loading model from local file: {LOCAL_MODEL_PATH}")
        with open(LOCAL_MODEL_PATH, "rb") as f:
            return pickle.load(f)

    # Fall back to S3 (for AWS Lambda)
    print(f"Downloading model from s3://{BUCKET}/{MODEL_KEY}")
    import boto3
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=BUCKET, Key=MODEL_KEY)
    return pickle.loads(obj["Body"].read())


# Load once at cold start
try:
    model = load_model()
    print("Model loaded successfully")
except Exception as e:
    print(f"Model load failed: {e}")
    model = None


@app.route("/predict", methods=["POST"])
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


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "model_loaded": model is not None})


handler = Mangum(app)
