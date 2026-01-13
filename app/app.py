import os
import pickle
from flask import Flask, request, jsonify

app = Flask(__name__)

# S3 configuration from environment variable
BUCKET = os.environ.get("S3_BUCKET", "bnpl-credit-risk-model-bucket-anthonyb")
MODEL_KEY = "model.pkl"
LOCAL_MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

# Global model cache (loaded lazily)
_model = None


def get_model():
    """Load model lazily on first request."""
    global _model
    if _model is not None:
        return _model

    # Try local file first (for local development)
    if os.path.exists(LOCAL_MODEL_PATH):
        print(f"Loading model from local file: {LOCAL_MODEL_PATH}")
        with open(LOCAL_MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
            return _model

    # Fall back to S3 (for AWS Lambda)
    print(f"Downloading model from s3://{BUCKET}/{MODEL_KEY}")
    import boto3
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=BUCKET, Key=MODEL_KEY)
    _model = pickle.loads(obj["Body"].read())
    print("Model loaded successfully")
    return _model


@app.route("/predict", methods=["POST"])
def predict():
    try:
        import pandas as pd
        model = get_model()
        data = request.get_json(force=True)
        df = pd.DataFrame([data])
        prob = float(model.predict_proba(df)[0][1])
        return jsonify({
            "default_probability": round(prob, 4),
            "risk_level": "HIGH" if prob > 0.5 else "LOW"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "model_loaded": _model is not None})


def handler(event, context):
    """AWS Lambda handler using awsgi."""
    import awsgi
    return awsgi.response(app, event, context)
