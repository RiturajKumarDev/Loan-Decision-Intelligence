import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib

app = Flask(__name__)
CORS(app)

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model files safely
model = joblib.load(os.path.join(BASE_DIR, "loan_default_model.pkl"))
columns = joblib.load(os.path.join(BASE_DIR, "model_columns.pkl"))


@app.route("/")
def home():
    return jsonify({"message": "Loan Default Prediction API is running"})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No input data provided"}), 400

        input_df = pd.DataFrame([data])

        # One-hot encoding
        input_df = pd.get_dummies(input_df)

        # Match training columns
        input_df = input_df.reindex(columns=columns, fill_value=0)

        # Prediction
        pred = model.predict(input_df)[0]

        return jsonify({"prediction": int(pred)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
