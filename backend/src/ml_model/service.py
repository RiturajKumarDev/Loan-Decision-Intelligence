import os
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = None
columns = None


def load_models():
    global model, columns
    if model is None:
        model_path = os.path.join(BASE_DIR, "models", "loan_default_model.pkl")
        columns_path = os.path.join(BASE_DIR, "models", "model_columns.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        if not os.path.exists(columns_path):
            raise FileNotFoundError(f"Columns file not found at {columns_path}")
        model = joblib.load(model_path)
        columns = joblib.load(columns_path)
    return model, columns
