"""
=============================================================================
  Multi-Disease Prediction System — Flask Backend API
  Trained Models Integrated:
    1. Heart Disease       -> Logistic Regression (with StandardScaler)
    2. Type-2 Diabetes     -> Gradient Boosting Classifier
    3. Parkinson's Disease -> Decision Tree Classifier
=============================================================================
"""

import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

# Initialize Flask Application
app = Flask(__name__)
CORS(app)  # Allows your frontend (HTML/React/Flutter) to communicate with this backend

# Define path to the saved models directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")

print("--- Loading Trained Machine Learning Models ---")
# 1. Heart Disease Model & Scaler
heart_model = joblib.load(os.path.join(MODEL_DIR, "heart_logistic_regression.pkl"))
heart_scaler = joblib.load(os.path.join(MODEL_DIR, "heart_scaler.pkl"))

# 2. Type-2 Diabetes Model
diab_model = joblib.load(os.path.join(MODEL_DIR, "diabetes_gradient_boosting.pkl"))

# 3. Parkinson's Disease Model
park_model = joblib.load(os.path.join(MODEL_DIR, "parkinsons_decision_tree.pkl"))

print("[OK] All 3 Machine Learning models and scalers loaded successfully!\n")

# Define the exact feature names expected by each model
HEART_FEATURES = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]

DIABETES_FEATURES = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
]

PARKINSONS_FEATURES = [
    'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 
    'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 
    'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 
    'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 
    'spread1', 'spread2', 'D2', 'PPE'
]


# =============================================================================
# Helper function: extracts feature values from request JSON
# Supports both:
#   A) {"features": [val1, val2, ...]}
#   B) {"age": 55, "sex": 1, ...} (named dictionary keys)
# =============================================================================
def extract_input_data(req_data, expected_feature_list):
    if not req_data:
        raise ValueError("No JSON data provided in request body.")

    # Format A: user passed a list inside "features"
    if "features" in req_data and isinstance(req_data["features"], list):
        if len(req_data["features"]) != len(expected_feature_list):
            raise ValueError(
                f"Expected {len(expected_feature_list)} features, but received {len(req_data['features'])}."
            )
        values = [float(x) for x in req_data["features"]]
        return pd.DataFrame([values], columns=expected_feature_list)

    # Format B: user passed key-value pairs
    missing_keys = [col for col in expected_feature_list if col not in req_data]
    if missing_keys:
        raise ValueError(f"Missing required fields: {missing_keys}")

    values = [float(req_data[col]) for col in expected_feature_list]
    return pd.DataFrame([values], columns=expected_feature_list)


# =============================================================================
# 0. Health Check / Welcome Route
# =============================================================================
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "status": "online",
        "system": "Multi-Disease Clinical Decision Support Backend",
        "available_endpoints": [
            {"method": "POST", "endpoint": "/predict/heart", "features_required": 13},
            {"method": "POST", "endpoint": "/predict/diabetes", "features_required": 8},
            {"method": "POST", "endpoint": "/predict/parkinsons", "features_required": 22}
        ]
    })


# =============================================================================
# 1. 🫀 Heart Disease Prediction Endpoint
# =============================================================================
@app.route('/predict/heart', methods=['POST'])
def predict_heart():
    try:
        req_data = request.get_json(force=True)
        input_df = extract_input_data(req_data, HEART_FEATURES)

        # Heart Disease requires feature standardization using saved scaler
        scaled_input = heart_scaler.transform(input_df)

        prediction = int(heart_model.predict(scaled_input)[0])
        probabilities = heart_model.predict_proba(scaled_input)[0]
        prob_disease = round(float(probabilities[1]) * 100, 2)

        return jsonify({
            "success": True,
            "disease": "Coronary Heart Disease",
            "prediction": prediction,
            "risk_status": "High Risk Detected" if prediction == 1 else "Normal / Low Risk",
            "probability_percentage": prob_disease,
            "clinical_notes": (
                "Cardiovascular risk detected. Immediate follow-up with ECG/angiography recommended."
                if prediction == 1 else
                "Cardiovascular biomarkers within normal physiological thresholds."
            )
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


# =============================================================================
# 2. 🩸 Type-2 Diabetes Prediction Endpoint
# =============================================================================
@app.route('/predict/diabetes', methods=['POST'])
def predict_diabetes():
    try:
        req_data = request.get_json(force=True)
        input_df = extract_input_data(req_data, DIABETES_FEATURES)

        # Gradient Boosting operates on raw features (no scaling required)
        prediction = int(diab_model.predict(input_df)[0])
        probabilities = diab_model.predict_proba(input_df)[0]
        prob_diabetic = round(float(probabilities[1]) * 100, 2)

        return jsonify({
            "success": True,
            "disease": "Type-2 Diabetes Mellitus",
            "prediction": prediction,
            "risk_status": "High Risk Detected" if prediction == 1 else "Normal / Low Risk",
            "probability_percentage": prob_diabetic,
            "clinical_notes": (
                "Metabolic anomaly detected. HbA1c test and dietary glycemic control advised."
                if prediction == 1 else
                "Metabolic parameters nominal."
            )
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


# =============================================================================
# 3. 🧠 Parkinson's Disease Prediction Endpoint
# =============================================================================
@app.route('/predict/parkinsons', methods=['POST'])
def predict_parkinsons():
    try:
        req_data = request.get_json(force=True)
        input_df = extract_input_data(req_data, PARKINSONS_FEATURES)

        # Decision Tree operates on raw acoustic features
        prediction = int(park_model.predict(input_df)[0])
        probabilities = park_model.predict_proba(input_df)[0]
        prob_parkinsons = round(float(probabilities[1]) * 100, 2)

        return jsonify({
            "success": True,
            "disease": "Parkinson's Disease (Acoustic Biomarkers)",
            "prediction": prediction,
            "risk_status": "Parkinson's Disease Detected" if prediction == 1 else "Healthy Voice Biomarkers",
            "probability_percentage": prob_parkinsons,
            "clinical_notes": (
                "Acoustic perturbation patterns indicate vocal impairment consistent with Parkinson's."
                if prediction == 1 else
                "Vocal frequency and tremor measures within normal boundaries."
            )
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


# =============================================================================
# Run the Flask Server
# =============================================================================
if __name__ == '__main__':
    # Runs the local development server on port 5000
    print("\n🚀 Starting Flask Backend on http://127.0.0.1:5000 ...")
    app.run(debug=True, host='0.0.0.0', port=5000)
