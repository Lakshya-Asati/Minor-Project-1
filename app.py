"""
=============================================================================
  Multi-Disease Clinical Decision Support Suite — Production Backend API
  Academic Minor Project (Part-I) | Department of CSE, JUET Guna
  Supervision: Dr. Rahul Pachauri
  Team: Vijay Kumar Rajak (241B286), Piyush Vishwakarma (241B177), Lakshya Asati (241B134)
=============================================================================
  Integrated Clinical Disease Domains:
    1. Coronary Heart Disease (Cleveland UCI Repository, 1,025 Records)
    2. Type-2 Diabetes Mellitus (Pima Indians NIDDK Cohort, 768 Records)
    3. Parkinson's Disease (Oxford Vocal Acoustic Biomarkers, 195 Records)
=============================================================================
"""

import os
import io
import csv
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# Initialize Flask Application
app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")

print("--- [INITIALIZING] Loading Trained Clinical Models & Scalers ---")

# =============================================================================
# 1. MODEL REPOSITORIES (17 Classifiers + 3 StandardScalers)
# =============================================================================
# Scalers
heart_scaler = joblib.load(os.path.join(MODEL_DIR, "heart_scaler.pkl"))
diabetes_scaler = joblib.load(os.path.join(MODEL_DIR, "diabetes_scaler.pkl"))
parkinsons_scaler = joblib.load(os.path.join(MODEL_DIR, "parkinsons_scaler.pkl"))

# Heart Disease Models: {key: (model_instance, requires_scaler, display_name, accuracy, recall, f1, auc)}
HEART_MODELS = {
    'logistic_regression': (
        joblib.load(os.path.join(MODEL_DIR, "heart_logistic_regression.pkl")),
        True, "Logistic Regression (Primary Linear Baseline)", 92.25, 94.50, 0.9242, 0.9638
    ),
    'gradient_boosting': (
        joblib.load(os.path.join(MODEL_DIR, "heart_gradient_boosting.pkl")),
        False, "Gradient Boosting Classifier", 91.75, 92.00, 0.9177, 0.9710
    ),
    'random_forest': (
        joblib.load(os.path.join(MODEL_DIR, "heart_random_forest.pkl")),
        False, "Random Forest Ensemble (100 Trees)", 92.00, 91.00, 0.9192, 0.9719
    ),
    'rbf_svm': (
        joblib.load(os.path.join(MODEL_DIR, "heart_rbf_svm.pkl")),
        True, "Support Vector Machine (RBF Kernel)", 91.25, 93.00, 0.9140, 0.9688
    ),
    'knn': (
        joblib.load(os.path.join(MODEL_DIR, "heart_knn.pkl")),
        True, "K-Nearest Neighbors (k=7)", 89.75, 95.50, 0.9031, 0.9558
    ),
    'decision_tree': (
        joblib.load(os.path.join(MODEL_DIR, "heart_decision_tree.pkl")),
        False, "Decision Tree (CART, max_depth=5)", 85.75, 84.00, 0.8550, 0.9371
    )
}

# Type-2 Diabetes Models
DIABETES_MODELS = {
    'gradient_boosting': (
        joblib.load(os.path.join(MODEL_DIR, "diabetes_gradient_boosting.pkl")),
        False, "Gradient Boosting Classifier (Optimal)", 85.25, 88.00, 0.8564, 0.9361
    ),
    'random_forest': (
        joblib.load(os.path.join(MODEL_DIR, "diabetes_random_forest.pkl")),
        False, "Random Forest Ensemble (100 Trees)", 84.00, 86.00, 0.8431, 0.9156
    ),
    'decision_tree': (
        joblib.load(os.path.join(MODEL_DIR, "diabetes_decision_tree.pkl")),
        False, "Decision Tree (CART, max_depth=4)", 82.25, 89.00, 0.8337, 0.8739
    ),
    'logistic_regression': (
        joblib.load(os.path.join(MODEL_DIR, "diabetes_logistic_regression.pkl")),
        True, "Logistic Regression", 80.25, 79.50, 0.8010, 0.8842
    ),
    'rbf_svm': (
        joblib.load(os.path.join(MODEL_DIR, "diabetes_rbf_svm.pkl")),
        True, "Support Vector Machine (RBF Kernel)", 79.50, 82.00, 0.8000, 0.8917
    ),
    'knn': (
        joblib.load(os.path.join(MODEL_DIR, "diabetes_knn.pkl")),
        True, "K-Nearest Neighbors (k=7)", 78.50, 77.50, 0.7828, 0.8583
    )
}

# Parkinson's Disease Models
PARKINSONS_MODELS = {
    'decision_tree': (
        joblib.load(os.path.join(MODEL_DIR, "parkinsons_decision_tree.pkl")),
        False, "Decision Tree (White-Box Rule Interpretability)", 93.75, 94.17, 0.9378, 0.9457
    ),
    'random_forest': (
        joblib.load(os.path.join(MODEL_DIR, "parkinsons_random_forest.pkl")),
        False, "Random Forest Ensemble (Optimal F1)", 95.42, 94.17, 0.9536, 0.9926
    ),
    'rbf_svm': (
        joblib.load(os.path.join(MODEL_DIR, "parkinsons_rbf_svm.pkl")),
        True, "Support Vector Machine (RBF Kernel)", 92.92, 88.33, 0.9258, 0.9849
    ),
    'knn': (
        joblib.load(os.path.join(MODEL_DIR, "parkinsons_knn.pkl")),
        True, "K-Nearest Neighbors (k=5)", 91.25, 93.33, 0.9143, 0.9741
    ),
    'linear_svm': (
        joblib.load(os.path.join(MODEL_DIR, "parkinsons_linear_svm.pkl")),
        True, "Linear Support Vector Machine", 86.25, 82.50, 0.8571, 0.9283
    )
}

print(f"[OK] Successfully loaded {len(HEART_MODELS) + len(DIABETES_MODELS) + len(PARKINSONS_MODELS)} classifiers and 3 scalers.\n")

# Feature Schemas
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
# 2. CLINICAL BIOMARKER RISK ENGINE
# =============================================================================
def analyze_heart_biomarkers(row):
    """Evaluates cardiovascular biomarkers against ACC/AHA clinical guidelines."""
    findings = []

    # Blood Pressure (AHA Guideline)
    bp = row['trestbps']
    if bp >= 140:
        findings.append({"marker": "Resting Blood Pressure", "value": f"{bp} mmHg", "status": "Stage 2 Hypertension", "severity": "danger", "note": "Elevated vascular resistance"})
    elif bp >= 130:
        findings.append({"marker": "Resting Blood Pressure", "value": f"{bp} mmHg", "status": "Stage 1 Hypertension", "severity": "warning", "note": "Borderline elevation"})
    else:
        findings.append({"marker": "Resting Blood Pressure", "value": f"{bp} mmHg", "status": "Normotensive", "severity": "normal", "note": "Within healthy systolic range"})

    # Serum Cholesterol
    chol = row['chol']
    if chol >= 240:
        findings.append({"marker": "Total Serum Cholesterol", "value": f"{chol} mg/dL", "status": "Hypercholesterolemia", "severity": "danger", "note": "Atherosclerotic plaque risk"})
    elif chol >= 200:
        findings.append({"marker": "Total Serum Cholesterol", "value": f"{chol} mg/dL", "status": "Borderline High", "severity": "warning", "note": "Dietary modification recommended"})
    else:
        findings.append({"marker": "Total Serum Cholesterol", "value": f"{chol} mg/dL", "status": "Desirable", "severity": "normal", "note": "Low atherogenic profile"})

    # Max Heart Rate vs Expected Age Max
    expected_max = 220 - row['age']
    achieved = row['thalach']
    if achieved < (expected_max * 0.70):
        findings.append({"marker": "Max Heart Rate Achieved", "value": f"{achieved} bpm (Exp: ~{expected_max})", "status": "Blunted Chronotropic Response", "severity": "warning", "note": "Impaired exercise tolerance"})
    else:
        findings.append({"marker": "Max Heart Rate Achieved", "value": f"{achieved} bpm", "status": "Adequate Inotropic Reserve", "severity": "normal", "note": "Normal physiological heart rate reserve"})

    # ST Depression (oldpeak)
    st = row['oldpeak']
    if st >= 2.0:
        findings.append({"marker": "ST Depression (oldpeak)", "value": f"{st} mm", "status": "Severe Subendocardial Ischemia", "severity": "danger", "note": "Strong electrocardiographic indicator of ischemia"})
    elif st >= 1.0:
        findings.append({"marker": "ST Depression (oldpeak)", "value": f"{st} mm", "status": "Equivocal / Mild Ischemia", "severity": "warning", "note": "Stress-induced repolarization anomaly"})
    else:
        findings.append({"marker": "ST Depression (oldpeak)", "value": f"{st} mm", "status": "Nominal Baseline", "severity": "normal", "note": "No ischemic ST depression"})

    # Fluoroscopy Major Vessels (ca)
    ca = int(row['ca'])
    if ca > 0:
        findings.append({"marker": "Coronary Vessels Fluoroscopy", "value": f"{ca} vessels blocked", "status": "Coronary Stenosis Evident", "severity": "danger", "note": "Confirmed vascular calcification"})
    else:
        findings.append({"marker": "Coronary Vessels Fluoroscopy", "value": "0 vessels", "status": "Patent Arteries", "severity": "normal", "note": "No major luminal narrowing detected"})

    # Chest Pain Type (cp)
    cp_map = {0: "Typical Angina", 1: "Atypical Angina", 2: "Non-anginal Pain", 3: "Asymptomatic"}
    cp_val = int(row['cp'])
    findings.append({"marker": "Chest Pain Classification", "value": cp_map.get(cp_val, str(cp_val)), "status": "Clinical Presentation", "severity": "warning" if cp_val in [0, 1] else "normal", "note": "Symptom subjective report"})

    return findings


def analyze_diabetes_biomarkers(row):
    """Evaluates metabolic biomarkers against ADA (American Diabetes Association) criteria."""
    findings = []

    # Blood Glucose
    glu = row['Glucose']
    if glu >= 140:
        findings.append({"marker": "Fasting / Plasma Glucose", "value": f"{glu} mg/dL", "status": "Diabetic Hyperglycemia", "severity": "danger", "note": "Exceeds standard diagnostic threshold (ADA >= 126 mg/dL)"})
    elif glu >= 100:
        findings.append({"marker": "Fasting / Plasma Glucose", "value": f"{glu} mg/dL", "status": "Impaired Fasting Glucose", "severity": "warning", "note": "Pre-diabetic metabolic state"})
    else:
        findings.append({"marker": "Fasting / Plasma Glucose", "value": f"{glu} mg/dL", "status": "Euglycemic", "severity": "normal", "note": "Optimal glucose homeostasis"})

    # Body Mass Index (BMI)
    bmi = row['BMI']
    if bmi >= 30.0:
        findings.append({"marker": "Body Mass Index (BMI)", "value": f"{bmi:.1f} kg/m²", "status": "Class I/II Obesity", "severity": "danger", "note": "Significant adipose tissue insulin resistance risk"})
    elif bmi >= 25.0:
        findings.append({"marker": "Body Mass Index (BMI)", "value": f"{bmi:.1f} kg/m²", "status": "Overweight", "severity": "warning", "note": "Moderate metabolic risk"})
    else:
        findings.append({"marker": "Body Mass Index (BMI)", "value": f"{bmi:.1f} kg/m²", "status": "Healthy Weight", "severity": "normal", "note": "Low metabolic risk index"})

    # Serum Insulin
    ins = row['Insulin']
    if ins >= 166:
        findings.append({"marker": "Serum Insulin Concentration", "value": f"{ins:.1f} μU/mL", "status": "Hyperinsulinemia", "severity": "warning", "note": "Indicates peripheral insulin resistance"})
    else:
        findings.append({"marker": "Serum Insulin Concentration", "value": f"{ins:.1f} μU/mL", "status": "Physiological Range", "severity": "normal", "note": "Normal baseline basal secretion"})

    # Blood Pressure (Diastolic)
    bp = row['BloodPressure']
    if bp >= 90:
        findings.append({"marker": "Diastolic Blood Pressure", "value": f"{bp} mmHg", "status": "Stage 2 Diastolic Hypertension", "severity": "danger", "note": "Elevated vascular resistance"})
    elif bp >= 80:
        findings.append({"marker": "Diastolic Blood Pressure", "value": f"{bp} mmHg", "status": "Stage 1 Diastolic Hypertension", "severity": "warning", "note": "Pre-hypertensive"})
    else:
        findings.append({"marker": "Diastolic Blood Pressure", "value": f"{bp} mmHg", "status": "Nominal Diastolic BP", "severity": "normal", "note": "Optimal range (< 80 mmHg)"})

    # Diabetes Pedigree Function
    dpf = row['DiabetesPedigreeFunction']
    if dpf >= 0.60:
        findings.append({"marker": "Diabetes Pedigree Function", "value": f"{dpf:.3f}", "status": "Elevated Genetic Risk", "severity": "warning", "note": "Familial predisposition present"})
    else:
        findings.append({"marker": "Diabetes Pedigree Function", "value": f"{dpf:.3f}", "status": "Low Familial Predisposition", "severity": "normal", "note": "Average genetic incidence"})

    return findings


def analyze_parkinsons_biomarkers(row):
    """Evaluates phonatory acoustic biomarkers against Oxford Speech Pathology thresholds."""
    findings = []

    # Frequency Perturbation (Jitter %)
    jitter = row['MDVP:Jitter(%)']
    if jitter >= 0.007:
        findings.append({"marker": "Cycle-to-Cycle Jitter (%)", "value": f"{jitter * 100:.3f}%", "status": "Elevated Micro-Tremor (Dysphonia)", "severity": "danger", "note": "Motor instability in laryngeal nerve control"})
    elif jitter >= 0.005:
        findings.append({"marker": "Cycle-to-Cycle Jitter (%)", "value": f"{jitter * 100:.3f}%", "status": "Borderline Perturbation", "severity": "warning", "note": "Mild micro-frequency variation"})
    else:
        findings.append({"marker": "Cycle-to-Cycle Jitter (%)", "value": f"{jitter * 100:.3f}%", "status": "Stable Frequency", "severity": "normal", "note": "Physiological vocal cord oscillation"})

    # Amplitude Perturbation (Shimmer)
    shim = row['MDVP:Shimmer']
    if shim >= 0.035:
        findings.append({"marker": "Vocal Shimmer (Amplitude)", "value": f"{shim:.4f}", "status": "Acoustic Glottal Incompetence", "severity": "danger", "note": "Uneven glottal airflow and volume perturbation"})
    else:
        findings.append({"marker": "Vocal Shimmer (Amplitude)", "value": f"{shim:.4f}", "status": "Uniform Amplitude", "severity": "normal", "note": "Normal sustained vowel amplitude"})

    # Harmonics-to-Noise Ratio (HNR)
    hnr = row['HNR']
    if hnr < 18.0:
        findings.append({"marker": "Harmonics-to-Noise Ratio (HNR)", "value": f"{hnr:.2f} dB", "status": "Breathiness / Acoustic Noise", "severity": "danger", "note": "Significant turbulent air noise across vocal folds"})
    elif hnr < 22.0:
        findings.append({"marker": "Harmonics-to-Noise Ratio (HNR)", "value": f"{hnr:.2f} dB", "status": "Moderate Purity", "severity": "warning", "note": "Subtle phonatory breathiness"})
    else:
        findings.append({"marker": "Harmonics-to-Noise Ratio (HNR)", "value": f"{hnr:.2f} dB", "status": "Clear Harmonic Spectrum", "severity": "normal", "note": "High vocal purity (> 20 dB)"})

    # Pitch Period Entropy (PPE)
    ppe = row['PPE']
    if ppe >= 0.22:
        findings.append({"marker": "Pitch Period Entropy (PPE)", "value": f"{ppe:.4f}", "status": "Impaired Nonlinear Pitch Dynamics", "severity": "danger", "note": "Entropy signature highly indicative of parkinsonian tremor"})
    else:
        findings.append({"marker": "Pitch Period Entropy (PPE)", "value": f"{ppe:.4f}", "status": "Nominal Entropy", "severity": "normal", "note": "Ordered pitch period distribution"})

    # Nonlinear Pitch Fundamental Spread 1 (spread1)
    spread1 = row['spread1']
    if spread1 >= -4.5:
        findings.append({"marker": "Nonlinear Fundamental Spread (spread1)", "value": f"{spread1:.3f}", "status": "Pathological Dispersion", "severity": "danger", "note": "High vocal tract irregularity (Parkinson's marker)"})
    elif spread1 <= -6.5:
        findings.append({"marker": "Nonlinear Fundamental Spread (spread1)", "value": f"{spread1:.3f}", "status": "Stable Nonlinear Dynamic", "severity": "normal", "note": "Normal voice dynamic attractor"})
    else:
        findings.append({"marker": "Nonlinear Fundamental Spread (spread1)", "value": f"{spread1:.3f}", "status": "Borderline Dispersion", "severity": "warning", "note": "Intermediate phonatory variation"})

    return findings


# =============================================================================
# 3. HELPER FUNCTIONS FOR INPUT EXTRACTION
# =============================================================================
def extract_input_data(req_data, expected_feature_list):
    if not req_data:
        raise ValueError("No input payload provided in request.")

    # Format 1: {"features": [v1, v2, ...]}
    if "features" in req_data and isinstance(req_data["features"], list):
        if len(req_data["features"]) != len(expected_feature_list):
            raise ValueError(
                f"Feature dimension mismatch: Expected {len(expected_feature_list)} features, received {len(req_data['features'])}."
            )
        values = [float(x) for x in req_data["features"]]
        return pd.DataFrame([values], columns=expected_feature_list)

    # Format 2: Named key-value pairs
    missing_keys = [col for col in expected_feature_list if col not in req_data]
    if missing_keys:
        raise ValueError(f"Missing required clinical parameters: {missing_keys}")

    values = [float(req_data[col]) for col in expected_feature_list]
    return pd.DataFrame([values], columns=expected_feature_list)


# =============================================================================
# 4. API ROUTES: HEALTH & METADATA
# =============================================================================
@app.route('/', methods=['GET'])
def index():
    accept = request.headers.get('Accept', '')
    if 'application/json' in accept and 'text/html' not in accept:
        return jsonify({
            "status": "online",
            "system": "Multi-Disease Clinical Decision Support Suite (Minor Project Part-I)",
            "version": "1.0.0-production",
            "institution": "Jaypee University of Engineering & Technology, Guna",
            "supervision": "Dr. Rahul Pachauri",
            "available_endpoints": [
                {"endpoint": "/predict/heart", "method": "POST", "features_required": 13},
                {"endpoint": "/predict/diabetes", "method": "POST", "features_required": 8},
                {"endpoint": "/predict/parkinsons", "method": "POST", "features_required": 22},
                {"endpoint": "/api/models", "method": "GET"},
                {"endpoint": "/api/benchmark", "method": "GET"},
                {"endpoint": "/api/guidelines", "method": "GET"},
                {"endpoint": "/api/batch-predict", "method": "POST"}
            ]
        })
    return render_template('index.html')


@app.route('/health', methods=['GET'])
@app.route('/api/status', methods=['GET'])
def health_check():
    return jsonify({
        "status": "online",
        "service": "Multi-Disease Clinical Decision Support Backend",
        "models_loaded": {
            "heart_disease": len(HEART_MODELS),
            "diabetes": len(DIABETES_MODELS),
            "parkinsons": len(PARKINSONS_MODELS)
        },
        "total_active_classifiers": 17,
        "scalers_active": 3,
        "environment": "Flask / Scikit-Learn 1.6 / Python 3.13"
    })


@app.route('/api/models', methods=['GET'])
def get_available_models():
    """Returns directory of all 17 trained machine learning classifiers with benchmark metrics."""
    def serialize_repo(repo):
        res = []
        for key, (_, req_sc, name, acc, rec, f1, auc) in repo.items():
            res.append({
                "model_key": key,
                "display_name": name,
                "requires_standard_scaler": req_sc,
                "holdout_accuracy_pct": acc,
                "holdout_recall_pct": rec,
                "holdout_f1_score": f1,
                "holdout_roc_auc": auc
            })
        return res

    return jsonify({
        "heart_disease": serialize_repo(HEART_MODELS),
        "diabetes": serialize_repo(DIABETES_MODELS),
        "parkinsons": serialize_repo(PARKINSONS_MODELS)
    })


@app.route('/api/benchmark', methods=['GET'])
def get_benchmark_leaderboard():
    """Returns executive benchmarking summary comparing top classifiers per domain."""
    leaderboard = [
        {
            "domain": "Coronary Heart Disease",
            "top_model": "Logistic Regression",
            "accuracy_pct": 92.25,
            "recall_pct": 94.50,
            "precision_pct": 90.43,
            "f1_score": 0.9242,
            "roc_auc": 0.9638,
            "false_negatives": 11,
            "false_positives": 20,
            "clinical_rationale": "Strongest diagnostic F1-Score (0.9242) with 94.50% clinical sensitivity, minimizing life-threatening missed cardiovascular events."
        },
        {
            "domain": "Parkinson's Disease (Acoustics)",
            "top_model": "Random Forest Ensemble / Decision Tree",
            "accuracy_pct": 95.42,
            "recall_pct": 94.17,
            "precision_pct": 96.58,
            "f1_score": 0.9536,
            "roc_auc": 0.9926,
            "false_negatives": 7,
            "false_positives": 4,
            "clinical_rationale": "Near-perfect ROC-AUC (0.9926) with 95.42% accuracy. Decision Tree alternative provides instant white-box rule transparency for speech pathologists."
        },
        {
            "domain": "Type-2 Diabetes Mellitus",
            "top_model": "Gradient Boosting Classifier",
            "accuracy_pct": 85.25,
            "recall_pct": 88.00,
            "precision_pct": 83.41,
            "f1_score": 0.8564,
            "roc_auc": 0.9361,
            "false_negatives": 24,
            "false_positives": 35,
            "clinical_rationale": "Non-linear ensemble learning yields 88.00% screening recall, effectively detecting subtle metabolic dysregulation."
        }
    ]
    return jsonify({"success": True, "leaderboard": leaderboard})


@app.route('/api/guidelines', methods=['GET'])
def get_clinical_guidelines():
    """Provides physiological reference boundaries used by the system."""
    return jsonify({
        "heart_disease": {
            "resting_bp": {"normal": "< 120 mmHg", "elevated": "120-129 mmHg", "hypertension_stage1": "130-139 mmHg", "hypertension_stage2": ">= 140 mmHg"},
            "total_cholesterol": {"desirable": "< 200 mg/dL", "borderline": "200-239 mg/dL", "high": ">= 240 mg/dL"},
            "fasting_blood_sugar": {"normal": "<= 120 mg/dL", "elevated": "> 120 mg/dL"},
            "st_depression": {"normal": "< 1.0 mm", "ischemia": ">= 1.5 mm"}
        },
        "diabetes": {
            "fasting_glucose": {"normal": "< 100 mg/dL", "prediabetes": "100-125 mg/dL", "diabetic": ">= 126 mg/dL"},
            "bmi": {"normal": "18.5 - 24.9 kg/m²", "overweight": "25.0 - 29.9 kg/m²", "obese": ">= 30.0 kg/m²"},
            "insulin": {"normal_basal": "16 - 166 μU/mL", "hyperinsulinemia": "> 166 μU/mL"}
        },
        "parkinsons": {
            "jitter_percent": {"normal": "< 0.60%", "dysphonia": ">= 0.70%"},
            "shimmer": {"normal": "< 0.03", "perturbed": ">= 0.035"},
            "hnr": {"normal_voice": "> 20 dB", "breathiness": "< 18 dB"},
            "ppe": {"nominal": "< 0.15", "tremor_entropy": ">= 0.20"}
        }
    })


# =============================================================================
# 5. CLINICAL INFERENCE ENDPOINTS
# =============================================================================
@app.route('/predict/heart', methods=['POST'])
def predict_heart():
    try:
        req_data = request.get_json(force=True)
        input_df = extract_input_data(req_data, HEART_FEATURES)

        # Select model (defaults to logistic_regression)
        model_key = req_data.get('model', 'logistic_regression').lower().replace('-', '_').replace(' ', '_')
        if model_key not in HEART_MODELS:
            model_key = 'logistic_regression'

        model, req_scaler, display_name, acc, rec, f1, auc = HEART_MODELS[model_key]

        # Apply scaling if model requires standardization
        scaled_input = heart_scaler.transform(input_df) if req_scaler else input_df

        # Raw prediction from trained model
        raw_pred = int(model.predict(scaled_input)[0])
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(scaled_input)[0]
            # In Cleveland Heart Disease dataset, class 0 corresponds to Heart Disease (severe CAD/ischemia)
            # and class 1 corresponds to Absence / Healthy control.
            prob_disease = round(float(probabilities[0]) * 100, 2)
        else:
            prob_disease = 90.0 if raw_pred == 0 else 10.0

        is_high_risk = (prob_disease >= 50.0) or (raw_pred == 0)

        # Detailed clinical biomarker analysis
        biomarker_breakdown = analyze_heart_biomarkers(input_df.iloc[0])

        return jsonify({
            "success": True,
            "disease": "Coronary Heart Disease",
            "prediction": 1 if is_high_risk else 0,
            "risk_status": "High Cardiovascular Risk Detected" if is_high_risk else "Normal / Low Cardiovascular Risk",
            "probability_percentage": prob_disease,
            "model_used": {
                "key": model_key,
                "name": display_name,
                "benchmark_accuracy": f"{acc}%",
                "benchmark_recall": f"{rec}%",
                "benchmark_f1": f1
            },
            "biomarker_findings": biomarker_breakdown,
            "clinical_notes": (
                "Cardiovascular pathology detected with high probability. Immediate comprehensive clinical evaluation, ECG, and angiographic correlation strongly advised."
                if is_high_risk else
                "Cardiovascular parameters demonstrate physiological stability with no acute signs of myocardial ischemia."
            )
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route('/predict/diabetes', methods=['POST'])
def predict_diabetes():
    try:
        req_data = request.get_json(force=True)
        input_df = extract_input_data(req_data, DIABETES_FEATURES)

        model_key = req_data.get('model', 'gradient_boosting').lower().replace('-', '_').replace(' ', '_')
        if model_key not in DIABETES_MODELS:
            model_key = 'gradient_boosting'

        model, req_scaler, display_name, acc, rec, f1, auc = DIABETES_MODELS[model_key]

        scaled_input = diabetes_scaler.transform(input_df) if req_scaler else input_df

        prediction = int(model.predict(scaled_input)[0])
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(scaled_input)[0]
            prob_diabetic = round(float(probabilities[1]) * 100, 2)
        else:
            prob_diabetic = 85.0 if prediction == 1 else 15.0

        biomarker_breakdown = analyze_diabetes_biomarkers(input_df.iloc[0])

        return jsonify({
            "success": True,
            "disease": "Type-2 Diabetes Mellitus",
            "prediction": prediction,
            "risk_status": "Diabetic Risk Detected" if prediction == 1 else "Normal / Non-Diabetic Profile",
            "probability_percentage": prob_diabetic,
            "model_used": {
                "key": model_key,
                "name": display_name,
                "benchmark_accuracy": f"{acc}%",
                "benchmark_recall": f"{rec}%",
                "benchmark_f1": f1
            },
            "biomarker_findings": biomarker_breakdown,
            "clinical_notes": (
                "Metabolic dysregulation detected. Fasting blood glucose & HbA1c diagnostic testing recommended alongside nutritional counseling."
                if prediction == 1 else
                "Metabolic parameters within acceptable clinical parameters. Continue standard lifestyle maintenance."
            )
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route('/predict/parkinsons', methods=['POST'])
def predict_parkinsons():
    try:
        req_data = request.get_json(force=True)
        input_df = extract_input_data(req_data, PARKINSONS_FEATURES)

        model_key = req_data.get('model', 'decision_tree').lower().replace('-', '_').replace(' ', '_')
        if model_key not in PARKINSONS_MODELS:
            model_key = 'decision_tree'

        model, req_scaler, display_name, acc, rec, f1, auc = PARKINSONS_MODELS[model_key]

        scaled_input = parkinsons_scaler.transform(input_df) if req_scaler else input_df

        prediction = int(model.predict(scaled_input)[0])
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(scaled_input)[0]
            prob_parkinsons = round(float(probabilities[1]) * 100, 2)
        else:
            prob_parkinsons = 92.0 if prediction == 1 else 8.0

        biomarker_breakdown = analyze_parkinsons_biomarkers(input_df.iloc[0])

        return jsonify({
            "success": True,
            "disease": "Parkinson's Disease (Acoustic Biomarkers)",
            "prediction": prediction,
            "risk_status": "Parkinsonian Dysphonia Detected" if prediction == 1 else "Healthy Phonatory Acoustic Profile",
            "probability_percentage": prob_parkinsons,
            "model_used": {
                "key": model_key,
                "name": display_name,
                "benchmark_accuracy": f"{acc}%",
                "benchmark_recall": f"{rec}%",
                "benchmark_f1": f1
            },
            "biomarker_findings": biomarker_breakdown,
            "clinical_notes": (
                "Acoustic perturbation patterns demonstrate micro-tremor and elevated entropy consistent with early Parkinson's vocal impairment. Neurological referral advised."
                if prediction == 1 else
                "Phonatory fundamental frequencies, jitter, and harmonic ratios within normal acoustic boundaries."
            )
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


# =============================================================================
# 6. BATCH PREDICTION ENDPOINT
# =============================================================================
@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    """Performs screening on a list of patient records for any target disease."""
    try:
        req_data = request.get_json(force=True)
        disease = req_data.get('disease', '').lower()
        patients = req_data.get('patients', [])

        if not patients or not isinstance(patients, list):
            return jsonify({"success": False, "error": "Expected 'patients' array in JSON body."}), 400

        if disease == 'heart':
            features = HEART_FEATURES
            model, req_scaler, name, _, _, _, _ = HEART_MODELS['logistic_regression']
            scaler = heart_scaler
        elif disease == 'diabetes':
            features = DIABETES_FEATURES
            model, req_scaler, name, _, _, _, _ = DIABETES_MODELS['gradient_boosting']
            scaler = diabetes_scaler
        elif disease in ['parkinsons', 'parkinson']:
            features = PARKINSONS_FEATURES
            model, req_scaler, name, _, _, _, _ = PARKINSONS_MODELS['decision_tree']
            scaler = parkinsons_scaler
        else:
            return jsonify({"success": False, "error": f"Unknown disease: '{disease}'. Choose 'heart', 'diabetes', or 'parkinsons'."}), 400

        results = []
        for i, pat in enumerate(patients):
            df_row = extract_input_data(pat, features)
            eval_data = scaler.transform(df_row) if req_scaler else df_row
            raw_pred = int(model.predict(eval_data)[0])
            if disease == 'heart':
                prob = round(float(model.predict_proba(eval_data)[0][0]) * 100, 2) if hasattr(model, 'predict_proba') else (90.0 if raw_pred == 0 else 10.0)
                is_high_risk = (prob >= 50.0) or (raw_pred == 0)
                pred = 1 if is_high_risk else 0
            else:
                prob = round(float(model.predict_proba(eval_data)[0][1]) * 100, 2) if hasattr(model, 'predict_proba') else (90.0 if raw_pred == 1 else 10.0)
                pred = raw_pred
            results.append({
                "patient_index": i + 1,
                "prediction": pred,
                "risk_status": "High Risk" if pred == 1 else "Normal",
                "probability_percentage": prob
            })

        return jsonify({
            "success": True,
            "disease": disease,
            "model_applied": name,
            "total_processed": len(results),
            "high_risk_count": sum(1 for r in results if r['prediction'] == 1),
            "normal_count": sum(1 for r in results if r['prediction'] == 0),
            "predictions": results
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


# =============================================================================
# 7. SERVER ENTRYPOINT
# =============================================================================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("=" * 75)
    print(" MULTI-DISEASE CLINICAL DECISION SUPPORT SYSTEM — FLASK SERVER")
    print(f" Access GUI Dashboard at: http://127.0.0.1:{port}")
    print("=" * 75)
    app.run(debug=True, host='0.0.0.0', port=port)
