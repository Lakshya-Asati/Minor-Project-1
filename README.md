# Multi-Disease Prediction System Using Machine Learning
### Minor Project (Part-I / RP-2) — Clinical Decision Support Suite
**Department of Computer Science & Engineering, Jaypee University of Engineering & Technology, Guna**

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-v1.6-orange.svg)](https://scikit-learn.org/)
[![Flask](https://img.shields.io/badge/Flask-3.1-black.svg)](https://flask.palletsprojects.com/)
[![Tests](https://img.shields.io/badge/Tests-100%25%20Passed-brightgreen.svg)]()
[![Status](https://img.shields.io/badge/Status-100%25%20Complete%20%7C%20Production%20Ready-brightgreen.svg)]()

An end-to-end Machine Learning suite for the early, non-invasive diagnostic screening of three major chronic disease domains: **Coronary Heart Disease**, **Parkinson's Disease (Acoustic Vocal Biomarkers)**, and **Type-2 Diabetes Mellitus**.

---

## 📌 Project Overview
Chronic illnesses such as cardiovascular disorders, neurodegenerative conditions, and metabolic diseases frequently advance asymptomatically, causing critical treatment delays. This project develops an interpretable, reproducible machine learning framework to assist clinicians with rapid diagnostic screening.

The system is constructed modularly across five verified stages:
1. **Clinical Data Ingestion & Quality Cleaning:** Physiological zero-anomaly auditing and class-stratified median imputation with data leakage prevention.
2. **Exploratory Data Analysis (EDA) & Visualization:** 18 publication-quality figures analyzing clinical distributions, correlation structures, and biomarker cluster boundaries.
3. **Model Architecture & Training:** 17 diverse classifiers trained across linear, decision tree, distance, and ensemble paradigms.
4. **Empirical Benchmarking & Evaluation:** Comprehensive testing on held-out patient cohorts across Accuracy, Sensitivity (Recall), Precision, F1-Score, and ROC-AUC.
5. **Interactive Clinical Web Platform & API:** Production Flask REST API paired with a responsive clinical GUI featuring dynamic classifier selection, automated biomarker risk attribution against ACC/AHA and ADA guidelines, and printable medical diagnostic cards.

---

## 🗺️ Project Directory Structure

```text
Minor-Project-1/
│
├── Dataset minor/                             <- Original raw clinical datasets
│   ├── heart.csv                              <- Cleveland Heart Disease dataset (1,025 records)
│   ├── parkinsons.data                        <- Oxford Parkinson's voice dataset (195 records)
│   └── pima-indians-diabetes.csv              <- Pima Indians Diabetes database (768 records)
│
├── processed_data/                            <- Verified, cleaned datasets & anomaly audits
│   ├── cleaned_heart_disease.csv
│   ├── cleaned_parkinsons.csv
│   ├── cleaned_diabetes.csv
│   └── diabetes_zero_audit.csv
│
├── visualizations/                            <- Rendered publication figures (300 DPI)
│   ├── heart_disease/                         (ROC curves, confusion matrices, distributions)
│   ├── parkinsons/                            (Acoustic ROC, confusion matrices, PPE vs spread1)
│   ├── diabetes/                              (Metabolic ROC, imputation validation, correlations)
│   ├── 08_multidisease_benchmark_summary.png  <- Multi-disease holdout comparison chart
│   └── 09_feature_importance_comparison.png   <- Gini importance metrics across all domains
│
├── saved_models/                              <- 20 serialized model binaries & scalers (.pkl)
│   ├── heart_*.pkl                            (6 classifiers + 1 StandardScaler)
│   ├── parkinsons_*.pkl                       (5 classifiers + 1 StandardScaler)
│   └── diabetes_*.pkl                         (6 classifiers + 1 StandardScaler)
│
├── templates/                                 <- Flask HTML templates
│   └── index.html                             <- Responsive Clinical GUI with dynamic model selector
│
├── 01_data_cleaning.ipynb                     <- [Task 1] Unified clinical data cleaning & auditing
├── 02_heart_disease_visualization.ipynb       <- [Task 2] Heart disease EDA & visualization
├── 02_parkinsons_visualization.ipynb          <- [Task 2] Parkinson's acoustic voice EDA
├── 02_diabetes_visualization.ipynb            <- [Task 2] Diabetes metabolic EDA & imputation checks
├── 03_heart_disease_training.ipynb            <- [Task 3] Heart disease model training & export
├── 03_parkinsons_training.ipynb               <- [Task 3] Parkinson's model training & export
├── 03_diabetes_training.ipynb                 <- [Task 3] Diabetes model training & export
├── 04_model_evaluation.ipynb                  <- [Task 4] Multi-metric evaluation & benchmarking
│
├── app.py                                     <- Production Flask RESTful Backend API
├── index.html                                 <- Standalone GUI dashboard (with offline fallback)
├── test_suite.py                              <- Automated test suite (100% pass rate, 7 suites)
├── test_predict.py                            <- CLI testing harness for predictions
├── run_app.bat                                <- One-click Windows application launcher
├── PROJECT_REPORT.md                          <- Exhaustive academic project report (8 Chapters)
├── final_presentation.pptx                    <- 13-slide master presentation slide deck
├── panel-1.pptx                               <- Updated presentation slides
└── README.md                                  <- Master project documentation
```

---

## 📊 Dataset Profiles

| Domain | Source & Cohort | Records | Features | Target Variable | Clinical Focus |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Heart Disease** | Cleveland Clinic Foundation (UCI Repository) | **1,025** | 13 | `target` (0: Healthy, 1: Disease) | Chest pain type (`cp`), Resting blood pressure (`trestbps`), Max heart rate (`thalach`), ST depression (`oldpeak`), Major vessels (`ca`). |
| **Parkinson's Disease** | Oxford Telemonitoring Study (Max Little et al.) | **195** | 22 | `status` (0: Healthy, 1: Parkinson's) | Vocal fundamental frequencies (`Fo`, `Fhi`, `Flo`), Jitter (frequency perturbation), Shimmer (amplitude perturbation), `HNR`, `PPE`. |
| **Diabetes** | Pima Indians Diabetes Database (NIDDK) | **768** | 8 | `Outcome` (0: Non-Diabetic, 1: Diabetic) | Fasting glucose, Diastolic blood pressure, Serum insulin, `BMI`, Diabetes pedigree function, Age. |

---

## 🏆 Empirical Benchmark Leaderboard (Holdout Test Cohorts)

All models evaluated on strict **80/20 Stratified Partitioning** with zero data leakage:

| Disease Domain | Classifier Architecture | Holdout Accuracy | Sensitivity (Recall) | Precision | F1-Score | ROC-AUC | Clinical Error Profile |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **🫀 Heart Disease** | **Logistic Regression (Primary)** | **92.25%** | **94.50%** | **90.43%** | **0.9242** | **0.9638** | **FN: 11, FP: 20** |
| 🫀 Heart Disease | Random Forest Ensemble | 92.00% | 91.00% | 92.86% | 0.9192 | 0.9719 | FN: 18, FP: 14 |
| 🫀 Heart Disease | Gradient Boosting Classifier | 91.75% | 92.00% | 91.54% | 0.9177 | 0.9710 | FN: 16, FP: 17 |
| 🫀 Heart Disease | Support Vector Machine (RBF) | 91.25% | 93.00% | 89.86% | 0.9140 | 0.9688 | FN: 14, FP: 21 |
| 🫀 Heart Disease | K-Nearest Neighbors (k=7) | 89.75% | 95.50% | 85.65% | 0.9031 | 0.9558 | FN: 9, FP: 32 |
| 🫀 Heart Disease | Decision Tree (CART) | 85.75% | 84.00% | 87.05% | 0.8550 | 0.9371 | FN: 32, FP: 25 |
| **🧠 Parkinson's** | **Random Forest Ensemble (Top F1)** | **95.42%** | **94.17%** | **96.58%** | **0.9536** | **0.9926** | **FN: 7, FP: 4** |
| **🧠 Parkinson's** | **Decision Tree (Interpretable)** | **93.75%** | **94.17%** | **93.39%** | **0.9378** | **0.9457** | **FN: 7, FP: 8** |
| 🧠 Parkinson's | Support Vector Machine (RBF) | 92.92% | 88.33% | 97.25% | 0.9258 | 0.9849 | FN: 14, FP: 3 |
| 🧠 Parkinson's | K-Nearest Neighbors (k=5) | 91.25% | 93.33% | 89.60% | 0.9143 | 0.9741 | FN: 8, FP: 13 |
| 🧠 Parkinson's | Linear Support Vector Machine | 86.25% | 82.50% | 89.19% | 0.8571 | 0.9283 | FN: 21, FP: 12 |
| **🩸 Type-2 Diabetes**| **Gradient Boosting (Optimal)** | **85.25%** | **88.00%** | **83.41%** | **0.8564** | **0.9361** | **FN: 24, FP: 35** |
| 🩸 Type-2 Diabetes | Random Forest Ensemble | 84.00% | 86.00% | 82.69% | 0.8431 | 0.9156 | FN: 28, FP: 36 |
| 🩸 Type-2 Diabetes | Decision Tree (CART) | 82.25% | 89.00% | 78.41% | 0.8337 | 0.8739 | FN: 22, FP: 49 |
| 🩸 Type-2 Diabetes | Logistic Regression | 80.25% | 79.50% | 80.71% | 0.8010 | 0.8842 | FN: 41, FP: 38 |
| 🩸 Type-2 Diabetes | Support Vector Machine (RBF) | 79.50% | 82.00% | 78.10% | 0.8000 | 0.8917 | FN: 36, FP: 46 |
| 🩸 Type-2 Diabetes | K-Nearest Neighbors (k=7) | 78.50% | 77.50% | 79.08% | 0.7828 | 0.8583 | FN: 45, FP: 41 |

---

## 🚀 Completed Milestones (100% Work Plan Completion)

- [x] **Task 1: Clinical Data Ingestion & Quality Cleaning**
  * Identified physiological anomalies (48.7% zeros in Insulin, 29.6% in SkinThickness).
  * Executed class-stratified median imputation without data leakage.
  * Dropped patient recording identifiers from Parkinson's acoustic data.
- [x] **Task 2: Exploratory Data Analysis & Visualizations**
  * Rendered 18 publication-quality figures at 300 DPI across all three conditions.
  * Produced correlation heatmaps, biomarker pair-plots, and non-linear cluster separations.
- [x] **Task 3: Model Architecture & Training**
  * Trained 17 diverse classifiers across 80/20 stratified holdout cohorts.
  * Serialized all fitted models and StandardScalers into `saved_models/`.
- [x] **Task 4: Model Evaluation & Performance Benchmarking**
  * Evaluated on holdout cohorts: Accuracy, Sensitivity (Recall), Precision, F1-Score, and ROC-AUC.
  * Published Confusion Matrices and Multi-Classifier ROC curves.
  * Minimizing life-threatening false negatives (Recall: 88.00% to 94.50%).
- [x] **Task 5: Production Web Platform & Clinical Risk Factor Engine**
  * High-performance Flask REST API exposing prediction endpoints and model catalogs.
  * Dynamic classifier selector allowing clinicians to compare algorithms in real-time.
  * Automated **Clinical Biomarker Risk Attribution Engine** cross-referencing ACC/AHA and ADA guidelines.
  * Printable patient diagnostic summary report cards and offline client simulation fallback.
- [x] **Task 6: Software Verification & Academic Deliverables**
  * Comprehensive automated test suite (`test_suite.py`) with 100% pass rate.
  * 8-Chapter University Project Report (`PROJECT_REPORT.md`).
  * 13-slide Master Presentation Slide Deck (`final_presentation.pptx`).

---

## 💻 Installation & Quickstart

### 1. Prerequisites
Ensure you have **Python 3.10+** (tested on Python 3.13) installed on Windows, macOS, or Linux.

### 2. Navigate to Project Directory
```powershell
cd C:\Users\Lakshya\OneDrive\Desktop\Minor-Project-1
```

### 3. Install Dependencies
```powershell
pip install pandas numpy scikit-learn matplotlib seaborn joblib flask flask-cors python-pptx
```

### 4. Launch the Clinical Decision Support Web Application
You can launch the web application using the provided Windows batch script:
```powershell
.\run_app.bat
```
Or start the Flask server directly via Python:
```powershell
python app.py
```
Open your browser and navigate to: **`http://127.0.0.1:5000`**

### 5. Run the Automated Verification Test Suite
```powershell
python test_suite.py
```
Output:
```text
======================================================================
   RUNNING AUTOMATED TEST SUITE — CLINICAL DECISION SUPPORT SUITE
======================================================================
  [PASS] Test 01: Health check & active models verified.
  [PASS] Test 02: Metadata, model catalog & guidelines endpoints verified.
  [PASS] Test 03: Heart disease inference across all 6 models & risk engine verified.
  [PASS] Test 04: Diabetes inference across all 6 models & risk engine verified.
  [PASS] Test 05: Parkinson's inference across all 5 models & acoustic engine verified.
  [PASS] Test 06: Batch prediction endpoint verified.
  [PASS] Test 07: Robust error handling & validation verified.
======================================================================
   ALL TESTS PASSED SUCCESSFULLY! (100% PASS RATE)
======================================================================
```

---

## 🌐 RESTful API Endpoints

| Method | Endpoint | Description | Sample Request |
| :--- | :--- | :--- | :--- |
| `GET` | `/` | Web GUI Dashboard | Browser request |
| `GET` | `/health` | API status & loaded model count | None |
| `GET` | `/api/models` | Catalog of all 17 classifiers & benchmarks | None |
| `GET` | `/api/benchmark` | Comparative leaderboard JSON | None |
| `GET` | `/api/guidelines` | ACC/AHA, ADA & Oxford clinical cutoff thresholds | None |
| `POST` | `/predict/heart` | 13-feature Heart Disease prediction | `{"features": [...], "model": "logistic_regression"}` |
| `POST` | `/predict/diabetes` | 8-feature Type-2 Diabetes prediction | `{"features": [...], "model": "gradient_boosting"}` |
| `POST` | `/predict/parkinsons` | 22-feature Parkinson's vocal prediction | `{"features": [...], "model": "decision_tree"}` |
| `POST` | `/api/batch-predict` | Batch cohort prediction across patient lists | `{"disease": "diabetes", "patients": [...]}` |

---

## 👥 Project Team & Supervision

* **Vijay Kumar Rajak** (241B286) — *Model Engineering & Data Preprocessing*
* **Piyush Vishwakarma** (241B177) — *Exploratory Data Analysis & Visualization*
* **Lakshya Asati** (241B134) — *System Architecture, Software Integration & API*

**Supervisor:** **Dr. Rahul Pachauri**  
*Department of Computer Science & Engineering, Jaypee University of Engineering & Technology, Guna (M.P.)*
