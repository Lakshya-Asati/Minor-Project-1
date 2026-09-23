# Multi-Disease Prediction System Using Machine Learning
### Minor Project (Part-I) — Clinical Decision Support Suite

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-v1.6-orange.svg)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626.svg)](https://jupyter.org/)
[![Status](https://img.shields.io/badge/Status-Phase%203%20Complete-brightgreen.svg)]()

An end-to-end Machine Learning suite for the early, non-invasive screening of three major chronic disease domains: **Coronary Heart Disease**, **Parkinson's Disease (Acoustic Vocal Biomarkers)**, and **Type-2 Diabetes Mellitus**.

---

## 📌 Project Overview
Chronic illnesses such as cardiovascular disorders, neurodegenerative conditions, and metabolic diseases frequently advance asymptomatically, causing critical treatment delays. This project develops an interpretable, reproducible machine learning framework to assist clinicians with diagnostic screening.

The system is constructed modularly across sequential, verified stages:
1. **Clinical Data Ingestion & Quality Cleaning** (physiological anomaly handling & data leakage prevention).
2. **Exploratory Data Analysis (EDA) & Visualization** (distribution analysis, correlation mapping, bi-variate clinical interactions).
3. **Model Architecture & Training** (stratified training of 17 diverse classifiers across linear, tree, distance, and ensemble paradigms).
4. **Model Serialization** (persisting trained weights and fitted scalers for future evaluation and deployment).

---

## 🗺️ Project Structure

```text
Minor-Project-1/
│
├── Dataset minor/                             <- Original raw clinical data files
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
├── visualizations/                            <- Rendered publication-quality figures (300 DPI)
│   ├── heart_disease/                         (5 clinical diagnostic figures)
│   ├── parkinsons/                            (5 acoustic vocal biomarker figures)
│   └── diabetes/                              (5 metabolic interaction figures)
│
├── saved_models/                              <- 20 serialized model binaries & scalers (.pkl)
│   ├── heart_*.pkl                            (6 classifiers + 1 StandardScaler)
│   ├── parkinsons_*.pkl                       (5 classifiers + 1 StandardScaler)
│   └── diabetes_*.pkl                         (6 classifiers + 1 StandardScaler)
│
├── 01_data_cleaning.ipynb                     <- [Task 1] Unified clinical data cleaning & auditing
├── 02_heart_disease_visualization.ipynb       <- [Task 2] Heart disease EDA & visualization
├── 02_parkinsons_visualization.ipynb          <- [Task 2] Parkinson's acoustic voice EDA
├── 02_diabetes_visualization.ipynb            <- [Task 2] Diabetes metabolic EDA & imputation checks
├── 03_heart_disease_training.ipynb            <- [Task 3] Heart disease model training & export
├── 03_parkinsons_training.ipynb               <- [Task 3] Parkinson's model training & export
├── 03_diabetes_training.ipynb                 <- [Task 3] Diabetes model training & export
└── README.md                                  <- Project documentation & progress report
```

---

## 📊 Dataset Profiles

| Domain | Source & Cohort | Records | Features | Target Variable | Clinical Focus |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Heart Disease** | Cleveland Clinic Foundation (UCI Repository) | **1,025** | 13 | `target` (0: Healthy, 1: Disease) | Chest pain type (`cp`), Resting blood pressure (`trestbps`), Max heart rate (`thalach`), ST depression (`oldpeak`), Major vessels (`ca`). |
| **Parkinson's Disease** | Oxford Telemonitoring Study (Max Little et al.) | **195** | 22 | `status` (0: Healthy, 1: Parkinson's) | Vocal fundamental frequencies (`Fo`, `Fhi`, `Flo`), Jitter (frequency perturbation), Shimmer (amplitude perturbation), `HNR`, `PPE`. |
| **Diabetes** | Pima Indians Diabetes Database (NIDDK) | **768** | 8 | `Outcome` (0: Non-Diabetic, 1: Diabetic) | Fasting glucose, Diastolic blood pressure, Serum insulin, `BMI`, Diabetes pedigree function, Age. |

---

## 🚀 Progress & Completed Milestones

### ✅ Task 1: Clinical Data Ingestion & Cleaning
* **File:** [`01_data_cleaning.ipynb`](01_data_cleaning.ipynb)
* **Cleveland Heart Disease:**
  * Schema and datatype audit: verified 0 null values across all 1,025 records.
  * Verified clinical parameter ranges (age, resting BP, cholesterol, max heart rate).
* **Oxford Parkinson's Disease:**
  * **Data Leakage Prevention:** Dropped the subject recording identifier column (`name`) so models learn general acoustic vocal pathology rather than memorizing individual subject identifiers.
  * Verified zero missing or infinite values across 22 acoustic continuous features.
* **Pima Indians Diabetes:**
  * **Physiological Zero-Anomaly Audit:** Identified biologically impossible zeros in living patients: `Insulin` (48.7%), `SkinThickness` (29.6%), `BloodPressure` (4.6%), `BMI` (1.4%), and `Glucose` (0.65%).
  * **Class-Stratified Median Imputation:** Replaced zeros with `NaN` and imputed based on the median of the patient's diagnostic outcome class (`Outcome=0` vs `Outcome=1`), preventing distribution distortion and preserving distinct biomarker boundaries.
* **Exports:** Cleaned datasets saved in [`processed_data/`](processed_data/).

---

### ✅ Task 2: Exploratory Data Analysis & Visualization
Each disease dataset is visualized in a dedicated, standalone notebook using `matplotlib` and `seaborn`:

#### 1. 🫀 Heart Disease ([`02_heart_disease_visualization.ipynb`](02_heart_disease_visualization.ipynb))
* Target class distribution (51.3% Disease vs 48.7% Healthy).
* Continuous clinical biomarker distributions (KDE and histogram overlays for `age`, `trestbps`, `chol`, `thalach`, `oldpeak`).
* Masked Pearson correlation matrix across all 13 clinical predictors.
* Multi-panel countplots for categorical features: Chest pain type (`cp`), Sex, Exercise angina (`exang`), and Fluoroscopy vessels (`ca`).
* Bi-variate scatter and linear regression trendline of Age vs Max Heart Rate (`thalach`).

#### 2. 🧠 Parkinson's Disease ([`02_parkinsons_visualization.ipynb`](02_parkinsons_visualization.ipynb))
* Diagnostic status distribution (75.4% Parkinson's vs 24.6% Healthy Control).
* Vocal fundamental frequency ranges violin plots (`MDVP:Fo(Hz)`, `MDVP:Fhi(Hz)`, `MDVP:Flo(Hz)`).
* Boxplot comparisons of key non-linear markers: Pitch Period Entropy (`PPE`), `spread1`, and Harmonics-to-Noise Ratio (`HNR`).
* 22-feature acoustic correlation matrix identifying phonatory instability metrics.
* 2D cluster separation in nonlinear feature space (`PPE` vs `spread1`).

#### 3. 🩸 Diabetes ([`02_diabetes_visualization.ipynb`](02_diabetes_visualization.ipynb))
* Outcome class distribution (65.1% Non-Diabetic vs 34.9% Diabetic).
* **Imputation Validation:** Density curves (KDE) comparing raw vs imputed distributions for `Insulin` and `SkinThickness`, demonstrating restoration of continuous physiological curves.
* Pearson correlation matrix highlighting glucose as the leading linear correlate ($r = 0.49$).
* Glucose vs Insulin co-dependency scatter plot with diagnostic regression trendlines illustrating insulin resistance.
* Stratified boxplots for `BMI` and `Age` risk factors.

---

### ✅ Task 3: Model Architecture & Training (No Evaluation)
Models are trained in dedicated notebooks strictly focusing on **data partitioning, feature scaling, model fitting, and binary serialization**. Test sets have been preserved untouched for the future evaluation phase.

* **Heart Disease Training ([`03_heart_disease_training.ipynb`](03_heart_disease_training.ipynb)):**
  * 80/20 Stratified Split ($N_{\text{train}} = 820$, $N_{\text{test}} = 205$, `random_state=23`).
  * `StandardScaler` fitted strictly on `X_train`.
  * Trained 6 classifiers: Logistic Regression, Decision Tree, Random Forest, RBF SVM, KNN, Gradient Boosting.
* **Parkinson's Disease Training ([`03_parkinsons_training.ipynb`](03_parkinsons_training.ipynb)):**
  * 80/20 Stratified Split ($N_{\text{train}} = 156$, $N_{\text{test}} = 39$, `random_state=42`).
  * `StandardScaler` fitted on continuous acoustic features.
  * Trained 5 classifiers: Linear SVM, RBF SVM, Decision Tree, Random Forest, KNN.
* **Diabetes Training ([`03_diabetes_training.ipynb`](03_diabetes_training.ipynb)):**
  * 80/20 Stratified Split ($N_{\text{train}} = 614$, $N_{\text{test}} = 154$, `random_state=42`).
  * `StandardScaler` fitted on imputed metabolic indicators.
  * Trained 6 classifiers: Logistic Regression, Decision Tree, Random Forest, RBF SVM, KNN, Gradient Boosting.
* **Model Persistence:** 17 classifier `.pkl` files and 3 scaler `.pkl` files exported to [`saved_models/`](saved_models/).

---

## 🔮 Upcoming Milestones
* [ ] **Task 4: Model Evaluation & Performance Benchmarking**
  * Multi-metric testing on preserved test cohorts: Accuracy, Precision, Recall / Sensitivity, Specificity, F1-Score, and ROC-AUC.
  * Clinical explainability: Odds Ratio analysis ($e^\beta$) for Logistic Regression, Gini feature importance for Random Forest.
  * Publication of ROC Curves and Confusion Matrices.
* [ ] **Task 5: Interactive Web Application / Clinical GUI**
  * Streamlit / FastAPI interactive clinical inference dashboard for real-time risk calculation.

---

## 💻 Installation & Quickstart

### 1. Prerequisites
Ensure you have Python 3.10+ installed.

### 2. Clone / Navigate to Directory
```powershell
cd C:\Users\Lakshya\OneDrive\Desktop\Minor-Project-1
```

### 3. Install Dependencies
```powershell
pip install pandas numpy scikit-learn matplotlib seaborn joblib jupyter
```

### 4. Launch Jupyter Notebook
```powershell
jupyter notebook
```
Open any of the numbered notebooks (`01_*`, `02_*`, `03_*`) to view or run the pipelines interactively.
