"""
=============================================================================
  Comprehensive Automated Test Suite — Multi-Disease Prediction System
  Verifies REST API, 17 Classifiers, Scalers, Risk Engine, and Batch Inference
=============================================================================
"""

import sys
import unittest
from app import app, HEART_MODELS, DIABETES_MODELS, PARKINSONS_MODELS

class MultiDiseaseTestSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.testing = True
        cls.client = app.test_client()

    # 1. Health & Status
    def test_01_health_and_status(self):
        res = self.client.get('/health')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data['status'], 'online')
        self.assertEqual(data['total_active_classifiers'], 17)
        self.assertEqual(data['scalers_active'], 3)
        print("  [PASS] Test 01: Health check & active models verified.")

    # 2. Metadata Endpoints
    def test_02_metadata_endpoints(self):
        # Models catalog
        res_models = self.client.get('/api/models')
        self.assertEqual(res_models.status_code, 200)
        models_data = res_models.get_json()
        self.assertEqual(len(models_data['heart_disease']), 6)
        self.assertEqual(len(models_data['diabetes']), 6)
        self.assertEqual(len(models_data['parkinsons']), 5)

        # Benchmark leaderboard
        res_bench = self.client.get('/api/benchmark')
        self.assertEqual(res_bench.status_code, 200)
        bench_data = res_bench.get_json()
        self.assertTrue(bench_data['success'])
        self.assertEqual(len(bench_data['leaderboard']), 3)

        # Guidelines
        res_guide = self.client.get('/api/guidelines')
        self.assertEqual(res_guide.status_code, 200)
        print("  [PASS] Test 02: Metadata, model catalog & guidelines endpoints verified.")

    # 3. Heart Disease Classifiers & Risk Engine
    def test_03_heart_disease_inference(self):
        # Normal patient: clear arteries, normal BP, normal chol, no ST depression, no angina
        normal_payload = {
            "features": [45, 0, 2, 115, 180, 0, 0, 175, 0, 0.0, 0, 0, 2],
            "model": "logistic_regression"
        }
        res_norm = self.client.post('/predict/heart', json=normal_payload)
        self.assertEqual(res_norm.status_code, 200)
        norm_data = res_norm.get_json()
        self.assertTrue(norm_data['success'])
        self.assertEqual(norm_data['prediction'], 0)
        self.assertLess(norm_data['probability_percentage'], 10.0)
        self.assertEqual(norm_data['risk_status'], "Normal / Low Cardiovascular Risk")
        self.assertIn("biomarker_findings", norm_data)
        self.assertGreater(len(norm_data['biomarker_findings']), 0)

        # High risk patient tested across all 6 heart classifiers:
        # typical angina, BP 155, chol 275, ST depression 2.8, 3 blocked vessels, angina
        high_risk_payload = {
            "features": [65, 1, 0, 155, 275, 1, 1, 108, 1, 2.8, 2, 3, 3]
        }
        for model_key in HEART_MODELS.keys():
            payload = dict(high_risk_payload, model=model_key)
            res = self.client.post('/predict/heart', json=payload)
            self.assertEqual(res.status_code, 200, f"Heart model failed: {model_key}")
            data = res.get_json()
            self.assertTrue(data['success'])
            self.assertEqual(data['prediction'], 1)
            self.assertGreater(data['probability_percentage'], 80.0)
            self.assertEqual(data['risk_status'], "High Cardiovascular Risk Detected")
            self.assertIn('probability_percentage', data)
            self.assertIn('model_used', data)
            self.assertEqual(data['model_used']['key'], model_key)
        print("  [PASS] Test 03: Heart disease inference across all 6 models & risk engine verified.")

    # 4. Type-2 Diabetes Classifiers & Risk Engine
    def test_04_diabetes_inference(self):
        normal_payload = {
            "features": [1, 95, 70, 20, 80, 22.4, 0.25, 25],
            "model": "gradient_boosting"
        }
        res_norm = self.client.post('/predict/diabetes', json=normal_payload)
        self.assertEqual(res_norm.status_code, 200)
        norm_data = res_norm.get_json()
        self.assertTrue(norm_data['success'])
        self.assertEqual(norm_data['prediction'], 0)
        self.assertLess(norm_data['probability_percentage'], 10.0)

        high_risk_payload = {
            "features": [6, 168, 88, 38, 220, 35.2, 0.78, 52]
        }
        for model_key in DIABETES_MODELS.keys():
            payload = dict(high_risk_payload, model=model_key)
            res = self.client.post('/predict/diabetes', json=payload)
            self.assertEqual(res.status_code, 200, f"Diabetes model failed: {model_key}")
            data = res.get_json()
            self.assertTrue(data['success'])
            self.assertEqual(data['prediction'], 1)
            self.assertGreater(data['probability_percentage'], 75.0)
            self.assertIn('biomarker_findings', data)
        print("  [PASS] Test 04: Diabetes inference across all 6 models & risk engine verified.")

    # 5. Parkinson's Disease Classifiers & Risk Engine
    def test_05_parkinsons_inference(self):
        normal_payload = {
            "features": [
                197.076, 206.896, 192.055, 0.00289, 0.00001, 0.00166, 
                0.00168, 0.00498, 0.01098, 0.097, 0.00563, 0.0068, 
                0.00802, 0.01689, 0.00339, 26.775, 0.422229, 0.741367, 
                -7.348399, 0.177551, 1.743867, 0.085569
            ],
            "model": "decision_tree"
        }
        res_norm = self.client.post('/predict/parkinsons', json=normal_payload)
        self.assertEqual(res_norm.status_code, 200)
        norm_data = res_norm.get_json()
        self.assertTrue(norm_data['success'])
        self.assertEqual(norm_data['prediction'], 0)
        self.assertLess(norm_data['probability_percentage'], 20.0)

        high_risk_payload = {
            "features": [
                119.992, 157.302, 74.997, 0.00784, 0.00007, 0.0037, 
                0.00554, 0.01109, 0.04374, 0.426, 0.02182, 0.0313, 
                0.02971, 0.06545, 0.02211, 21.033, 0.414783, 0.815285, 
                -4.813031, 0.266482, 2.301442, 0.284654
            ]
        }
        for model_key in PARKINSONS_MODELS.keys():
            payload = dict(high_risk_payload, model=model_key)
            res = self.client.post('/predict/parkinsons', json=payload)
            self.assertEqual(res.status_code, 200, f"Parkinsons model failed: {model_key}")
            data = res.get_json()
            self.assertTrue(data['success'])
            self.assertEqual(data['prediction'], 1)
            self.assertGreater(data['probability_percentage'], 80.0)
            self.assertIn('biomarker_findings', data)
        print("  [PASS] Test 05: Parkinson's inference across all 5 models & acoustic engine verified.")

    # 6. Batch Inference
    def test_06_batch_inference(self):
        batch_payload = {
            "disease": "diabetes",
            "patients": [
                {"features": [1, 95, 70, 20, 80, 22.4, 0.25, 25]},
                {"features": [6, 168, 88, 38, 220, 35.2, 0.78, 52]}
            ]
        }
        res = self.client.post('/api/batch-predict', json=batch_payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data['success'])
        self.assertEqual(data['total_processed'], 2)
        print("  [PASS] Test 06: Batch prediction endpoint verified.")

    # 7. Error Handling & Validation
    def test_07_error_handling(self):
        # Missing features
        res_err1 = self.client.post('/predict/heart', json={"features": [1, 2, 3]})
        self.assertEqual(res_err1.status_code, 400)

        # Empty body
        res_err2 = self.client.post('/predict/diabetes', data="")
        self.assertEqual(res_err2.status_code, 400)

        # Invalid batch disease
        res_err3 = self.client.post('/api/batch-predict', json={"disease": "invalid_disease", "patients": []})
        self.assertEqual(res_err3.status_code, 400)
        print("  [PASS] Test 07: Robust error handling & validation verified.")

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("   RUNNING AUTOMATED TEST SUITE — CLINICAL DECISION SUPPORT SUITE")
    print("=" * 70)
    suite = unittest.TestLoader().loadTestsFromTestCase(MultiDiseaseTestSuite)
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    if result.wasSuccessful():
        print("=" * 70)
        print("   ALL TESTS PASSED SUCCESSFULLY! (100% PASS RATE)")
        print("=" * 70 + "\n")
        sys.exit(0)
    else:
        sys.exit(1)
