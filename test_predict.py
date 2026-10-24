import json
import requests
import sys

BASE_URL = "http://127.0.0.1:5000"

# Sample clinical test data
SAMPLES = {
    "diabetes": {
        "title": "Type-2 Diabetes Mellitus",
        "url": f"{BASE_URL}/predict/diabetes",
        "payload": {
            "features": [6, 148, 72, 35, 0, 33.6, 0.627, 50]
        },
        "description": "Patient: Age 50, Glucose 148, BMI 33.6, BP 72"
    },
    "heart": {
        "title": "Coronary Heart Disease",
        "url": f"{BASE_URL}/predict/heart",
        "payload": {
            "features": [50, 1, 3, 85, 133, 1, 0, 72, 0, 2.3, 0, 0, 1]
        },
        "description": "Patient: Age 63, Male, RestBP 145, Chol 233, MaxHR 150"
    },
    "parkinsons": {
        "title": "Parkinson's Disease (Acoustic Biomarkers)",
        "url": f"{BASE_URL}/predict/parkinsons",
        "payload": {
            "features": [
                119.992, 157.302, 74.997, 0.00784, 0.00007, 0.0037, 
                0.00554, 0.01109, 0.04374, 0.426, 0.02182, 0.0313, 
                0.02971, 0.06545, 0.02211, 21.033, 0.414783, 0.815285, 
                -4.813031, 0.266482, 2.301442, 0.284654
            ]
        },
        "description": "Patient: Voice frequency and vocal perturbation features"
    }
}

def test_disease(key):
    info = SAMPLES[key]
    print("=" * 65)
    print(f"Testing Model : {info['title']}")
    print(f"Sample Input  : {info['description']}")
    print("-" * 65)
    try:
        response = requests.post(info['url'], json=info['payload'], timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f" [+] Disease Name        : {data.get('disease')}")
            print(f" [+] Prediction Code     : {data.get('prediction')} (1 = Positive / High Risk, 0 = Normal)")
            print(f" [+] Risk Status         : {data.get('risk_status')}")
            print(f" [+] Probability / Score : {data.get('probability_percentage')}%")
            print(f" [+] Clinical Advice     : {data.get('clinical_notes')}")
        else:
            print(f" [-] HTTP Error {response.status_code}: {response.text}")
    except requests.exceptions.ConnectionError:
        print(" [!] ERROR: Could not connect to backend.")
        print("     Make sure 'python app.py' is running in another terminal window!")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    target = sys.argv[1].lower() if len(sys.argv) > 1 else "all"

    print("\n" + "#" * 65)
    print("   MULTI-DISEASE BACKEND PREDICTION TEST HARNESS")
    print("#" * 65 + "\n")

    if target in SAMPLES:
        test_disease(target)
    else:
        for k in SAMPLES:
            test_disease(k)
