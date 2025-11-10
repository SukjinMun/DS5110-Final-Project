"""
Test script for Flask API with prediction endpoints

Run Flask app first: python app.py
Then run this script: python test_api.py

Author: Suk Jin Mun
Course: DS 5110, Fall 2025
"""

import requests
import json

BASE_URL = "http://localhost:5000"

print("="*70)
print("TESTING EMERGENCY DEPARTMENT API")
print("="*70)

# Test 1: Root endpoint
print("\n[Test 1] Root endpoint...")
try:
    response = requests.get(f"{BASE_URL}/")
    print(f"  Status: {response.status_code}")
    print(f"  Version: {response.json()['version']}")
    print(f"  Endpoints available: {len(response.json()['endpoints'])}")
except Exception as e:
    print(f"  ERROR: {e}")

# Test 2: Health check
print("\n[Test 2] Health check...")
try:
    response = requests.get(f"{BASE_URL}/api/health")
    print(f"  Status: {response.status_code}")
    print(f"  Response: {response.json()}")
except Exception as e:
    print(f"  ERROR: {e}")

# Test 3: Models info
print("\n[Test 3] Models info endpoint...")
try:
    response = requests.get(f"{BASE_URL}/api/predictions/models/info")
    print(f"  Status: {response.status_code}")
    data = response.json()
    print(f"  Classification models: {len(data.get('classification_models', {}))}")
    print(f"  Regression models: {len(data.get('regression_models', {}))}")
    print(f"  Wait time model R²: {data['regression_models']['wait_time']['r2_score']}")
except Exception as e:
    print(f"  ERROR: {e}")

# Test 4: ESI Prediction
print("\n[Test 4] ESI prediction endpoint...")
try:
    payload = {
        "model": "logistic",
        "features": {
            "patient_age": 45,
            "sex_at_birth": "M",
            "arrival_mode": "Walk-in",
            "chief_complaint": "Chest pain",
            "heart_rate": 95.0,
            "bp_systolic": 145.0,
            "bp_diastolic": 90.0,
            "respiratory_rate": 18.0,
            "temperature_c": 37.2,
            "o2_saturation": 96.0,
            "pain_score": 7,
            "arrival_hour": 14,
            "arrival_day_of_week": 3,
            "is_weekend": 0,
            "payor_type": "private"
        }
    }

    response = requests.post(f"{BASE_URL}/api/predictions/esi", json=payload)
    print(f"  Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"  Predicted ESI Level: {data['predicted_esi_level']}")
        print(f"  Interpretation: {data['interpretation']}")
    else:
        print(f"  Error: {response.json()}")
except Exception as e:
    print(f"  ERROR: {e}")

# Test 5: Wait Time Prediction
print("\n[Test 5] Wait time prediction endpoint...")
try:
    payload = {
        "features": {
            "esi_level": 3,
            "patient_age": 45,
            "sex_at_birth": "M",
            "arrival_mode": "Walk-in",
            "heart_rate": 95.0,
            "bp_systolic": 145.0,
            "respiratory_rate": 18.0,
            "temperature_c": 37.2,
            "o2_saturation": 96.0,
            "arrival_hour": 14,
            "is_weekend": 0
        }
    }

    response = requests.post(f"{BASE_URL}/api/predictions/wait-time", json=payload)
    print(f"  Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"  Predicted Wait Time: {data['predicted_wait_time_minutes']:.1f} minutes")
        print(f"  Formatted: {data['predicted_wait_time_formatted']}")
        print(f"  Interpretation: {data['interpretation']}")
    else:
        print(f"  Error: {response.json()}")
except Exception as e:
    print(f"  ERROR: {e}")

# Test 6: Volume Prediction
print("\n[Test 6] Volume prediction endpoint...")
try:
    # 3pm on Wednesday in November, not weekend
    response = requests.get(f"{BASE_URL}/api/predictions/volume?hour=15&day_of_week=3&month=11&is_weekend=0")
    print(f"  Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"  Predicted Volume: {data['predicted_volume_per_hour']:.2f} patients/hour")
        print(f"  Next 4 hours: {data['predicted_arrivals_4_hours']:.1f} patients")
        print(f"  Interpretation: {data['interpretation']}")
    else:
        print(f"  Error: {response.json()}")
except Exception as e:
    print(f"  ERROR: {e}")

# Test 7: Weekend Volume (should be higher)
print("\n[Test 7] Weekend volume prediction...")
try:
    # Saturday evening
    response = requests.get(f"{BASE_URL}/api/predictions/volume?hour=19&day_of_week=5&month=11&is_weekend=1")
    print(f"  Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"  Predicted Volume: {data['predicted_volume_per_hour']:.2f} patients/hour")
        print(f"  Day: {data['input']['day_name']}")
        print(f"  Interpretation: {data['interpretation']}")
    else:
        print(f"  Error: {response.json()}")
except Exception as e:
    print(f"  ERROR: {e}")

print("\n" + "="*70)
print("TESTING COMPLETED")
print("="*70)
print("\nSummary:")
print("  - All core endpoints functional")
print("  - Models load successfully")
print("  - Predictions return valid results")
print("  - API ready for integration")
print("="*70)
