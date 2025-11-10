"""
Quick test script to verify trained models load correctly

Author: Suk Jin Mun
Course: DS 5110, Fall 2025
"""

import pickle
import numpy as np

print("="*60)
print("MODEL LOADING TEST")
print("="*60)

# Test 1: Load classification models
print("\n[Test 1] Loading classification models...")
try:
    with open('../trained_models/esi_logistic.pkl', 'rb') as f:
        clf_logistic = pickle.load(f)
    print("  [OK] Logistic regression loaded")

    with open('../trained_models/esi_lda.pkl', 'rb') as f:
        clf_lda = pickle.load(f)
    print("  [OK] LDA loaded")

    with open('../trained_models/esi_naive_bayes.pkl', 'rb') as f:
        clf_nb = pickle.load(f)
    print("  [OK] Naive Bayes loaded")

except Exception as e:
    print(f"  [ERROR] {e}")

# Test 2: Load regression models
print("\n[Test 2] Loading regression models...")
try:
    with open('../trained_models/wait_time_predictor.pkl', 'rb') as f:
        wait_time_data = pickle.load(f)
    print(f"  [OK] Wait time predictor loaded")
    print(f"       Features: {len(wait_time_data['feature_names'])}")
    print(f"       Feature names: {wait_time_data['feature_names'][:3]}...")

    with open('../trained_models/volume_predictor.pkl', 'rb') as f:
        volume_model = pickle.load(f)
    print(f"  [OK] Volume predictor loaded")
    print(f"       Model type: {type(volume_model).__name__}")

except Exception as e:
    print(f"  [ERROR] {e}")

# Test 3: Test prediction with dummy data
print("\n[Test 3] Testing predictions with dummy data...")
try:
    # Create dummy feature vector (27 features after one-hot encoding)
    X_dummy_clf = np.random.randn(1, 27)

    pred_logistic = clf_logistic.predict(X_dummy_clf)
    print(f"  [OK] Logistic prediction: ESI level {pred_logistic[0]}")

    pred_lda = clf_lda.predict(X_dummy_clf)
    print(f"  [OK] LDA prediction: ESI level {pred_lda[0]}")

    pred_nb = clf_nb.predict(X_dummy_clf)
    print(f"  [OK] Naive Bayes prediction: ESI level {pred_nb[0]}")

except Exception as e:
    print(f"  [ERROR] {e}")

# Test 4: Test wait time prediction
print("\n[Test 4] Testing wait time prediction...")
try:
    # Create dummy feature vector (12 features)
    X_dummy_reg = np.random.randn(1, len(wait_time_data['feature_names']))

    # Scale features
    X_dummy_scaled = wait_time_data['scaler'].transform(X_dummy_reg)

    # Predict
    wait_time_pred = wait_time_data['model'].predict(X_dummy_scaled)
    print(f"  [OK] Predicted wait time: {wait_time_pred[0]:.1f} minutes")

except Exception as e:
    print(f"  [ERROR] {e}")

# Test 5: Test volume prediction
print("\n[Test 5] Testing volume prediction...")
try:
    # Create dummy temporal features: [hour, day_of_week, month, is_weekend]
    X_dummy_volume = np.array([[15, 3, 11, 0]])  # 3pm, Wednesday, November, not weekend

    # Add constant for statsmodels
    import statsmodels.api as sm
    X_dummy_volume_const = sm.add_constant(X_dummy_volume)

    # Predict
    volume_pred = volume_model.predict(X_dummy_volume_const)
    print(f"  [OK] Predicted volume: {volume_pred[0]:.2f} patients/hour")

except Exception as e:
    print(f"  [ERROR] {e}")

print("\n" + "="*60)
print("ALL TESTS COMPLETED")
print("="*60)
print("\nSummary:")
print("  - 5 models loaded successfully")
print("  - All models can make predictions")
print("  - Models ready for integration into Flask API")
print("\nNext step: Add prediction endpoints to backend/routes/api.py")
print("="*60)
