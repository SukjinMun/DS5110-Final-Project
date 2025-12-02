"""
Prediction API endpoints for trained models

Provides endpoints for ESI classification, wait time prediction, and volume forecasting.

Author: Suk Jin Mun
Course: DS 5110, Fall 2025
"""

from flask import Blueprint, jsonify, request
import pickle
import numpy as np
import pandas as pd
import os
import statsmodels.api as sm

predictions_bp = Blueprint('predictions', __name__)

# Global variables to store loaded models
_models_loaded = False
_esi_logistic = None
_esi_lda = None
_esi_nb = None
_esi_rf = None
_esi_gb = None
_wait_time_model = None
_volume_model = None

def load_models():
    """Load all trained models on first request"""
    global _models_loaded, _esi_logistic, _esi_lda, _esi_nb, _esi_rf, _esi_gb, _wait_time_model, _volume_model

    if _models_loaded:
        return

    models_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'trained_models')

    try:
        # Load classification models (new format: dict with 'model' and 'scaler')
        with open(os.path.join(models_dir, 'esi_logistic.pkl'), 'rb') as f:
            _esi_logistic = pickle.load(f)

        with open(os.path.join(models_dir, 'esi_lda.pkl'), 'rb') as f:
            _esi_lda = pickle.load(f)

        with open(os.path.join(models_dir, 'esi_naive_bayes.pkl'), 'rb') as f:
            _esi_nb = pickle.load(f)

        with open(os.path.join(models_dir, 'esi_random_forest.pkl'), 'rb') as f:
            _esi_rf = pickle.load(f)

        with open(os.path.join(models_dir, 'esi_gradient_boosting.pkl'), 'rb') as f:
            _esi_gb = pickle.load(f)

        # Load regression models
        with open(os.path.join(models_dir, 'wait_time_predictor.pkl'), 'rb') as f:
            _wait_time_model = pickle.load(f)

        with open(os.path.join(models_dir, 'volume_predictor.pkl'), 'rb') as f:
            _volume_model = pickle.load(f)

        _models_loaded = True
        print("[INFO] All models loaded successfully")

    except Exception as e:
        print(f"[ERROR] Failed to load models: {e}")
        raise


@predictions_bp.route('/models/info', methods=['GET'])
def models_info():
    """Get information about available models"""
    load_models()

    return jsonify({
        'classification_models': {
            'random_forest': {
                'description': 'Random Forest for ESI prediction (BEST ACCURACY)',
                'accuracy': '94.06%',
                'weighted_f1': '0.9401',
                'macro_recall': '93.05%',
                'best_for': 'Highest accuracy classification',
                'endpoint': '/api/predictions/esi'
            },
            'logistic_regression': {
                'description': 'Logistic Regression with SMOTE for ESI prediction',
                'accuracy': '93.44%',
                'weighted_f1': '0.9350',
                'macro_recall': '93.73%',
                'best_for': 'Best recall for critical ESI levels',
                'endpoint': '/api/predictions/esi'
            },
            'gradient_boosting': {
                'description': 'Gradient Boosting for ESI prediction',
                'accuracy': '93.28%',
                'weighted_f1': '0.9326',
                'macro_recall': '93.00%',
                'best_for': 'Strong accuracy with good generalization',
                'endpoint': '/api/predictions/esi'
            },
            'lda': {
                'description': 'Linear Discriminant Analysis with SMOTE for ESI prediction',
                'accuracy': '90.16%',
                'weighted_f1': '0.9026',
                'macro_recall': '91.96%',
                'best_for': 'Probabilistic classification with balanced recall',
                'endpoint': '/api/predictions/esi'
            },
            'naive_bayes': {
                'description': 'Gaussian Naive Bayes with SMOTE for ESI prediction',
                'accuracy': '90.16%',
                'weighted_f1': '0.9019',
                'macro_recall': '90.72%',
                'best_for': 'Fast predictions',
                'endpoint': '/api/predictions/esi'
            }
        },
        'regression_models': {
            'wait_time': {
                'description': 'Linear regression for wait time prediction',
                'r2_score': '0.8463',
                'rmse': '13.63 minutes',
                'mae': '10.87 minutes',
                'endpoint': '/api/predictions/wait-time'
            },
            'volume': {
                'description': 'Poisson GLM for patient volume forecasting',
                'rmse': '0.84 patients/hour',
                'mae': '0.66 patients/hour',
                'endpoint': '/api/predictions/volume'
            }
        },
        'feature_requirements': {
            'esi_prediction': _wait_time_model['feature_names'] if _wait_time_model else [],
            'wait_time_prediction': _wait_time_model['feature_names'] if _wait_time_model else [],
            'volume_prediction': ['hour', 'day_of_week', 'month', 'is_weekend']
        }
    })


@predictions_bp.route('/esi', methods=['POST'])
def predict_esi():
    """
    Predict ESI level from patient characteristics

    Request body:
    {
        "model": "random_forest" | "gradient_boosting" | "logistic" | "lda" | "naive_bayes",
        "features": {
            "patient_age": int,
            "sex_at_birth": "M" | "F",
            "arrival_mode": "Walk-in" | "EMS" | "Transfer",
            "chief_complaint": str,
            "heart_rate": float,
            "bp_systolic": float,
            "bp_diastolic": float,
            "respiratory_rate": float,
            "temperature_c": float,
            "o2_saturation": float,
            "pain_score": int (0-10),
            "arrival_hour": int (0-23),
            "arrival_day_of_week": int (0-6),
            "is_weekend": 0 | 1,
            "payor_type": "public" | "private" | "self"
        }
    }
    """
    load_models()

    try:
        data = request.get_json()
        model_type = data.get('model', 'random_forest')  # Default to best model
        features = data.get('features', {})

        # Validate required features
        required = ['patient_age', 'sex_at_birth', 'arrival_mode', 'chief_complaint',
                   'heart_rate', 'bp_systolic', 'bp_diastolic', 'respiratory_rate',
                   'temperature_c', 'o2_saturation', 'pain_score',
                   'arrival_hour', 'arrival_day_of_week', 'is_weekend', 'payor_type']

        missing = [f for f in required if f not in features]
        if missing:
            return jsonify({'error': f'Missing required features: {missing}'}), 400

        # Create DataFrame for one-hot encoding (matches training format)
        df = pd.DataFrame([features])

        # One-hot encode categorical variables
        df_encoded = pd.get_dummies(df,
                                    columns=['sex_at_birth', 'arrival_mode', 'chief_complaint', 'payor_type'],
                                    drop_first=True)

        # Ensure all expected columns exist (add missing with 0)
        # This handles cases where the input doesn't have all categorical values
        expected_cols = 27  # From training
        if len(df_encoded.columns) < expected_cols:
            for i in range(len(df_encoded.columns), expected_cols):
                df_encoded[f'dummy_{i}'] = 0

        # Select model (all models now contain dict with 'model' and 'scaler')
        model_dict = None
        if model_type == 'random_forest':
            model_dict = _esi_rf
        elif model_type == 'gradient_boosting':
            model_dict = _esi_gb
        elif model_type == 'logistic':
            model_dict = _esi_logistic
        elif model_type == 'lda':
            model_dict = _esi_lda
        elif model_type == 'naive_bayes':
            model_dict = _esi_nb
        else:
            return jsonify({'error': f'Unknown model type: {model_type}. Options: random_forest, gradient_boosting, logistic, lda, naive_bayes'}), 400

        # Get model and scaler from dict
        model = model_dict['model']
        scaler = model_dict['scaler']

        # Predict (scale features first)
        X = df_encoded.iloc[:, :27].values  # Take first 27 columns
        X_scaled = scaler.transform(X)
        prediction = model.predict(X_scaled)[0]
        probabilities = model.predict_proba(X_scaled)[0] if hasattr(model, 'predict_proba') else None

        result = {
            'predicted_esi_level': int(prediction),
            'model_used': model_type,
            'confidence': {
                'esi_1': float(probabilities[0]) if probabilities is not None else None,
                'esi_2': float(probabilities[1]) if probabilities is not None else None,
                'esi_3': float(probabilities[2]) if probabilities is not None else None,
                'esi_4': float(probabilities[3]) if probabilities is not None else None,
                'esi_5': float(probabilities[4]) if probabilities is not None else None
            } if probabilities is not None else None,
            'interpretation': get_esi_interpretation(int(prediction))
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@predictions_bp.route('/wait-time', methods=['POST'])
def predict_wait_time():
    """
    Predict wait time from patient characteristics

    Request body:
    {
        "features": {
            "esi_level": int (1-5),
            "patient_age": int,
            "sex_at_birth": "M" | "F",
            "arrival_mode": "Walk-in" | "EMS" | "Transfer",
            "heart_rate": float,
            "bp_systolic": float,
            "respiratory_rate": float,
            "temperature_c": float,
            "o2_saturation": float,
            "arrival_hour": int (0-23),
            "is_weekend": 0 | 1
        }
    }
    """
    load_models()

    try:
        data = request.get_json()
        features = data.get('features', {})

        # Validate required features
        required = ['esi_level', 'patient_age', 'sex_at_birth', 'arrival_mode',
                   'heart_rate', 'bp_systolic', 'respiratory_rate',
                   'temperature_c', 'o2_saturation', 'arrival_hour', 'is_weekend']

        missing = [f for f in required if f not in features]
        if missing:
            return jsonify({'error': f'Missing required features: {missing}'}), 400

        # Create DataFrame
        df = pd.DataFrame([features])

        # One-hot encode categorical variables
        df_encoded = pd.get_dummies(df,
                                    columns=['sex_at_birth', 'arrival_mode'],
                                    drop_first=True)

        # Align with training features
        feature_names = _wait_time_model['feature_names']
        for col in feature_names:
            if col not in df_encoded.columns:
                df_encoded[col] = 0

        X = df_encoded[feature_names].values

        # Scale features
        X_scaled = _wait_time_model['scaler'].transform(X)

        # Predict
        wait_time = _wait_time_model['model'].predict(X_scaled)[0]

        result = {
            'predicted_wait_time_minutes': float(max(0, wait_time)),  # Ensure non-negative
            'predicted_wait_time_formatted': format_minutes(max(0, wait_time)),
            'confidence_interval': {
                'note': 'Point estimate only. Confidence intervals require additional computation.'
            },
            'interpretation': get_wait_time_interpretation(wait_time, features['esi_level'])
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@predictions_bp.route('/volume', methods=['GET'])
def predict_volume():
    """
    Predict patient arrival volume

    Query parameters:
    - hour: int (0-23) - Hour of day
    - day_of_week: int (0-6) - Day of week (0=Monday)
    - month: int (1-12) - Month
    - is_weekend: int (0|1) - Weekend indicator

    Example: /api/predictions/volume?hour=15&day_of_week=3&month=11&is_weekend=0
    """
    load_models()

    try:
        hour = request.args.get('hour', type=int)
        day_of_week = request.args.get('day_of_week', type=int)
        month = request.args.get('month', type=int)
        is_weekend = request.args.get('is_weekend', type=int, default=0)

        # Validate
        if hour is None or day_of_week is None or month is None:
            return jsonify({'error': 'Missing required parameters: hour, day_of_week, month'}), 400

        if not (0 <= hour <= 23):
            return jsonify({'error': 'hour must be 0-23'}), 400
        if not (0 <= day_of_week <= 6):
            return jsonify({'error': 'day_of_week must be 0-6'}), 400
        if not (1 <= month <= 12):
            return jsonify({'error': 'month must be 1-12'}), 400

        # Create feature vector
        X = np.array([[hour, day_of_week, month, is_weekend]])

        # Add constant for statsmodels (force add for single-row prediction)
        X_const = sm.add_constant(X, has_constant='add')

        # Predict
        volume = _volume_model.predict(X_const)[0]

        result = {
            'predicted_volume_per_hour': float(volume),
            'predicted_arrivals_4_hours': float(volume * 4),
            'predicted_arrivals_8_hours': float(volume * 8),
            'input': {
                'hour': hour,
                'day_of_week': day_of_week,
                'day_name': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][day_of_week],
                'month': month,
                'is_weekend': bool(is_weekend)
            },
            'interpretation': get_volume_interpretation(volume, is_weekend, hour)
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Helper functions

def get_esi_interpretation(esi_level):
    """Get clinical interpretation of ESI level"""
    interpretations = {
        1: "Level 1 (Resuscitation): Immediate life-saving intervention required",
        2: "Level 2 (Emergent): High risk, confused/lethargic/disoriented, severe pain/distress",
        3: "Level 3 (Urgent): Moderate symptoms, stable vital signs, may need multiple resources",
        4: "Level 4 (Less Urgent): Minor symptoms, one resource expected",
        5: "Level 5 (Non-Urgent): Chronic or minor problem, no resources expected"
    }
    return interpretations.get(esi_level, "Unknown ESI level")


def get_wait_time_interpretation(wait_time, esi_level):
    """Get interpretation of predicted wait time"""
    if wait_time < 0:
        return "Invalid wait time prediction"
    elif wait_time < 15:
        status = "Immediate care"
    elif wait_time < 30:
        status = "Short wait"
    elif wait_time < 60:
        status = "Moderate wait"
    elif wait_time < 120:
        status = "Long wait"
    else:
        status = "Very long wait"

    return f"{status}. ESI {esi_level} patients typically experience this range of wait times."


def get_volume_interpretation(volume, is_weekend, hour):
    """Get interpretation of predicted volume"""
    avg_volume = 1.58  # From training data

    if volume < avg_volume * 0.5:
        status = "Low volume"
    elif volume < avg_volume * 1.5:
        status = "Normal volume"
    else:
        status = "High volume"

    weekend_note = " (weekends typically 29% higher)" if is_weekend else ""
    time_note = ""
    if 6 <= hour <= 10:
        time_note = " Morning hours show increasing arrivals."
    elif 18 <= hour <= 22:
        time_note = " Evening hours typically peak."

    return f"{status}{weekend_note}.{time_note}"


def format_minutes(minutes):
    """Format minutes as hours and minutes"""
    hours = int(minutes // 60)
    mins = int(minutes % 60)

    if hours == 0:
        return f"{mins} minutes"
    elif hours == 1:
        return f"1 hour {mins} minutes"
    else:
        return f"{hours} hours {mins} minutes"
