"""
Complete model training pipeline for Emergency Department Analysis

This script:
1. Loads corrected dataset with ISO 8601 timestamps
2. Engineers features (wait times, temporal features, one-hot encoding)
3. Trains classification models (ESI prediction)
4. Trains regression models (wait time prediction)
5. Evaluates models and saves results
6. Saves trained models to pickle files

Author: Suk Jin Mun
Course: DS 5110, Fall 2025
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

import pandas as pd
import numpy as np
from datetime import datetime
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                             r2_score, mean_squared_error, mean_absolute_error,
                             f1_score, recall_score, precision_score)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
import statsmodels.api as sm
from statsmodels.genmod.families import Poisson
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

print("="*80)
print("Emergency Department Model Training Pipeline")
print("="*80)

# ============================================================================
# STEP 1: Load Dataset
# ============================================================================
print("\n[STEP 1] Loading corrected dataset...")

# Load all tables
encounters = pd.read_csv('../dataset/encounter.csv')
patients = pd.read_csv('../dataset/patient.csv')
vitals = pd.read_csv('../dataset/vitals.csv')
diagnoses = pd.read_csv('../dataset/diagnosis.csv')
payors = pd.read_csv('../dataset/encounter_payor.csv')

print(f"  [OK] Loaded {len(encounters)} encounters")
print(f"  [OK] Loaded {len(patients)} patients")
print(f"  [OK] Loaded {len(vitals)} vitals records")
print(f"  [OK] Loaded {len(diagnoses)} diagnosis records")
print(f"  [OK] Loaded {len(payors)} payor records")

# ============================================================================
# STEP 2: Feature Engineering
# ============================================================================
print("\n[STEP 2] Feature engineering...")

# Parse timestamps
encounters['arrival_ts'] = pd.to_datetime(encounters['arrival_ts'])
encounters['triage_start_ts'] = pd.to_datetime(encounters['triage_start_ts'])
encounters['triage_end_ts'] = pd.to_datetime(encounters['triage_end_ts'])
encounters['provider_start_ts'] = pd.to_datetime(encounters['provider_start_ts'])
encounters['dispo_decision_ts'] = pd.to_datetime(encounters['dispo_decision_ts'])
encounters['departure_ts'] = pd.to_datetime(encounters['departure_ts'])

# Calculate wait time (arrival to provider) in minutes
encounters['wait_time_minutes'] = (
    (encounters['provider_start_ts'] - encounters['arrival_ts']).dt.total_seconds() / 60
)

# Calculate length of stay in minutes
encounters['los_minutes'] = (
    (encounters['departure_ts'] - encounters['arrival_ts']).dt.total_seconds() / 60
)

# Extract temporal features
encounters['arrival_hour'] = encounters['arrival_ts'].dt.hour
encounters['arrival_day_of_week'] = encounters['arrival_ts'].dt.dayofweek
encounters['arrival_month'] = encounters['arrival_ts'].dt.month
encounters['is_weekend'] = encounters['arrival_day_of_week'].isin([5, 6]).astype(int)

print(f"  [OK] Calculated wait times (mean: {encounters['wait_time_minutes'].mean():.1f} min)")
print(f"  [OK] Calculated length of stay (mean: {encounters['los_minutes'].mean():.1f} min)")
print(f"  [OK] Extracted temporal features")

# Merge with patient demographics
patients['dob'] = pd.to_datetime(patients['dob'])
df = encounters.merge(patients, on='patient_id', how='left')

# Calculate patient age in years
df['patient_age'] = ((df['arrival_ts'] - df['dob']).dt.days / 365.25).astype(int)

# Merge first vitals for each encounter
first_vitals = vitals.sort_values('taken_ts').groupby('encounter_id').first().reset_index()
df = df.merge(first_vitals[['encounter_id', 'heart_rate', 'systolic_bp', 'diastolic_bp',
                             'respiratory_rate', 'temperature_c', 'spo2', 'pain_score']],
              on='encounter_id', how='left')

# Rename vital columns for consistency
df = df.rename(columns={'systolic_bp': 'bp_systolic', 'diastolic_bp': 'bp_diastolic', 'spo2': 'o2_saturation'})

# Merge payor information
df = df.merge(payors[['encounter_id', 'payor_name', 'payor_type']],
              on='encounter_id', how='left')

print(f"  [OK] Merged patient demographics")
print(f"  [OK] Merged vitals (first measurement per encounter)")
print(f"  [OK] Merged payor information")

# ============================================================================
# STEP 3: Prepare Features for Classification (ESI Prediction)
# ============================================================================
print("\n[STEP 3] Preparing features for ESI classification...")

# Select features for classification
classification_features = [
    'patient_age', 'sex_at_birth', 'arrival_mode', 'chief_complaint',
    'heart_rate', 'bp_systolic', 'bp_diastolic', 'respiratory_rate',
    'temperature_c', 'o2_saturation', 'pain_score',
    'arrival_hour', 'arrival_day_of_week', 'is_weekend',
    'payor_type'
]

# Create a clean dataset for classification (remove rows with missing ESI or vitals)
df_classification = df[classification_features + ['esi_level']].copy()
df_classification = df_classification.dropna()

print(f"  [OK] Classification dataset: {len(df_classification)} samples")

# One-hot encode categorical variables
df_classification_encoded = pd.get_dummies(df_classification,
                                           columns=['sex_at_birth', 'arrival_mode', 'chief_complaint', 'payor_type'],
                                           drop_first=True)

# Separate features and target
X_clf = df_classification_encoded.drop('esi_level', axis=1)
y_clf = df_classification_encoded['esi_level']

print(f"  [OK] Feature matrix shape: {X_clf.shape}")
print(f"  [OK] Target distribution:\n{y_clf.value_counts().sort_index()}")

# Split into train/test
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf, test_size=0.3, random_state=42, stratify=y_clf
)

print(f"  [OK] Training set: {len(X_train_clf)} samples")
print(f"  [OK] Test set: {len(X_test_clf)} samples")

# Scale features for better model performance
print("\n  [SCALING] Standardizing features...")
scaler_clf = StandardScaler()
X_train_clf_scaled = scaler_clf.fit_transform(X_train_clf)
X_test_clf_scaled = scaler_clf.transform(X_test_clf)
X_train_clf_scaled = pd.DataFrame(X_train_clf_scaled, columns=X_train_clf.columns)
X_test_clf_scaled = pd.DataFrame(X_test_clf_scaled, columns=X_test_clf.columns)
print(f"  [OK] Features standardized")

# Apply SMOTE to handle class imbalance (on scaled data)
print("\n  [SMOTE] Applying SMOTE oversampling to handle class imbalance...")
smote = SMOTE(random_state=42)
X_train_clf_smote, y_train_clf_smote = smote.fit_resample(X_train_clf_scaled, y_train_clf)
print(f"  [OK] After SMOTE: {len(X_train_clf_smote)} samples")
print(f"  [OK] Resampled target distribution:\n{pd.Series(y_train_clf_smote).value_counts().sort_index()}")

# ============================================================================
# STEP 4: Train Classification Models
# ============================================================================
print("\n[STEP 4] Training classification models...")

# 4.1: Logistic Regression with SMOTE data (scaled)
print("\n  [4.1] Logistic Regression with SMOTE (scaled features)...")
clf_logistic = LogisticRegression(max_iter=5000, random_state=42, solver='lbfgs')
clf_logistic.fit(X_train_clf_smote, y_train_clf_smote)
y_pred_logistic = clf_logistic.predict(X_test_clf_scaled)

acc_logistic = accuracy_score(y_test_clf, y_pred_logistic)
f1_logistic = f1_score(y_test_clf, y_pred_logistic, average='weighted')
recall_logistic = recall_score(y_test_clf, y_pred_logistic, average='macro')
print(f"    [OK] Accuracy: {acc_logistic:.4f}")
print(f"    [OK] Weighted F1: {f1_logistic:.4f}")
print(f"    [OK] Macro Recall: {recall_logistic:.4f}")
print("\n    Classification Report:")
print(classification_report(y_test_clf, y_pred_logistic))

# Cross-validation (on scaled data)
X_clf_scaled = scaler_clf.fit_transform(X_clf)
cv_scores_logistic = cross_val_score(clf_logistic, X_clf_scaled, y_clf, cv=5, scoring='f1_weighted')
print(f"    [OK] 5-Fold CV Weighted F1: {cv_scores_logistic.mean():.4f} (+/- {cv_scores_logistic.std():.4f})")

# Save model with scaler
with open('../trained_models/esi_logistic.pkl', 'wb') as f:
    pickle.dump({'model': clf_logistic, 'scaler': scaler_clf}, f)
print(f"    [OK] Model saved to ../trained_models/esi_logistic.pkl")

# 4.2: Linear Discriminant Analysis (trained on SMOTE data, scaled)
print("\n  [4.2] Linear Discriminant Analysis (LDA) with SMOTE (scaled)...")
clf_lda = LinearDiscriminantAnalysis()
clf_lda.fit(X_train_clf_smote, y_train_clf_smote)
y_pred_lda = clf_lda.predict(X_test_clf_scaled)

acc_lda = accuracy_score(y_test_clf, y_pred_lda)
f1_lda = f1_score(y_test_clf, y_pred_lda, average='weighted')
recall_lda = recall_score(y_test_clf, y_pred_lda, average='macro')
print(f"    [OK] Accuracy: {acc_lda:.4f}")
print(f"    [OK] Weighted F1: {f1_lda:.4f}")
print(f"    [OK] Macro Recall: {recall_lda:.4f}")
print("\n    Classification Report:")
print(classification_report(y_test_clf, y_pred_lda))

# Save model with scaler
with open('../trained_models/esi_lda.pkl', 'wb') as f:
    pickle.dump({'model': clf_lda, 'scaler': scaler_clf}, f)
print(f"    [OK] Model saved to ../trained_models/esi_lda.pkl")

# 4.3: Gaussian Naive Bayes (trained on SMOTE data, scaled)
print("\n  [4.3] Gaussian Naive Bayes with SMOTE (scaled)...")
clf_nb = GaussianNB()
clf_nb.fit(X_train_clf_smote, y_train_clf_smote)
y_pred_nb = clf_nb.predict(X_test_clf_scaled)

acc_nb = accuracy_score(y_test_clf, y_pred_nb)
f1_nb = f1_score(y_test_clf, y_pred_nb, average='weighted')
recall_nb = recall_score(y_test_clf, y_pred_nb, average='macro')
print(f"    [OK] Accuracy: {acc_nb:.4f}")
print(f"    [OK] Weighted F1: {f1_nb:.4f}")
print(f"    [OK] Macro Recall: {recall_nb:.4f}")
print("\n    Classification Report:")
print(classification_report(y_test_clf, y_pred_nb))

# Save model with scaler
with open('../trained_models/esi_naive_bayes.pkl', 'wb') as f:
    pickle.dump({'model': clf_nb, 'scaler': scaler_clf}, f)
print(f"    [OK] Model saved to ../trained_models/esi_naive_bayes.pkl")

# 4.4: Random Forest - tuned for >80% accuracy
print("\n  [4.4] Random Forest (tuned hyperparameters)...")
clf_rf = RandomForestClassifier(
    n_estimators=500,           # More trees
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
    max_depth=30,               # Deeper trees
    min_samples_leaf=1,         # Allow more specific leaves
    min_samples_split=2,
    max_features='sqrt'
)
clf_rf.fit(X_train_clf_scaled, y_train_clf)
y_pred_rf = clf_rf.predict(X_test_clf_scaled)

acc_rf = accuracy_score(y_test_clf, y_pred_rf)
f1_rf = f1_score(y_test_clf, y_pred_rf, average='weighted')
recall_rf = recall_score(y_test_clf, y_pred_rf, average='macro')
print(f"    [OK] Accuracy: {acc_rf:.4f}")
print(f"    [OK] Weighted F1: {f1_rf:.4f}")
print(f"    [OK] Macro Recall: {recall_rf:.4f}")
print("\n    Classification Report:")
print(classification_report(y_test_clf, y_pred_rf))

# Save model with scaler
with open('../trained_models/esi_random_forest.pkl', 'wb') as f:
    pickle.dump({'model': clf_rf, 'scaler': scaler_clf}, f)
print(f"    [OK] Model saved to ../trained_models/esi_random_forest.pkl")

# 4.5: Gradient Boosting Classifier - tuned
print("\n  [4.5] Gradient Boosting Classifier (tuned)...")
clf_gb = GradientBoostingClassifier(
    n_estimators=300,           # More iterations
    max_depth=8,                # Deeper trees
    learning_rate=0.1,
    min_samples_leaf=2,
    min_samples_split=4,
    random_state=42
)
clf_gb.fit(X_train_clf_scaled, y_train_clf)
y_pred_gb = clf_gb.predict(X_test_clf_scaled)

acc_gb = accuracy_score(y_test_clf, y_pred_gb)
f1_gb = f1_score(y_test_clf, y_pred_gb, average='weighted')
recall_gb = recall_score(y_test_clf, y_pred_gb, average='macro')
print(f"    [OK] Accuracy: {acc_gb:.4f}")
print(f"    [OK] Weighted F1: {f1_gb:.4f}")
print(f"    [OK] Macro Recall: {recall_gb:.4f}")
print("\n    Classification Report:")
print(classification_report(y_test_clf, y_pred_gb))

# Save model with scaler
with open('../trained_models/esi_gradient_boosting.pkl', 'wb') as f:
    pickle.dump({'model': clf_gb, 'scaler': scaler_clf}, f)
print(f"    [OK] Model saved to ../trained_models/esi_gradient_boosting.pkl")

# ============================================================================
# STEP 5: Prepare Features for Regression (Wait Time Prediction)
# ============================================================================
print("\n[STEP 5] Preparing features for wait time regression...")

# Filter encounters with valid wait times (exclude LWBS)
df_regression = df[df['wait_time_minutes'].notna()].copy()
df_regression = df_regression[df_regression['wait_time_minutes'] > 0].copy()

regression_features = [
    'esi_level', 'patient_age', 'sex_at_birth', 'arrival_mode',
    'heart_rate', 'bp_systolic', 'respiratory_rate', 'temperature_c', 'o2_saturation',
    'arrival_hour', 'is_weekend'
]

df_reg_clean = df_regression[regression_features + ['wait_time_minutes']].copy()
df_reg_clean = df_reg_clean.dropna()

print(f"  [OK] Regression dataset: {len(df_reg_clean)} samples")

# One-hot encode categorical variables
df_reg_encoded = pd.get_dummies(df_reg_clean,
                                columns=['sex_at_birth', 'arrival_mode'],
                                drop_first=True)

# Separate features and target
X_reg = df_reg_encoded.drop('wait_time_minutes', axis=1)
y_reg = df_reg_encoded['wait_time_minutes']

print(f"  [OK] Feature matrix shape: {X_reg.shape}")
print(f"  [OK] Target distribution: mean={y_reg.mean():.1f}, std={y_reg.std():.1f}")

# Split into train/test
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42
)

print(f"  [OK] Training set: {len(X_train_reg)} samples")
print(f"  [OK] Test set: {len(X_test_reg)} samples")

# ============================================================================
# STEP 6: Train Regression Models
# ============================================================================
print("\n[STEP 6] Training wait time regression model...")

# Standardize features
scaler = StandardScaler()
X_train_reg_scaled = scaler.fit_transform(X_train_reg)
X_test_reg_scaled = scaler.transform(X_test_reg)

# Train linear regression
reg_model = LinearRegression()
reg_model.fit(X_train_reg_scaled, y_train_reg)
y_pred_reg = reg_model.predict(X_test_reg_scaled)

# Evaluate
r2 = r2_score(y_test_reg, y_pred_reg)
rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))
mae = mean_absolute_error(y_test_reg, y_pred_reg)

print(f"    [OK] R² Score: {r2:.4f}")
print(f"    [OK] RMSE: {rmse:.2f} minutes")
print(f"    [OK] MAE: {mae:.2f} minutes")

# Train with statsmodels for detailed output
X_train_reg_const = sm.add_constant(X_train_reg_scaled)
ols_model = sm.OLS(y_train_reg, X_train_reg_const).fit()
print("\n    Statsmodels Summary:")
print(ols_model.summary())

# Save model and scaler
with open('../trained_models/wait_time_predictor.pkl', 'wb') as f:
    pickle.dump({'model': reg_model, 'scaler': scaler, 'feature_names': X_reg.columns.tolist()}, f)
print(f"\n    [OK] Model saved to ../trained_models/wait_time_predictor.pkl")

# ============================================================================
# STEP 7: Train Poisson GLM for Volume Prediction
# ============================================================================
print("\n[STEP 7] Training Poisson GLM for patient volume prediction...")

# Aggregate arrivals by hour
df['arrival_date_hour'] = df['arrival_ts'].dt.floor('H')
volume_by_hour = df.groupby('arrival_date_hour').size().reset_index(name='patient_count')

# Extract temporal features
volume_by_hour['hour'] = volume_by_hour['arrival_date_hour'].dt.hour
volume_by_hour['day_of_week'] = volume_by_hour['arrival_date_hour'].dt.dayofweek
volume_by_hour['month'] = volume_by_hour['arrival_date_hour'].dt.month
volume_by_hour['is_weekend'] = volume_by_hour['day_of_week'].isin([5, 6]).astype(int)

# Prepare features for Poisson GLM
X_poisson = volume_by_hour[['hour', 'day_of_week', 'month', 'is_weekend']]
y_poisson = volume_by_hour['patient_count']

print(f"  [OK] Volume dataset: {len(volume_by_hour)} hourly observations")
print(f"  [OK] Mean patients per hour: {y_poisson.mean():.2f}")

# Split train/test
X_train_poisson, X_test_poisson, y_train_poisson, y_test_poisson = train_test_split(
    X_poisson, y_poisson, test_size=0.3, random_state=42
)

# Train Poisson GLM
X_train_poisson_const = sm.add_constant(X_train_poisson)
poisson_model = sm.GLM(y_train_poisson, X_train_poisson_const, family=Poisson()).fit()

print("\n    Poisson GLM Summary:")
print(poisson_model.summary())

# Evaluate
X_test_poisson_const = sm.add_constant(X_test_poisson)
y_pred_poisson = poisson_model.predict(X_test_poisson_const)

poisson_rmse = np.sqrt(mean_squared_error(y_test_poisson, y_pred_poisson))
poisson_mae = mean_absolute_error(y_test_poisson, y_pred_poisson)

print(f"\n    [OK] RMSE: {poisson_rmse:.2f} patients/hour")
print(f"    [OK] MAE: {poisson_mae:.2f} patients/hour")
print(f"    [OK] AIC: {poisson_model.aic:.2f}")
print(f"    [OK] BIC: {poisson_model.bic:.2f}")

# Save model
with open('../trained_models/volume_predictor.pkl', 'wb') as f:
    pickle.dump(poisson_model, f)
print(f"    [OK] Model saved to ../trained_models/volume_predictor.pkl")

# ============================================================================
# STEP 8: Summary
# ============================================================================
print("\n" + "="*80)
print("TRAINING COMPLETE - MODEL SUMMARY")
print("="*80)
print("\nClassification Models (ESI Prediction):")
print(f"  • Logistic Regression:  Accuracy = {acc_logistic:.4f}, Weighted F1 = {f1_logistic:.4f}, Macro Recall = {recall_logistic:.4f}")
print(f"  • LDA (SMOTE):          Accuracy = {acc_lda:.4f}, Weighted F1 = {f1_lda:.4f}, Macro Recall = {recall_lda:.4f}")
print(f"  • Naive Bayes (SMOTE):  Accuracy = {acc_nb:.4f}, Weighted F1 = {f1_nb:.4f}, Macro Recall = {recall_nb:.4f}")
print(f"  • Random Forest:        Accuracy = {acc_rf:.4f}, Weighted F1 = {f1_rf:.4f}, Macro Recall = {recall_rf:.4f}")
print(f"  • Gradient Boosting:    Accuracy = {acc_gb:.4f}, Weighted F1 = {f1_gb:.4f}, Macro Recall = {recall_gb:.4f}")

print("\nRegression Models:")
print(f"  • Wait Time Prediction: R² = {r2:.4f}, RMSE = {rmse:.2f} min")
print(f"  • Volume Prediction:    RMSE = {poisson_rmse:.2f} patients/hour")

print("\nSaved Models:")
print("  [OK] ../trained_models/esi_logistic.pkl")
print("  [OK] ../trained_models/esi_lda.pkl")
print("  [OK] ../trained_models/esi_naive_bayes.pkl")
print("  [OK] ../trained_models/esi_random_forest.pkl")
print("  [OK] ../trained_models/esi_gradient_boosting.pkl")
print("  [OK] ../trained_models/wait_time_predictor.pkl")
print("  [OK] ../trained_models/volume_predictor.pkl")
print("\n" + "="*80)
