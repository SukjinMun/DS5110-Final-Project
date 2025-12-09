"""
Generate Additional Model Validation Figures - DS5110 Final Project

This script generates missing figures based on DS5110 course materials:
1. Residual Analysis (Wait Time Regression)
2. Feature Importance (Random Forest & Logistic Regression)
3. Prediction Intervals (95% CI for Wait Time)
4. ESI Class Distribution (Before/After SMOTE)
5. Precision-Recall Curves (Per-class)
6. Influence Diagnostics (Cook's Distance, Leverage)
7. Model Comparison Summary Bar Chart
8. Volume Prediction Time Series
9. Data Pipeline Stages Visualization

Dataset Stages:
- Stage 1: 8,000 generated encounters
- Stage 2: 7,486 encounters after ETL (in database)
- Stage 3: 5,369 classification-ready (after dropna)

Author: Suk Jin Mun
Course: DS 5110, Fall 2025
"""

import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (confusion_matrix, classification_report, roc_curve, auc,
                             precision_recall_curve, average_precision_score,
                             accuracy_score, r2_score, mean_squared_error, mean_absolute_error)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
np.random.seed(42)

print("=" * 80)
print("GENERATING ADDITIONAL MODEL FIGURES")
print("=" * 80)

# ============================================================================
# STEP 1: Load Data from Database
# ============================================================================
print("\n[STEP 1] Loading data from database...")

db_path = '../ed_database.db'
conn = sqlite3.connect(db_path)

# Get counts at each stage
total_in_db = pd.read_sql_query("SELECT COUNT(*) as cnt FROM encounter", conn).iloc[0]['cnt']
print(f"  Stage 2 (After ETL): {total_in_db:,} encounters in database")

encounters = pd.read_sql_query("SELECT * FROM encounter", conn)
patients = pd.read_sql_query("SELECT * FROM patient", conn)
vitals = pd.read_sql_query("SELECT * FROM vitals", conn)
payors = pd.read_sql_query("SELECT * FROM encounter_payor", conn)

conn.close()

# Feature engineering
encounters['arrival_ts'] = pd.to_datetime(encounters['arrival_ts'], errors='coerce')
encounters['provider_start_ts'] = pd.to_datetime(encounters['provider_start_ts'], errors='coerce')
encounters['departure_ts'] = pd.to_datetime(encounters['departure_ts'], errors='coerce')
patients['dob'] = pd.to_datetime(patients['dob'], errors='coerce')

df = encounters.merge(patients, on='patient_id', how='left')
df['patient_age'] = ((df['arrival_ts'] - df['dob']).dt.days / 365.25)
df['patient_age'] = df['patient_age'].fillna(df['patient_age'].median()).astype(int)

vitals['taken_ts'] = pd.to_datetime(vitals['taken_ts'], errors='coerce')
first_vitals = vitals.sort_values('taken_ts').groupby('encounter_id').first().reset_index()
df = df.merge(first_vitals[['encounter_id', 'heart_rate', 'systolic_bp', 'diastolic_bp',
                             'respiratory_rate', 'temperature_c', 'spo2', 'pain_score']],
              on='encounter_id', how='left')
df = df.rename(columns={'systolic_bp': 'bp_systolic', 'diastolic_bp': 'bp_diastolic', 'spo2': 'o2_saturation'})
df = df.merge(payors[['encounter_id', 'payor_type']], on='encounter_id', how='left')

df['arrival_hour'] = df['arrival_ts'].dt.hour
df['arrival_day_of_week'] = df['arrival_ts'].dt.dayofweek
df['is_weekend'] = df['arrival_day_of_week'].isin([5, 6]).astype(int)

# Calculate wait time for regression
df['wait_time_minutes'] = (
    (df['provider_start_ts'] - df['arrival_ts']).dt.total_seconds() / 60
)

# ============================================================================
# STEP 2: Prepare Classification Data
# ============================================================================
print("\n[STEP 2] Preparing classification data...")

classification_features = ['patient_age', 'sex_at_birth', 'arrival_mode', 'chief_complaint',
    'heart_rate', 'bp_systolic', 'bp_diastolic', 'respiratory_rate',
    'temperature_c', 'o2_saturation', 'pain_score', 'arrival_hour',
    'arrival_day_of_week', 'is_weekend', 'payor_type']

# Store pre-dropna counts for ESI distribution
esi_before_dropna = df['esi_level'].value_counts().sort_index()

df_clf = df[classification_features + ['esi_level']].dropna()
print(f"  Stage 3 (Classification-ready): {len(df_clf):,} encounters after dropna()")

# Store post-dropna counts
esi_after_dropna = df_clf['esi_level'].value_counts().sort_index()

df_encoded = pd.get_dummies(df_clf, columns=['sex_at_birth', 'arrival_mode', 'chief_complaint', 'payor_type'], drop_first=True)

X = df_encoded.drop('esi_level', axis=1)
y = df_encoded['esi_level']

print(f"  Features: {X.shape[1]}")
print(f"  ESI Distribution:\n{y.value_counts().sort_index()}")

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Store pre-SMOTE distribution
y_train_before_smote = y_train.value_counts().sort_index()

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

# Store post-SMOTE distribution
y_train_after_smote = pd.Series(y_train_smote).value_counts().sort_index()

print(f"\n  Train set: {len(X_train):,} samples")
print(f"  Test set: {len(X_test):,} samples")
print(f"  After SMOTE: {len(X_train_smote):,} samples")

# ============================================================================
# STEP 3: Train Models
# ============================================================================
print("\n[STEP 3] Training models...")

models = {
    'Random Forest': RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                            random_state=42, n_jobs=1, max_depth=20),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=6, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=5000, random_state=42),
    'LDA': LinearDiscriminantAnalysis(),
    'Naive Bayes': GaussianNB()
}

trained_models = {}
cv_results = {}
test_accuracies = {}
auc_scores = {}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
X_scaled_full = scaler.fit_transform(X)
classes = sorted(y.unique())
y_test_bin = label_binarize(y_test, classes=classes)

for name, model in models.items():
    # Train on SMOTE data for Logistic, LDA, NB; original for RF, GB
    if name in ['Logistic Regression', 'LDA', 'Naive Bayes']:
        model.fit(X_train_smote, y_train_smote)
    else:
        model.fit(X_train_scaled, y_train)
    trained_models[name] = model

    # Cross-validation
    scores = cross_val_score(model, X_scaled_full, y, cv=cv, scoring='accuracy')
    cv_results[name] = scores

    # Test accuracy
    y_pred = model.predict(X_test_scaled)
    test_accuracies[name] = accuracy_score(y_test, y_pred)

    # AUC
    if hasattr(model, 'predict_proba'):
        y_score = model.predict_proba(X_test_scaled)
        auc_per_class = []
        for i in range(len(classes)):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            auc_per_class.append(auc(fpr, tpr))
        auc_scores[name] = np.mean(auc_per_class)

    print(f"  [OK] {name}: Acc={test_accuracies[name]:.4f}")

# ============================================================================
# FIGURE 1: Data Pipeline Stages
# ============================================================================
print("\n[FIGURE 1] Data Pipeline Stages...")

fig, ax = plt.subplots(figsize=(10, 6))

stages = ['Generated\n(Raw)', 'After ETL\n(Database)', 'Classification-\nReady']
counts = [8000, total_in_db, len(df_clf)]
colors = ['#3498db', '#2ecc71', '#9b59b6']

bars = ax.bar(stages, counts, color=colors, edgecolor='black', linewidth=1.5)

# Add value labels
for bar, count in zip(bars, counts):
    height = bar.get_height()
    ax.annotate(f'{count:,}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=14, fontweight='bold')

# Add retention percentages
ax.annotate(f'100%', xy=(0, counts[0]/2), ha='center', va='center', fontsize=11, color='white', fontweight='bold')
ax.annotate(f'{counts[1]/counts[0]*100:.1f}%', xy=(1, counts[1]/2), ha='center', va='center', fontsize=11, color='white', fontweight='bold')
ax.annotate(f'{counts[2]/counts[0]*100:.1f}%', xy=(2, counts[2]/2), ha='center', va='center', fontsize=11, color='white', fontweight='bold')

ax.set_ylabel('Number of Encounters', fontsize=12)
ax.set_title('Data Pipeline Stages\n(From Generation to Classification-Ready)', fontsize=14, fontweight='bold')
ax.set_ylim(0, 9000)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('../figs/data_pipeline_stages.png', dpi=150, bbox_inches='tight')
print("  Saved: ../figs/data_pipeline_stages.png")
plt.close()

# ============================================================================
# FIGURE 2: ESI Class Distribution (Before/After SMOTE)
# ============================================================================
print("\n[FIGURE 2] ESI Class Distribution...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Before SMOTE
x_pos = np.arange(5)
bars1 = axes[0].bar(x_pos, y_train_before_smote.values, color='#e74c3c', edgecolor='black', alpha=0.8)
axes[0].set_xlabel('ESI Level', fontsize=11)
axes[0].set_ylabel('Number of Samples', fontsize=11)
axes[0].set_title('Training Set: Before SMOTE\n(Imbalanced)', fontsize=12, fontweight='bold')
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(['ESI 1', 'ESI 2', 'ESI 3', 'ESI 4', 'ESI 5'])
for bar, val in zip(bars1, y_train_before_smote.values):
    axes[0].annotate(f'{val}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 3), textcoords='offset points', ha='center', fontsize=10)
axes[0].grid(True, alpha=0.3, axis='y')

# After SMOTE
bars2 = axes[1].bar(x_pos, y_train_after_smote.values, color='#27ae60', edgecolor='black', alpha=0.8)
axes[1].set_xlabel('ESI Level', fontsize=11)
axes[1].set_ylabel('Number of Samples', fontsize=11)
axes[1].set_title('Training Set: After SMOTE\n(Balanced)', fontsize=12, fontweight='bold')
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(['ESI 1', 'ESI 2', 'ESI 3', 'ESI 4', 'ESI 5'])
for bar, val in zip(bars2, y_train_after_smote.values):
    axes[1].annotate(f'{val}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 3), textcoords='offset points', ha='center', fontsize=10)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('../figs/esi_class_distribution.png', dpi=150, bbox_inches='tight')
print("  Saved: ../figs/esi_class_distribution.png")
plt.close()

# ============================================================================
# FIGURE 3: Feature Importance (Random Forest)
# ============================================================================
print("\n[FIGURE 3] Feature Importance (Random Forest)...")

rf_model = trained_models['Random Forest']
feature_names = X.columns.tolist()
importances = rf_model.feature_importances_

# Sort by importance
indices = np.argsort(importances)[::-1][:15]  # Top 15

fig, ax = plt.subplots(figsize=(10, 8))
colors = plt.cm.viridis(np.linspace(0.2, 0.8, 15))

bars = ax.barh(range(15), importances[indices][::-1], color=colors[::-1], edgecolor='black')
ax.set_yticks(range(15))
ax.set_yticklabels([feature_names[i] for i in indices[::-1]], fontsize=10)
ax.set_xlabel('Feature Importance', fontsize=11)
ax.set_title('Top 15 Feature Importances (Random Forest)\nfor ESI Level Prediction', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

# Add value labels
for bar, val in zip(bars, importances[indices][::-1]):
    ax.annotate(f'{val:.3f}', xy=(bar.get_width(), bar.get_y() + bar.get_height()/2),
                xytext=(3, 0), textcoords='offset points', ha='left', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('../figs/feature_importance_rf.png', dpi=150, bbox_inches='tight')
print("  Saved: ../figs/feature_importance_rf.png")
plt.close()

# ============================================================================
# FIGURE 4: Logistic Regression Coefficients
# ============================================================================
print("\n[FIGURE 4] Logistic Regression Coefficients...")

lr_model = trained_models['Logistic Regression']

# Get mean absolute coefficients across all classes (multiclass)
mean_coefs = np.abs(lr_model.coef_).mean(axis=0)
indices = np.argsort(mean_coefs)[::-1][:15]

fig, ax = plt.subplots(figsize=(10, 8))
colors = plt.cm.plasma(np.linspace(0.2, 0.8, 15))

bars = ax.barh(range(15), mean_coefs[indices][::-1], color=colors[::-1], edgecolor='black')
ax.set_yticks(range(15))
ax.set_yticklabels([feature_names[i] for i in indices[::-1]], fontsize=10)
ax.set_xlabel('Mean |Coefficient| (across ESI classes)', fontsize=11)
ax.set_title('Top 15 Logistic Regression Coefficients\n(Production Model for ESI Prediction)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

for bar, val in zip(bars, mean_coefs[indices][::-1]):
    ax.annotate(f'{val:.3f}', xy=(bar.get_width(), bar.get_y() + bar.get_height()/2),
                xytext=(3, 0), textcoords='offset points', ha='left', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('../figs/logistic_coefficients.png', dpi=150, bbox_inches='tight')
print("  Saved: ../figs/logistic_coefficients.png")
plt.close()

# ============================================================================
# FIGURE 5: Precision-Recall Curves
# ============================================================================
print("\n[FIGURE 5] Precision-Recall Curves...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Random Forest PR curves
rf_probs = trained_models['Random Forest'].predict_proba(X_test_scaled)
colors_esi = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#3498db']
esi_labels = ['ESI 1 (Critical)', 'ESI 2 (Emergent)', 'ESI 3 (Urgent)', 'ESI 4 (Less Urgent)', 'ESI 5 (Non-Urgent)']

for i, (esi_level, color, label) in enumerate(zip(classes, colors_esi, esi_labels)):
    precision, recall, _ = precision_recall_curve(y_test_bin[:, i], rf_probs[:, i])
    ap = average_precision_score(y_test_bin[:, i], rf_probs[:, i])
    axes[0].plot(recall, precision, color=color, linewidth=2, label=f'{label} (AP={ap:.3f})')

axes[0].set_xlabel('Recall', fontsize=11)
axes[0].set_ylabel('Precision', fontsize=11)
axes[0].set_title('Precision-Recall Curves (Random Forest)\nPer-Class Performance', fontsize=12, fontweight='bold')
axes[0].legend(loc='lower left', fontsize=9)
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim([0, 1])
axes[0].set_ylim([0, 1.05])

# Average Precision comparison across models
ap_scores = {}
for name, model in trained_models.items():
    if hasattr(model, 'predict_proba'):
        y_probs = model.predict_proba(X_test_scaled)
        ap_per_class = []
        for i in range(len(classes)):
            ap = average_precision_score(y_test_bin[:, i], y_probs[:, i])
            ap_per_class.append(ap)
        ap_scores[name] = np.mean(ap_per_class)

bars = axes[1].barh(list(ap_scores.keys()), list(ap_scores.values()), color='steelblue', edgecolor='black')
axes[1].set_xlabel('Mean Average Precision', fontsize=11)
axes[1].set_title('Model Comparison: Average Precision', fontsize=12, fontweight='bold')
axes[1].set_xlim([0.5, 1.0])
axes[1].grid(True, alpha=0.3, axis='x')

for bar, val in zip(bars, ap_scores.values()):
    axes[1].annotate(f'{val:.4f}', xy=(val - 0.05, bar.get_y() + bar.get_height()/2),
                     ha='left', va='center', color='white', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig('../figs/precision_recall_curves.png', dpi=150, bbox_inches='tight')
print("  Saved: ../figs/precision_recall_curves.png")
plt.close()

# ============================================================================
# FIGURE 6: Model Comparison Summary
# ============================================================================
print("\n[FIGURE 6] Model Comparison Summary...")

fig, ax = plt.subplots(figsize=(12, 6))

model_names = list(models.keys())
x = np.arange(len(model_names))
width = 0.25

# Metrics
accuracies = [test_accuracies[name] for name in model_names]
aucs = [auc_scores.get(name, 0) for name in model_names]
cv_means = [cv_results[name].mean() for name in model_names]

bars1 = ax.bar(x - width, accuracies, width, label='Test Accuracy', color='#3498db', edgecolor='black')
bars2 = ax.bar(x, aucs, width, label='AUC Score', color='#e74c3c', edgecolor='black')
bars3 = ax.bar(x + width, cv_means, width, label='5-Fold CV', color='#27ae60', edgecolor='black')

ax.set_ylabel('Score', fontsize=11)
ax.set_title('Model Performance Comparison\n(ESI Level Classification)', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=15, ha='right', fontsize=10)
ax.legend(loc='lower right')
ax.set_ylim([0.5, 1.05])
ax.grid(True, alpha=0.3, axis='y')

# Add value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('../figs/model_comparison_summary.png', dpi=150, bbox_inches='tight')
print("  Saved: ../figs/model_comparison_summary.png")
plt.close()

# ============================================================================
# FIGURE 7: Wait Time Regression - Residual Analysis
# ============================================================================
print("\n[FIGURE 7] Wait Time Regression - Residual Analysis...")

# Prepare regression data
df_reg = df[df['wait_time_minutes'].notna() & (df['wait_time_minutes'] > 0)].copy()
reg_features = ['esi_level', 'patient_age', 'sex_at_birth', 'arrival_mode',
                'heart_rate', 'bp_systolic', 'respiratory_rate', 'temperature_c',
                'o2_saturation', 'arrival_hour', 'is_weekend']

df_reg_clean = df_reg[reg_features + ['wait_time_minutes']].dropna()
df_reg_encoded = pd.get_dummies(df_reg_clean, columns=['sex_at_birth', 'arrival_mode'], drop_first=True)

X_reg = df_reg_encoded.drop('wait_time_minutes', axis=1)
y_reg = df_reg_encoded['wait_time_minutes']

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42)

scaler_reg = StandardScaler()
X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
X_test_reg_scaled = scaler_reg.transform(X_test_reg)

# Train regression model
reg_model = LinearRegression()
reg_model.fit(X_train_reg_scaled, y_train_reg)
y_pred_reg = reg_model.predict(X_test_reg_scaled)
residuals = y_test_reg.values - y_pred_reg

# Metrics
r2 = r2_score(y_test_reg, y_pred_reg)
rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))
mae = mean_absolute_error(y_test_reg, y_pred_reg)

print(f"  Regression: R²={r2:.4f}, RMSE={rmse:.2f}, MAE={mae:.2f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Residuals vs Fitted
axes[0].scatter(y_pred_reg, residuals, alpha=0.5, s=20, edgecolors='none', c='steelblue')
axes[0].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[0].set_xlabel('Fitted Values (minutes)', fontsize=11)
axes[0].set_ylabel('Residuals (minutes)', fontsize=11)
axes[0].set_title('Residuals vs Fitted Values\n(Wait Time Prediction)', fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Add text box with metrics
textstr = f'R² = {r2:.4f}\nRMSE = {rmse:.2f} min\nMAE = {mae:.2f} min'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
axes[0].text(0.95, 0.95, textstr, transform=axes[0].transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right', bbox=props)

# Q-Q Plot
stats.probplot(residuals, dist="norm", plot=axes[1])
axes[1].set_title('Q-Q Plot (Normality Check)\nWait Time Residuals', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../figs/regression_residual_analysis.png', dpi=150, bbox_inches='tight')
print("  Saved: ../figs/regression_residual_analysis.png")
plt.close()

# ============================================================================
# FIGURE 8: Prediction Intervals (Wait Time)
# ============================================================================
print("\n[FIGURE 8] Prediction Intervals...")

# Calculate prediction intervals
n = len(X_train_reg_scaled)
p = X_train_reg_scaled.shape[1]
dof = n - p - 1
t_val = stats.t.ppf(0.975, dof)
mse = np.sum((y_train_reg.values - reg_model.predict(X_train_reg_scaled))**2) / dof
s = np.sqrt(mse)

# Sample 50 test points for visualization
sample_idx = np.random.choice(len(y_test_reg), size=min(50, len(y_test_reg)), replace=False)
sample_idx = np.sort(sample_idx)

y_pred_sample = y_pred_reg[sample_idx]
y_actual_sample = y_test_reg.values[sample_idx]

# Simplified prediction interval (approximation)
margin = t_val * s * np.sqrt(1 + 1/n)
lower = y_pred_sample - margin
upper = y_pred_sample + margin

fig, ax = plt.subplots(figsize=(12, 6))

indices = range(len(sample_idx))
ax.scatter(indices, y_actual_sample, color='blue', s=60, label='Actual Wait Time', zorder=3, edgecolors='black')
ax.scatter(indices, y_pred_sample, color='red', s=60, label='Predicted Wait Time', zorder=3, marker='x')
ax.errorbar(indices, y_pred_sample, yerr=[y_pred_sample - lower, upper - y_pred_sample],
            fmt='none', ecolor='gray', alpha=0.5, capsize=3, label='95% Prediction Interval')

ax.set_xlabel('Sample Index', fontsize=11)
ax.set_ylabel('Wait Time (minutes)', fontsize=11)
ax.set_title('Wait Time Predictions with 95% Prediction Intervals\n(Sample of 50 Test Cases)', fontsize=12, fontweight='bold')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

# Coverage calculation
coverage = np.sum((y_actual_sample >= lower) & (y_actual_sample <= upper)) / len(y_actual_sample) * 100
ax.text(0.02, 0.98, f'Coverage: {coverage:.1f}%', transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('../figs/regression_prediction_intervals.png', dpi=150, bbox_inches='tight')
print("  Saved: ../figs/regression_prediction_intervals.png")
plt.close()

# ============================================================================
# FIGURE 9: Influence Diagnostics (Cook's Distance & Leverage)
# ============================================================================
print("\n[FIGURE 9] Influence Diagnostics...")

# Use training data for influence analysis
X_with_intercept = np.column_stack([np.ones(len(X_train_reg_scaled)), X_train_reg_scaled])
try:
    H = X_with_intercept @ np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T
    leverage = np.diag(H)
except:
    # If singular, use pseudo-inverse
    H = X_with_intercept @ np.linalg.pinv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T
    leverage = np.diag(H)

train_residuals = y_train_reg.values - reg_model.predict(X_train_reg_scaled)
residual_std = train_residuals / train_residuals.std()

# Cook's distance
p_features = X_train_reg_scaled.shape[1]
cooks_d = (residual_std**2 / (p_features + 1)) * (leverage / np.maximum((1 - leverage)**2, 1e-10))

leverage_threshold = 2 * (p_features + 1) / len(X_train_reg_scaled)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Leverage plot
axes[0].scatter(range(len(leverage)), leverage, alpha=0.5, s=20, c='steelblue')
axes[0].axhline(y=leverage_threshold, color='red', linestyle='--', linewidth=2,
                label=f'Threshold = {leverage_threshold:.4f}')
axes[0].set_xlabel('Observation Index', fontsize=11)
axes[0].set_ylabel('Leverage', fontsize=11)
axes[0].set_title('Leverage Values\n(High-Leverage Point Detection)', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

high_leverage_count = np.sum(leverage > leverage_threshold)
axes[0].text(0.02, 0.98, f'High leverage points: {high_leverage_count}',
             transform=axes[0].transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Cook's distance plot
axes[1].scatter(range(len(cooks_d)), cooks_d, alpha=0.5, s=20, c='coral')
axes[1].axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Threshold = 0.5')
axes[1].axhline(y=4/len(cooks_d), color='orange', linestyle=':', linewidth=2,
                label=f'4/n = {4/len(cooks_d):.4f}')
axes[1].set_xlabel('Observation Index', fontsize=11)
axes[1].set_ylabel("Cook's Distance", fontsize=11)
axes[1].set_title("Cook's Distance\n(Influential Point Detection)", fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

influential_count = np.sum(cooks_d > 0.5)
axes[1].text(0.02, 0.98, f'Influential points (D > 0.5): {influential_count}',
             transform=axes[1].transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('../figs/influence_diagnostics.png', dpi=150, bbox_inches='tight')
print("  Saved: ../figs/influence_diagnostics.png")
plt.close()

# ============================================================================
# FIGURE 10: Volume Prediction Time Series
# ============================================================================
print("\n[FIGURE 10] Volume Prediction Time Series...")

# Aggregate by hour
df['arrival_date_hour'] = df['arrival_ts'].dt.floor('H')
volume_by_hour = df.groupby('arrival_date_hour').size().reset_index(name='patient_count')
volume_by_hour = volume_by_hour.dropna()

# Extract features
volume_by_hour['hour'] = volume_by_hour['arrival_date_hour'].dt.hour
volume_by_hour['day_of_week'] = volume_by_hour['arrival_date_hour'].dt.dayofweek
volume_by_hour['month'] = volume_by_hour['arrival_date_hour'].dt.month
volume_by_hour['is_weekend'] = volume_by_hour['day_of_week'].isin([5, 6]).astype(int)

# Simple linear model for volume (simplified from Poisson GLM)
X_vol = volume_by_hour[['hour', 'day_of_week', 'month', 'is_weekend']]
y_vol = volume_by_hour['patient_count']

vol_model = LinearRegression()
vol_model.fit(X_vol, y_vol)
y_vol_pred = vol_model.predict(X_vol)

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Time series plot (last 7 days of data)
last_week = volume_by_hour.tail(168)  # 7 days * 24 hours
if len(last_week) < 168:
    last_week = volume_by_hour.tail(min(168, len(volume_by_hour)))

axes[0].plot(range(len(last_week)), last_week['patient_count'].values,
             'b-', linewidth=1, label='Actual', alpha=0.7)
axes[0].plot(range(len(last_week)), vol_model.predict(last_week[['hour', 'day_of_week', 'month', 'is_weekend']]),
             'r--', linewidth=2, label='Predicted')
axes[0].set_xlabel('Hours (Last Week of Data)', fontsize=11)
axes[0].set_ylabel('Patient Arrivals', fontsize=11)
axes[0].set_title('Patient Volume: Actual vs Predicted\n(Hourly Arrivals)', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Hourly pattern (average by hour)
hourly_avg = volume_by_hour.groupby('hour')['patient_count'].mean()
hourly_std = volume_by_hour.groupby('hour')['patient_count'].std()

axes[1].bar(hourly_avg.index, hourly_avg.values, color='steelblue', edgecolor='black', alpha=0.7)
axes[1].errorbar(hourly_avg.index, hourly_avg.values, yerr=hourly_std.values,
                 fmt='none', color='black', capsize=3)
axes[1].set_xlabel('Hour of Day', fontsize=11)
axes[1].set_ylabel('Average Patient Arrivals', fontsize=11)
axes[1].set_title('Average Hourly Patient Volume Pattern\n(Mean +/- Std Dev)', fontsize=12, fontweight='bold')
axes[1].set_xticks(range(24))
axes[1].grid(True, alpha=0.3, axis='y')

# Volume prediction metrics
vol_rmse = np.sqrt(mean_squared_error(y_vol, y_vol_pred))
vol_mae = mean_absolute_error(y_vol, y_vol_pred)
axes[0].text(0.02, 0.98, f'RMSE = {vol_rmse:.2f} patients/hr\nMAE = {vol_mae:.2f} patients/hr',
             transform=axes[0].transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('../figs/volume_prediction_timeseries.png', dpi=150, bbox_inches='tight')
print("  Saved: ../figs/volume_prediction_timeseries.png")
plt.close()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("FIGURE GENERATION COMPLETE")
print("=" * 80)

print("\nGenerated figures in ../figs/:")
generated_figs = [
    "data_pipeline_stages.png",
    "esi_class_distribution.png",
    "feature_importance_rf.png",
    "logistic_coefficients.png",
    "precision_recall_curves.png",
    "model_comparison_summary.png",
    "regression_residual_analysis.png",
    "regression_prediction_intervals.png",
    "influence_diagnostics.png",
    "volume_prediction_timeseries.png"
]

for fig in generated_figs:
    print(f"  [OK] {fig}")

print("\nExisting figures (kept):")
existing_figs = [
    "validation_confusion_matrices.png",
    "validation_roc_curves.png",
    "validation_learning_curves.png"
]
for fig in existing_figs:
    print(f"  [OK] {fig}")

print("\n" + "=" * 80)
print(f"Total figures available: {len(generated_figs) + len(existing_figs)}")
print("=" * 80)
