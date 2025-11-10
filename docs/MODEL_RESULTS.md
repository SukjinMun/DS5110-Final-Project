# Emergency Department Model Training Results

**Author:** Suk Jin Mun
**Date:** November 10, 2025
**Course:** DS 5110, Fall 2025

## Executive Summary

Successfully trained and evaluated 5 statistical models using corrected Emergency Department dataset:
- 3 classification models for ESI level prediction
- 2 regression models for wait time and volume forecasting

**Dataset:** 8,000 encounters, 4,000 patients, 12,627 vitals, 13,067 diagnoses
**Training Duration:** ~7 seconds
**All Models:** Saved to `trained_models/` directory

---

## 1. Classification Models (ESI Level Prediction)

### Objective
Predict Emergency Severity Index (ESI) level (1-5) from patient demographics, vital signs, and arrival characteristics.

### Dataset Characteristics
- **Total Samples:** 6,397 encounters (after removing missing data)
- **Features:** 27 (after one-hot encoding)
- **Train/Test Split:** 70/30 (4,477 train, 1,920 test)
- **Class Distribution:**
  ```
  ESI Level 1:  120 samples (1.9%) - Most critical
  ESI Level 2:  867 samples (13.6%)
  ESI Level 3: 3,508 samples (54.8%) - Majority class
  ESI Level 4: 1,406 samples (22.0%)
  ESI Level 5:  496 samples (7.8%) - Least urgent
  ```

### Model 1: Logistic Regression (Multinomial)

**Performance:**
- **Accuracy:** 54.84%
- **5-Fold CV Accuracy:** 54.84% (±0.03%)

**Classification Report:**
```
              precision    recall  f1-score   support
ESI Level 1       0.00      0.00      0.00        36
ESI Level 2       0.00      0.00      0.00       260
ESI Level 3       0.55      1.00      0.71      1053
ESI Level 4       0.00      0.00      0.00       422
ESI Level 5       0.00      0.00      0.00       149

    accuracy                        0.55      1920
   macro avg      0.11      0.20      0.14      1920
weighted avg      0.30      0.55      0.39      1920
```

**Analysis:**
- Model predicts only ESI level 3 (majority class)
- **Critical issue:** Cannot detect level 1 (most urgent) patients
- Severe class imbalance problem
- Not suitable for clinical deployment without improvement

**Saved Model:** `trained_models/esi_logistic.pkl`

### Model 2: Linear Discriminant Analysis (LDA)

**Performance:**
- **Accuracy:** 54.84% (identical to logistic regression)

**Analysis:**
- Same behavior as logistic regression (predicts only level 3)
- Class imbalance affects LDA assumptions (multivariate normality)
- Requires same improvements as logistic regression

**Saved Model:** `trained_models/esi_lda.pkl`

### Model 3: Gaussian Naive Bayes

**Performance:**
- **Accuracy:** 46.98% (lowest of three models)

**Classification Report:**
```
              precision    recall  f1-score   support
ESI Level 1       0.00      0.00      0.00        36
ESI Level 2       0.18      0.11      0.14       260
ESI Level 3       0.55      0.80      0.65      1053
ESI Level 4       0.18      0.06      0.09       422
ESI Level 5       0.07      0.04      0.05       149

    accuracy                        0.47      1920
   macro avg      0.20      0.20      0.19      1920
weighted avg      0.37      0.47      0.40      1920
```

**Analysis:**
- Shows some prediction diversity (attempts all classes)
- Still poor performance on minority classes
- Slightly better than random for levels 2, 4, 5
- Independence assumption likely violated (vitals are correlated)

**Saved Model:** `trained_models/esi_naive_bayes.pkl`

### Classification Models: Recommendations

**Critical Issues:**
1. **Class Imbalance:** 55% samples are ESI level 3
2. **Rare Events:** Level 1 (1.9%) and Level 5 (7.8%) underrepresented
3. **Clinical Safety:** Cannot miss level 1 patients (life-threatening)

**Improvement Strategies:**
1. **Resampling:**
   - SMOTE (Synthetic Minority Over-sampling Technique)
   - Class weights: Penalize misclassification of rare classes
   - Stratified sampling in cross-validation

2. **Feature Engineering:**
   - Interaction terms: age × vital signs
   - Chief complaint text embeddings
   - Time-based features: arrival hour, day of week effects
   - Historical patient data if available

3. **Alternative Models:**
   - **Random Forest:** Better handles imbalance, feature interactions
   - **XGBoost/LightGBM:** State-of-art for tabular data
   - **Ensemble:** Combine predictions from all models
   - **Cost-sensitive learning:** Higher cost for missing level 1/2

4. **Evaluation Metrics:**
   - **Sensitivity for Level 1:** Must be >95% (clinical requirement)
   - **Macro-averaged F1:** Better for imbalanced data
   - **Cohen's Kappa:** Agreement beyond chance

---

## 2. Regression Model: Wait Time Prediction

### Objective
Predict wait time (minutes from arrival to provider) based on patient characteristics and ED conditions.

### Dataset Characteristics
- **Total Samples:** 6,271 encounters (excluded LWBS cases)
- **Features:** 12 (after encoding)
  - ESI level, patient age, sex, arrival mode
  - Vital signs: heart rate, BP systolic, respiratory rate, temperature, O2 saturation
  - Temporal: arrival hour, weekend indicator
- **Train/Test Split:** 70/30 (4,389 train, 1,882 test)
- **Target Distribution:**
  - Mean: 78.1 minutes
  - Std Dev: 32.7 minutes
  - Range: ~10-200 minutes (estimated)

### Performance Metrics

**Test Set Performance:**
- **R² Score:** 0.8146 (explains 81.46% of variance)
- **RMSE:** 14.05 minutes
- **MAE:** 11.19 minutes
- **Relative Error:** ~14% (14.05 / 78.1)

**Training Set Performance (from Statsmodels):**
- **Adjusted R²:** 0.826
- **F-statistic:** 1738 (p < 0.001) - Highly significant model
- **Log-Likelihood:** -17682
- **AIC:** 35,390
- **BIC:** 35,470

### Model Diagnostics

**Residual Analysis (from Statsmodels):**
- **Durbin-Watson:** 2.033 (no autocorrelation, ideal ~2.0)
- **Jarque-Bera:** 34.4 (p < 0.001) - Slight non-normality
- **Skewness:** 0.215 (nearly symmetric)
- **Kurtosis:** 3.056 (close to normal = 3.0)
- **Condition Number:** 1.49 (no multicollinearity)

### Feature Importance (OLS Coefficients)

**From Statsmodels OLS Summary:**

| Feature | Coefficient | Std Error | t-value | p-value | Significance |
|---------|-------------|-----------|---------|---------|--------------|
| const   | 78.42       | 0.206     | 381.5   | 0.000   | *** |
| **x1 (ESI Level)** | **29.66** | **0.206** | **144.1** | **0.000** | *** |
| x2      | -0.10       | 0.206     | -0.48   | 0.629   | - |
| x3      | 0.41        | 0.206     | 1.99    | 0.047   | * |
| x4      | 0.31        | 0.206     | 1.52    | 0.130   | - |
| x5-x12  | -0.19 to 0.16 | 0.206-0.222 | <2.0  | >0.05   | - |

**Key Findings:**
- **ESI Level (x1):** Dominant predictor (coef = 29.66, p < 0.001)
  - Each ESI level increase → ~30 min longer wait time
  - Validates clinical triage: higher acuity (lower ESI) = shorter wait
- **Feature x3:** Marginally significant (p = 0.047)
- **Other features:** Not statistically significant after standardization
- **Intercept:** 78.42 min (baseline wait time)

### Clinical Interpretation

**Model Performance Assessment:**
- **RMSE of 14 minutes:** Acceptable for ED planning
  - Provides useful estimates for patient communication
  - Error is ~18% of mean wait time
- **Strong ESI correlation:** Confirms triage process works
  - ESI 1 patients: ~0-15 min wait (immediate)
  - ESI 5 patients: ~150+ min wait (low urgency)

**Practical Applications:**
1. **Patient Communication:** Inform patients of expected wait at triage
2. **Resource Planning:** Predict peak demand periods
3. **Quality Metrics:** Identify deviations from expected wait times
4. **Triage Validation:** Verify ESI assignments align with actual urgency

**Limitations:**
- Does not capture ED occupancy/volume (data not included)
- Missing staff availability metrics
- No consideration of procedure complexity
- Point estimate only (no confidence intervals provided to patients)

**Saved Model:** `trained_models/wait_time_predictor.pkl`
**Includes:** model, scaler, feature_names

---

## 3. Regression Model: Patient Volume Prediction (Poisson GLM)

### Objective
Forecast patient arrival volumes by hour using temporal features.

### Dataset Characteristics
- **Total Observations:** 5,078 hourly periods
- **Train/Test Split:** 70/30 (3,554 train, 1,524 test)
- **Features:** 4 temporal predictors
  - Hour of day (0-23)
  - Day of week (0-6, Mon=0)
  - Month (1-12)
  - Weekend indicator (binary)
- **Target Distribution:**
  - Mean: 1.58 patients/hour
  - Count data (0, 1, 2, ...)

### Performance Metrics

**Test Set Performance:**
- **RMSE:** 0.84 patients/hour
- **MAE:** 0.66 patients/hour
- **Relative Error:** ~53% (0.84 / 1.58)

**Model Statistics:**
- **Deviance:** 1257.7
- **Pearson χ²:** 1440 (slight overdispersion)
- **AIC:** 9448.35
- **BIC:** -27758.30 (Note: Negative BIC due to statsmodels deviance-based calculation)
- **Pseudo R² (CS):** 0.0266 (low explanatory power)

### Poisson GLM Coefficients

**From Statsmodels GLM Summary:**

| Feature | Coefficient | Std Error | z-value | p-value | Significance |
|---------|-------------|-----------|---------|---------|--------------|
| const   | 0.257       | 0.046     | 5.55    | 0.000   | *** |
| **hour** | **0.009** | **0.002** | **4.32** | **0.000** | *** |
| day_of_week | -0.004  | 0.012     | -0.34   | 0.732   | - |
| month   | 0.00005     | 0.004     | 0.01    | 0.989   | - |
| **is_weekend** | **0.256** | **0.051** | **5.03** | **0.000** | *** |

**Key Findings:**
- **Hour of day:** Significant positive effect (exp(0.009) = 1.009)
  - Each hour later → ~0.9% increase in arrival rate
  - Captures daily peak patterns (likely afternoon/evening)
- **Weekend indicator:** Significant positive effect (exp(0.256) = 1.29)
  - Weekends have 29% higher arrival rate
  - Consistent with ED utilization patterns
- **Day of week, Month:** Not significant
  - May need finer categorization (e.g., Friday nights vs. Monday mornings)

### Model Assessment

**Strengths:**
- **Captures key patterns:** Hour and weekend effects are significant
- **Appropriate distribution:** Poisson suitable for count data
- **Fast prediction:** Simple model, quick inference

**Weaknesses:**
- **High relative error:** 53% RMSE/mean ratio
- **Low R²:** Only explains 2.7% of variance
- **Overdispersion:** Pearson χ² > Deviance suggests need for negative binomial
- **Missing predictors:** Weather, holidays, local events, seasonality

**Improvement Strategies:**
1. **Alternative Distributions:**
   - Negative Binomial GLM (handles overdispersion)
   - Zero-Inflated Poisson (if many zero-count hours)

2. **Additional Features:**
   - Holiday indicators (New Year's, July 4th, etc.)
   - Flu season / respiratory illness prevalence
   - Weather conditions (temperature, precipitation)
   - Local events (sports, concerts causing injuries)
   - Seasonal decomposition (trend, seasonality, residual)

3. **Time Series Methods:**
   - ARIMA/SARIMA (incorporate temporal autocorrelation)
   - Prophet (Facebook's forecasting library)
   - Lag features (arrivals in previous 1-3 hours)

4. **Granular Time Features:**
   - Interaction: hour × weekend
   - Binary indicators for peak hours (6-10pm)
   - Day-of-week specific effects

**Saved Model:** `trained_models/volume_predictor.pkl`

---

## 4. Overall Conclusions

### What Worked Well

1. **Wait Time Regression:** Excellent performance (R² = 0.81, RMSE = 14 min)
   - Clinically useful accuracy
   - ESI level is dominant predictor (validates triage)
   - Model ready for integration into patient communication system

2. **Data Pipeline:** Successfully trained all models end-to-end
   - Feature engineering from ISO timestamps
   - Proper train/test splitting
   - Comprehensive evaluation metrics
   - Model persistence (pickle files)

3. **Technical Implementation:**
   - Statsmodels integration for statistical inference
   - Cross-validation for classification
   - Multiple model comparisons

### What Needs Improvement

1. **Classification Models:** Poor ESI prediction (54% accuracy)
   - **Critical:** Cannot detect level 1 patients (safety issue)
   - Requires class rebalancing (SMOTE, class weights)
   - Consider ensemble methods

2. **Volume Prediction:** Moderate performance (53% relative error)
   - Add external data (weather, holidays)
   - Try negative binomial for overdispersion
   - Incorporate time series methods

3. **Feature Engineering:**
   - Add ED occupancy metrics
   - Include staff-to-patient ratios
   - Incorporate historical patient patterns

### Next Steps

#### Immediate (This Week):
1. Create visualization notebooks (confusion matrices, ROC curves, residual plots)
2. Implement SMOTE for classification retraining
3. Add model prediction API endpoints to Flask
4. Document model limitations in backend README

#### Short-term (Next 2 Weeks):
1. Train Random Forest and XGBoost classifiers
2. Implement negative binomial volume model
3. Add confidence intervals to wait time predictions
4. Create interactive model dashboard

#### Long-term (Future Work):
1. Collect real ED data for validation
2. Implement online learning (models update with new data)
3. Add explainability (SHAP values, LIME)
4. Conduct clinical user testing

---

## 5. File Inventory

### Trained Models (`trained_models/`)
```
esi_logistic.pkl      (2.5 KB)  - Logistic regression classifier
esi_lda.pkl           (4.8 KB)  - Linear discriminant analysis classifier
esi_naive_bayes.pkl   (3.5 KB)  - Gaussian naive Bayes classifier
wait_time_predictor.pkl (1.5 KB)  - Linear regression + scaler + feature names
volume_predictor.pkl  (575 KB)  - Poisson GLM fitted model
```

### Training Script (`scripts/`)
```
train_models.py       - Complete pipeline: data loading → training → evaluation → saving
```

### Evaluation Materials (`notebooks/`)
```
01_model_evaluation.ipynb  - Jupyter notebook with analysis and recommendations
```

---

## References

**Course Materials:**
- DS 5110 Ch3: Linear Regression
- DS 5110 Ch4: Classification (Logistic Regression, LDA, Naive Bayes)

**Statistical Methods:**
- Scikit-learn v1.3: https://scikit-learn.org/
- Statsmodels v0.14: https://www.statsmodels.org/

**Dataset:**
- CDC NHAMCS: https://www.cdc.gov/nchs/ahcd/about_ahcd.htm
- Emergency Severity Index (ESI): https://www.ahrq.gov/patient-safety/settings/emergency-dept/esi.html

---

**Document Version:** 1.0
**Last Updated:** November 10, 2025, 16:32 EST
**Prepared by:** Suk Jin Mun (NUID: 002082427)
