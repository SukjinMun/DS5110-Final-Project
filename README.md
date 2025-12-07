<div align="center">

# Emergency Department Database and Analysis System

**Course:** DS 5110: Introduction to Data Management and Processing, Fall 2025

**Team 22:** Suk Jin Mun, Xiaobai Li, Shaobo (Ben) Chen

</div>

## Overview

This project analyzes Emergency Department triage classification and patient flow using a normalized relational database and statistical modeling. Core focus: database design, statistical analysis, data visualization, and Flask web application development.

## Objectives

**Core (Required):**
1. Design normalized database (3NF) for patient visits, triage, treatments, wait times
2. Develop statistical models for patient urgency classification and wait time prediction
3. Create data visualizations for hospital administrators
4. Provide actionable recommendations for resource allocation
5. Build Flask web application with interactive dashboard

## Project Structure

```
DS5110-Final-Project/
├── backend/                    # Flask API server
│   ├── app.py                 # Main application entry point
│   ├── config/                # Database configuration
│   ├── models/                # ORM models and ML model classes
│   └── routes/                # API endpoint definitions
│       ├── api.py            # Data query endpoints
│       └── predictions.py    # ML prediction endpoints
├── frontend/                   # React + TypeScript web application
│   ├── src/
│   │   ├── pages/            # Dashboard, Encounters, Predictions, Staff
│   │   ├── lib/api.ts        # API client
│   │   └── types/            # TypeScript definitions
│   └── package.json
├── database/                   # SQL schema and setup scripts
│   ├── db_setup.sql          # Schema creation
│   └── db_import.sql         # Data import queries
├── dataset/                    # Generated ED data (CSV files)
│   ├── encounter.csv         # Patient encounters
│   ├── patient.csv           # Patient demographics
│   ├── vitals.csv            # Vital signs
│   ├── diagnosis.csv         # Diagnoses (ICD-10)
│   ├── staff.csv             # Staff information
│   └── generate_ed_csvs.py   # Data generation script
├── trained_models/             # Serialized ML models (.pkl)
│   ├── esi_random_forest.pkl    # ESI classification (Random Forest) - BEST
│   ├── esi_gradient_boosting.pkl # ESI classification (Gradient Boosting)
│   ├── esi_logistic.pkl         # ESI classification (Logistic Regression)
│   ├── esi_lda.pkl              # ESI classification (LDA)
│   ├── esi_naive_bayes.pkl      # ESI classification (Naive Bayes)
│   ├── wait_time_predictor.pkl  # Wait time regression
│   └── volume_predictor.pkl     # Patient volume (Poisson GLM)
├── notebooks/                  # Jupyter notebooks for analysis
│   └── 01_model_evaluation.ipynb
├── scripts/                    # Training and testing scripts
│   ├── train_models.py
│   └── test_models.py
├── docs/                       # Documentation
└── project_progress_tracker.xlsx
```

## Installation

### Prerequisites
- Python 3.9+
- Node.js 18+ (for frontend)

### Backend Setup

```bash
# Install Python dependencies
pip install flask flask-cors sqlalchemy pandas numpy scikit-learn statsmodels

# Run backend server (port 5001)
cd backend
python app.py
```

### Frontend Setup

```bash
# Install Node dependencies
cd frontend
npm install

# Run development server (port 5173)
npm run dev
```

### Database Setup

The database is auto-generated from CSV files on first run. CSV data files are in `dataset/`.

## API Endpoints

### Data Endpoints (`/api`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/encounters` | GET | List encounters (with pagination) |
| `/api/encounters/<id>` | GET | Encounter details |
| `/api/patients/<id>` | GET | Patient info and history |
| `/api/staff` | GET | Staff list |
| `/api/statistics/overview` | GET | ED overview statistics |
| `/api/statistics/esi` | GET | ESI level statistics |
| `/api/statistics/vitals` | GET | Vital signs statistics |
| `/api/statistics/payor` | GET | Payor distribution |
| `/api/statistics/diagnoses` | GET | Top diagnoses |

### Prediction Endpoints (`/api/predictions`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/predictions/models/info` | GET | Model information |
| `/api/predictions/esi` | POST | Predict ESI level |
| `/api/predictions/wait-time` | POST | Predict wait time |
| `/api/predictions/volume` | GET | Predict patient volume |

## Statistical Models

### Classification Models (ESI Prediction)

| Model | Accuracy | AUC | 5-Fold CV |
|-------|----------|-----|-----------|
| **Logistic Regression** | **85.66%** | 0.9756 | 84.75% ±0.83% |
| Random Forest | 85.47% | 0.9764 | 85.08% ±0.92% |
| Gradient Boosting | 84.30% | 0.9696 | 84.11% ±1.10% |
| LDA | 83.80% | 0.9712 | 82.96% ±0.53% |
| Naive Bayes | 60.58% | 0.9049 | 57.70% ±10.02% |

*Metrics from DS5110 class (Ch4, Week 6) and cited literature. Model accuracy (~84-86%) aligns with published ML studies on real ESI data (70-80%). Dataset uses 30% nurse variability to match real-world triage disagreement rates.*

### Regression Models

| Model | Metric | Value |
|-------|--------|-------|
| Wait Time (Linear Regression) | R² | 0.8570 |
| Wait Time (Linear Regression) | RMSE | 14.17 min |
| Volume (Poisson GLM) | RMSE | 0.86 patients/hour |

## Methodology

### Data Preprocessing
- **SMOTE (Synthetic Minority Over-sampling Technique)**: Applied to handle class imbalance in ESI levels (Level 3 is 55% majority class)
- **StandardScaler**: Normalized all numerical features for model training
- **One-hot encoding**: Converted categorical variables (sex, arrival mode, chief complaint, payor type)

### Feature Engineering
- Calculated wait times (arrival to provider start) and length of stay metrics
- Extracted temporal features: hour, day of week, month, weekend indicator
- Merged patient demographics, first vital signs per encounter, and payor information
- Created 31 engineered features total from 7,486 encounters

### Model Selection & Training
- **Classification Models**: Trained 5 models (Logistic Regression, Random Forest, Gradient Boosting, LDA, Naive Bayes) for ESI prediction
- **Regression Models**: Linear Regression for wait time prediction, Poisson GLM for volume forecasting
- **Cross-Validation**: 5-fold CV used to ensure generalization (all models show <1% standard deviation)
- **Train/Test Split**: 80/20 split with stratification on ESI levels

## Results & Model Performance

### ESI Classification Results

**Best Model: Logistic Regression (85.66% Accuracy)**
- Test Accuracy: 85.66%
- AUC Score: 0.9756
- 5-Fold CV Accuracy: 84.75% ± 0.83%

**Model Comparison:**

| Model | Test Accuracy | AUC | CV Accuracy (mean ± std) |
|-------|---------------|-----|--------------------------|
| Logistic Regression | 85.66% | 0.9756 | 84.75% ± 0.83% |
| Random Forest | 85.47% | 0.9764 | 85.08% ± 0.92% |
| Gradient Boosting | 84.30% | 0.9696 | 84.11% ± 1.10% |
| LDA | 83.80% | 0.9712 | 82.96% ± 0.53% |
| Naive Bayes | 60.58% | 0.9049 | 57.70% ± 10.02% |

**Confusion Matrix (Test Set):**
- Overall accuracy: 88.7%
- ESI Level 2 Recall: 94.4% (critical for patient safety - ensures high-acuity patients are correctly identified)
- Error pattern: Most errors occur between adjacent ESI levels, which is clinically realistic

### Wait Time Prediction Results
- R² Score: 0.857 (highly effective)
- RMSE: 14.17 minutes
- MAE: 11.32 minutes
- Key Finding: Each ESI level increase correlates with 40 minutes longer wait time, validating that urgent patients are seen faster

### Volume Forecasting Results
- RMSE: 0.86 patients/hour
- MAE: 0.67 patients/hour
- Key Findings:
  - Weekends show 29% higher patient volume
  - Evening hours (18:00-22:00) show peak arrivals
  - Useful for shift planning and resource allocation

### Validation Against Published Research
Our results (85% accuracy) align with published benchmarks:
- **Ivanov et al. (2021)**: KATE algorithm achieved 75.7% accuracy on 166,000 real ED cases
- **Levin et al. (2018)**: ML-based triage showed 70-80% accuracy on clinical data
- Our model includes 30% nurse variability matching real-world clinical practice

## Demo & Usage

### Running the Application

1. **Start Backend API** (Terminal 1):
   ```bash
   cd backend
   python app.py
   # API running at http://localhost:5001
   ```

2. **Start Frontend** (Terminal 2):
   ```bash
   cd frontend
   npm run dev
   # UI running at http://localhost:5173
   ```

3. **Open Browser**: Navigate to `http://localhost:5173`

### Training Models (Optional)

To retrain models from scratch:

```bash
# Train all models (ESI classification, wait time, volume forecasting)
cd scripts
python train_models.py

# This will:
# 1. Load data from ed_database.db
# 2. Engineer features
# 3. Train 5 classification models + 2 regression models
# 4. Save models to trained_models/ directory
# 5. Generate performance metrics and visualizations
```

### Model Evaluation Notebook

```bash
# View detailed model evaluation
cd notebooks
jupyter notebook 01_model_evaluation.ipynb
```

### Example API Calls

**Predict ESI Level:**
```bash
curl http://localhost:5001/api/predictions/esi -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "model": "random_forest",
    "features": {
      "patient_age": 45,
      "sex_at_birth": "M",
      "arrival_mode": "Walk-in",
      "chief_complaint": "Chest pain",
      "heart_rate": 95,
      "bp_systolic": 140,
      "bp_diastolic": 85,
      "respiratory_rate": 18,
      "temperature_c": 37.0,
      "o2_saturation": 97,
      "pain_score": 7,
      "arrival_hour": 14,
      "arrival_day_of_week": 3,
      "is_weekend": 0,
      "payor_type": "private"
    }
  }'
```

**Predict Wait Time:**
```bash
curl http://localhost:5001/api/predictions/wait-time -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "esi_level": 3,
      "patient_age": 45,
      "sex_at_birth": "M",
      "arrival_mode": "Walk-in",
      "heart_rate": 88,
      "bp_systolic": 130,
      "respiratory_rate": 16,
      "temperature_c": 37.0,
      "o2_saturation": 98,
      "arrival_hour": 14,
      "is_weekend": 0
    }
  }'
```

**Forecast Patient Volume:**
```bash
curl "http://localhost:5001/api/predictions/volume?hour=18&day_of_week=5&month=11&is_weekend=1"
```

## Team Roles

**Suk Jin Mun - Backend API Developer:**
- Flask API endpoints and database integration
- Statistical model training (classification, regression)
- Business logic and validation
- GitHub repository management

**Xiaobai Li - Frontend Developer:**
- React + TypeScript web application
- Data visualizations (Matplotlib, Seaborn, Plotly)
- Model evaluation notebook
- Presentation materials

**Shaobo Chen - Database Architect:**
- Database schema design (3NF normalization)
- Simulated ED data generation
- ETL pipeline development
- SQL analytical queries

## License

This project is for educational purposes as part of DS 5110 coursework at Northeastern University.
