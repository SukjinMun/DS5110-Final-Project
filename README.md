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

| Model | Accuracy | Description |
|-------|----------|-------------|
| **Random Forest** | **94.06%** | Ensemble classifier (BEST) |
| Gradient Boosting | 93.28% | Boosted trees classifier |
| Logistic Regression | 93.44% | Multinomial with SMOTE |
| Linear Discriminant Analysis | 90.16% | Probabilistic with SMOTE |
| Naive Bayes | 90.16% | Gaussian Naive Bayes with SMOTE |

### Regression Models

| Model | Metric | Value |
|-------|--------|-------|
| Wait Time (Linear Regression) | R² | 0.8463 |
| Wait Time (Linear Regression) | RMSE | 13.63 min |
| Volume (Poisson GLM) | RMSE | 0.84 patients/hour |

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

## Usage

1. Start the backend server:
   ```bash
   cd backend && python app.py
   ```

2. Start the frontend (in another terminal):
   ```bash
   cd frontend && npm run dev
   ```

3. Open browser to `http://localhost:5173`

## License

This project is for educational purposes as part of DS 5110 coursework at Northeastern University.
