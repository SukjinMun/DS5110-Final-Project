# Emergency Department API - Backend

Backend API for the Emergency Department Database and Analysis System.

**Developer:** Suk Jin Mun
**Course:** DS 5110, Fall 2025

## Architecture

Three-tier architecture:
- **Frontend** (Xiaobai): Web interface and visualizations
- **Backend** (Suk Jin): Flask API, business logic, statistical models
- **Database** (Shaobo): SQLite database with normalized schema

## Directory Structure

```
backend/
├── app.py                  # Flask application entry point
├── config/
│   └── database.py         # Database connection and configuration
├── models/
│   ├── orm_models.py       # SQLAlchemy ORM models
│   ├── classifiers.py      # ESI classification models
│   └── regressors.py       # Wait time prediction models
├── routes/
│   └── api.py              # REST API endpoints
├── utils/
│   └── validators.py       # Business logic and validation
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Setup

1. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create database (if not exists):
```bash
cd ..
sqlite3 ed_database.db < db_setup.sql
sqlite3 ed_database.db < db_import.sql
```

4. Run Flask app:
```bash
python app.py
```

API will be available at `http://localhost:5000`

## API Endpoints

### Health Check
- `GET /api/health` - Check API status

### Encounters
- `GET /api/encounters` - List encounters (supports filtering by ESI level, disposition)
- `GET /api/encounters/<id>` - Get detailed encounter information

### Patients
- `GET /api/patients/<id>` - Get patient information and encounter history

### Statistics
- `GET /api/statistics/overview` - Overall ED statistics
- `GET /api/statistics/esi` - ESI level statistics
- `GET /api/statistics/vitals` - Vital signs statistics
- `GET /api/statistics/payor` - Payor distribution
- `GET /api/statistics/diagnoses` - Diagnosis code statistics

### Other
- `GET /api/chief-complaints` - Chief complaint distribution
- `GET /api/staff` - Staff list

### Prediction Endpoints (NEW!)
- `GET /api/predictions/models/info` - Get information about available models
- `POST /api/predictions/esi` - Predict ESI level from patient characteristics
- `POST /api/predictions/wait-time` - Predict wait time for a patient
- `GET /api/predictions/volume` - Forecast patient arrival volume

### Future Endpoints
- `GET /api/wait-times` - Wait time analysis
- `GET /api/statistics/wait-times-by-esi` - Wait times by ESI level
- `GET /api/statistics/length-of-stay` - Length of stay analysis

## Statistical Models

### Classification Models (`models/classifiers.py`)

| Model | Accuracy | AUC | 5-Fold CV |
|-------|----------|-----|-----------|
| **Logistic Regression** | **85.66%** | 0.9756 | 84.75% ±0.83% |
| Random Forest | 85.47% | 0.9764 | 85.08% ±0.92% |
| Gradient Boosting | 84.30% | 0.9696 | 84.11% ±1.10% |
| LDA | 83.80% | 0.9712 | 82.96% ±0.53% |
| Naive Bayes | 60.58% | 0.9049 | 57.70% ±10.02% |

*Metrics from DS5110 class (Ch4, Week 6). Accuracy (~80-85%) aligns with published ML studies on real ESI data (70-80%). 30% nurse variability matches real-world triage disagreement.*

**Usage:**
```python
from models.classifiers import ESIClassifier

clf = ESIClassifier(model_type='logistic')
clf.train(X_train, y_train)
predictions = clf.predict(X_test)
results = clf.evaluate(X_test, y_test)
```

### Regression Models (`models/regressors.py`)
- **Linear Regression**: Wait time prediction
- **Poisson GLM**: Patient volume prediction

**Usage:**
```python
from models.regressors import WaitTimePredictor

predictor = WaitTimePredictor()
predictor.train(X_train, y_train)
wait_times = predictor.predict(X_test)
```

## Database Schema

The API connects to SQLite database with the following tables:
- `patient` - Patient demographics
- `staff` - ED staff information
- `encounter` - ED visits
- `encounter_payor` - Insurance information
- `vitals` - Vital sign measurements
- `diagnosis` - ICD-10 diagnosis codes
- `staff_assignment` - Staff-patient assignments

See `../db_setup.sql` for complete schema.

## Development Status

### Completed
- ✅ Flask application structure
- ✅ SQLAlchemy ORM models
- ✅ Basic API endpoints for data retrieval
- ✅ Statistics endpoints
- ✅ Business logic validation
- ✅ Classification models trained (Random Forest, Gradient Boosting, Logistic Regression, LDA, Naive Bayes)
- ✅ Regression models trained (Wait Time, Volume Prediction)
- ✅ Model evaluation (confusion matrix, ROC curves, learning curves)
- ✅ Model validation using DS5110 class methodologies
- ✅ API endpoints for model predictions
- ✅ Integration with frontend

### Model Performance
- **Best Classification Accuracy:** 85.66% (Logistic Regression), AUC: 0.9756
- **Best AUC:** 0.9764 (Random Forest)
- **Wait Time R²:** 0.8570
- **Volume RMSE:** 0.86 patients/hour

## Notes

- Frontend cannot directly access database - must call backend API
- All models validated using 5-fold cross-validation, ROC/AUC analysis, and learning curves
- ESI classification based on clinically-validated vital sign correlations (see dataset/README.md)
