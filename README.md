# Emergency Department Database and Analysis System

**Course:** DS 5110: Introduction to Data Management and Processing, Fall 2025
**Team 22:** Suk Jin Mun, Xiaobai Li, Shaobo (Ben) Chen

## Overview

This project analyzes Emergency Department triage classification and patient flow using a normalized relational database and statistical modeling. Core focus: data analysis and statistical modeling. Optional stretch goal: interactive web dashboard if time permits.

## Objectives

**Core (Required):**
1. Design normalized database (3NF) for patient visits, triage, treatments, wait times
2. Develop statistical models for patient urgency classification and wait time prediction
3. Create data visualizations for hospital administrators
4. Provide actionable recommendations for resource allocation

**Optional (Stretch):**
5. Build Flask web application
6. Create interactive web dashboard with HTML/CSS

## Course Concepts

**Database & SQL:** ERD design, 3NF normalization, complex queries with joins/aggregations, indexing

**Data Engineering:** ETL pipelines, Pandas cleaning/transformation, missing value handling, validation

**Visualization:** Matplotlib/Seaborn plots, dashboards, time series, heatmaps

**Statistical Modeling:**
- Linear regression (wait time prediction)
- Classification (logistic, LDA, naive Bayes for urgency levels)
- Poisson regression (daily visit forecasting)

**Optional Web:**
- Flask web framework
- HTML/CSS frontend
- Web application design

## Team Roles

**Suk Jin Mun - Backend API Developer:**
- Design RESTful API endpoints (receive frontend requests, return JSON)
- Access database layer (execute SQL on behalf of frontend)
- Train statistical models (classification, regression)
- Implement business logic validation
- Calculate statistical indicators (conversion rates, averages)
- Perform automatic calculations (wait times)
- Model evaluation and result interpretation
- GitHub management
- **Integration layer:** Frontend cannot directly access database, must call backend API

**Xiaobai Li - Frontend Developer:**
- Exploratory analysis visualizations
- Matplotlib/Seaborn dashboards
- Presentation materials
- (Optional) Flask web framework with HTML/CSS
- (Optional) Web forms that call backend API endpoints
- (Optional) Display analytical charts and SQL query results from backend API
- **Frontend constraint:** Cannot directly connect to database, must call backend API

**Shaobo Chen - Database Architect:**
- Database schema and ERD design
- 3NF normalization
- Generate simulated ED data (data.gov distributions)
- Build ETL pipelines
- SQL analytical queries (executed by backend API)
- Collaborate on statistical model training

## Technology Stack

**Core:**
- Database: SQLite
- Programming: Python 3.9+
- Libraries: Pandas, NumPy, Matplotlib, Seaborn, scikit-learn, statsmodels
- Documentation: Overleaf, Git, GitHub
- Tracking: Excel

**Optional (Web):**
- Backend API: Flask with RESTful endpoints, request handling, JSON responses
- Frontend: HTML/CSS (calls backend API)
- Architecture: 3-tier (Frontend ↔ Backend API ↔ Database)
- Database: PostgreSQL upgrade

## Data Generation

Simulated ED data avoiding HIPAA concerns. Realistic distributions for demographics, arrivals, vital signs, urgency, wait times based on medical literature.

## Timeline

**Iteration 02 (Oct Week 4):** Proposal, tracker, GitHub repo

**Iteration 03 (3 weeks):** Database schema, data generation, SQL queries, visualizations  
→ Week 3 checkpoint: Decide on optional web app

**Iteration 04 (3 weeks):** Data cleaning, statistical models, evaluation
→ (Optional) Flask web application development

**Iteration 05 (2 weeks):** Final report, presentation, documentation  
→ (Optional) Web demo

## Priorities

**Priority 1 (Core):**
- Normalized database with data
- Statistical models with evaluation
- Visualizations and analysis
- Final report

**Priority 2 (Optional):**
- Flask web application
- HTML/CSS dashboard
- Web deployment

Optional web app only pursued if core deliverables on track after Iteration 03 Week 3 checkpoint.
