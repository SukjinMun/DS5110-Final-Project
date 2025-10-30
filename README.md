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

## Team Roles

**Suk Jin Mun - Backend API Developer:**
- Design Flask API endpoints (receive frontend requests, return JSON)
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
- Flask web framework with HTML/CSS
- Web forms that call backend API endpoints
- Display analytical charts and SQL query results from backend API
- **Frontend constraint:** Cannot directly connect to database, must call backend API

**Shaobo Chen - Database Architect:**
- Database schema and ERD design
- 3NF normalization
- Generate simulated ED data (https://data.gov/ distributions)
- Build ETL pipelines
- SQL analytical queries (executed by backend API)
- Collaborate on statistical model training
