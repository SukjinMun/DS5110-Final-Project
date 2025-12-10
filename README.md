# Emergency Department Database and Analysis System

## Database Overview

### Temporal Coverage
The Emergency Department database (`ed_database.db`) contains a full year of synthetic patient encounter data:
- **Date Range:** January 1, 2025 - January 4, 2026
- **Total Days:** 367 days
- **Total Encounters:** 7,486 encounters
- **Average Daily Volume:** ~20 encounters per day

This database was created from prior SQL coursework and provides a comprehensive dataset for analysis.

### Date Filtering
The database supports date-based filtering using SQLite's `date()` function, which extracts the date portion from timestamp columns (`arrival_ts`, `departure_ts`). This enables queries like:
```sql
WHERE date(arrival_ts) >= '2025-01-14' AND date(arrival_ts) <= '2025-01-14'
```

---

## Post-Presentation Updates

The following components were added **after** the original DS5110 project submission to answer post-presentation questions:

### New API Endpoint: Staff Workload Statistics

### Overview
Added `/api/statistics/staff-workload` endpoint for Case Study 2 post-presentation requirements.

### Endpoint Details

**URL:** `GET /api/statistics/staff-workload`

**Query Parameters:**
- `start_date` (optional): Filter by start date in YYYY-MM-DD format
- `end_date` (optional): Filter by end date in YYYY-MM-DD format

### Usage Examples

**1. Get all staff workload:**
```bash
curl http://localhost:5001/api/statistics/staff-workload
```

**2. Get staff workload for a specific date (January 14, 2025):**
```bash
curl "http://localhost:5001/api/statistics/staff-workload?start_date=2025-01-14&end_date=2025-01-14"
```

**3. Get staff workload for a date range:**
```bash
curl "http://localhost:5001/api/statistics/staff-workload?start_date=2025-01-01&end_date=2025-01-31"
```

### Response Format

```json
{
  "total_assignments": 16,
  "total_unique_staff": 16,
  "total_patients_served": 16,
  "date_range": {
    "start_date": "2025-01-14",
    "end_date": "2025-01-14"
  },
  "staff_workload": [
    {
      "staff_id": 14,
      "staff_name": "John Doe",
      "staff_role": "MD_ATT",
      "patient_count": 5,
      "average_encounter_duration_minutes": 246.0,
      "average_assignment_duration_minutes": 75.0,
      "total_assignment_time_minutes": 375,
      "esi_distribution": {
        "1": 0,
        "2": 1,
        "3": 2,
        "4": 2,
        "5": 0
      },
      "assignment_roles": {
        "Attending": 5
      }
    }
  ],
  "workload_by_role": [
    {
      "role": "MD_ATT",
      "staff_count": 3,
      "total_patient_count": 15,
      "average_patients_per_staff": 5.0,
      "average_encounter_duration_minutes": 240.5
    }
  ]
}
```

### Use Case: Case Study 2

This endpoint enables SQL vs API validation for staff workload distribution:

1. **SQL Query:** Run direct database queries for staff metrics
2. **API Call:** Use this endpoint with same date filters
3. **Cross-validate:** Compare results to ensure consistency

### Implementation

**File Modified:** `backend/routes/api.py` (lines 860-1022)

**Database Tables Used:**
- `staff_assignment` - Staff-to-encounter assignments
- `staff` - Staff member details
- `encounter` - Patient encounter information
