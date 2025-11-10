-- ============================================
-- SQLite Script to Import CSV Data
-- ============================================
-- IMPORTANT: This script uses .import commands which require SQLite shell mode
-- 
-- Recommended usage (inside SQLite shell):
--   sqlite3 ed_database.db
--   .read import_data.sql
--
-- Alternative (if using shell redirection, may have issues with .import):
--   sqlite3 ed_database.db < import_data.sql

PRAGMA foreign_keys = OFF;  -- Disable temporarily for clearing data

-- Set CSV mode for importing
.mode csv
.separator ,

-- Note: .import tries to insert all rows including headers
-- We'll use a workaround: import to temp table, then copy data skipping first row

-- ============================================
-- Clear existing data (if any)
-- ============================================
DELETE FROM staff_assignment;
DELETE FROM diagnosis;
DELETE FROM vitals;
DELETE FROM encounter_payor;
DELETE FROM encounter;
DELETE FROM staff;
DELETE FROM patient;

-- ============================================
-- Import Data (Order matters due to foreign keys)
-- ============================================
-- Strategy: Import to temp tables, then copy data (skips header row)

-- 1. Import patient (no dependencies)
-- Create temp table, import CSV, then filter out header row
CREATE TEMP TABLE temp_patient(patient_id, dob, sex_at_birth, gender_identity, zip_code);
.import dataset/patient.csv temp_patient
INSERT INTO patient SELECT * FROM temp_patient WHERE CAST(patient_id AS TEXT) != 'patient_id';
DROP TABLE temp_patient;

-- 2. Import staff (no dependencies)
CREATE TEMP TABLE temp_staff(staff_id, first_name, last_name, role_code, department, is_active);
.import dataset/staff.csv temp_staff
INSERT INTO staff SELECT * FROM temp_staff WHERE CAST(staff_id AS TEXT) != 'staff_id';
DROP TABLE temp_staff;

-- 3. Import encounter (depends on patient)
-- Note: Dates in M/D/YY format will be imported as text
CREATE TEMP TABLE temp_encounter(encounter_id, patient_id, arrival_ts, triage_start_ts, triage_end_ts, provider_start_ts, dispo_decision_ts, departure_ts, arrival_mode, chief_complaint, esi_level, disposition_code, referral_code, left_without_being_seen, notes);
.import dataset/encounter.csv temp_encounter
INSERT INTO encounter SELECT * FROM temp_encounter WHERE CAST(encounter_id AS TEXT) != 'encounter_id';
DROP TABLE temp_encounter;

-- 4. Import encounter_payor (depends on encounter)
CREATE TEMP TABLE temp_encounter_payor(encounter_id, payor_name, payor_type, member_id);
.import dataset/encounter_payor.csv temp_encounter_payor
INSERT INTO encounter_payor SELECT * FROM temp_encounter_payor WHERE CAST(encounter_id AS TEXT) != 'encounter_id';
DROP TABLE temp_encounter_payor;

-- 5. Import vitals (depends on encounter)
CREATE TEMP TABLE temp_vitals(vital_id, encounter_id, taken_ts, heart_rate, systolic_bp, diastolic_bp, respiratory_rate, temperature_c, spo2, pain_score);
.import dataset/vitals.csv temp_vitals
INSERT INTO vitals SELECT * FROM temp_vitals WHERE CAST(vital_id AS TEXT) != 'vital_id';
DROP TABLE temp_vitals;

-- 6. Import diagnosis (depends on encounter)
CREATE TEMP TABLE temp_diagnosis(encounter_id, code, is_primary);
.import dataset/diagnosis.csv temp_diagnosis
INSERT INTO diagnosis SELECT * FROM temp_diagnosis WHERE CAST(encounter_id AS TEXT) != 'encounter_id';
DROP TABLE temp_diagnosis;

-- 7. Import staff_assignment (depends on encounter and staff)
CREATE TEMP TABLE temp_staff_assignment(encounter_id, staff_id, assignment_role, assigned_ts, released_ts);
.import dataset/staff_assignment.csv temp_staff_assignment
INSERT INTO staff_assignment SELECT * FROM temp_staff_assignment WHERE CAST(encounter_id AS TEXT) != 'encounter_id';
DROP TABLE temp_staff_assignment;

-- Re-enable foreign keys
PRAGMA foreign_keys = ON;

-- ============================================
-- IMPORTANT: Date Format Conversion
-- ============================================
-- encounter.csv uses M/D/YY format (e.g., "10/5/25 1:47")
-- SQLite's .import will import these as text strings
-- 
-- For proper date conversion, you have two options:
-- 
-- Option 1: Use the Python script (generate_db.py) which handles
--           date conversion automatically
-- 
-- Option 2: Import as-is and convert dates manually (complex in SQLite)
-- 
-- For now, dates will be imported as text. If you need ISO format dates,
-- consider using the Python import script instead.

-- ============================================
-- Verify imports
-- ============================================

SELECT 'patient' as table_name, COUNT(*) as row_count FROM patient
UNION ALL
SELECT 'staff', COUNT(*) FROM staff
UNION ALL
SELECT 'encounter', COUNT(*) FROM encounter
UNION ALL
SELECT 'encounter_payor', COUNT(*) FROM encounter_payor
UNION ALL
SELECT 'vitals', COUNT(*) FROM vitals
UNION ALL
SELECT 'diagnosis', COUNT(*) FROM diagnosis
UNION ALL
SELECT 'staff_assignment', COUNT(*) FROM staff_assignment;

-- ============================================
-- Note: encounter.csv has dates in M/D/YY format
-- You may need to convert these dates separately
-- or use the Python script (generate_db.py) which handles
-- date conversion automatically.
-- ============================================

