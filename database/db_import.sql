-- ============================================
-- SQLite Script to Import CSV Data with ETL Cleaning
-- ============================================
-- IMPORTANT: This script uses .import commands which require SQLite shell mode
-- 
-- Recommended usage (inside SQLite shell):
--   sqlite3 ed_database.db
--   .read database/db_import.sql
--
-- Alternative (if using shell redirection, may have issues with .import):
--   sqlite3 ed_database.db < database/db_import.sql

PRAGMA foreign_keys = OFF;  -- Disable temporarily for clearing data

-- Set CSV mode for importing
.mode csv
.separator ,

-- Clear cleaning log
DELETE FROM cleaning_log;

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
-- Import Data with Validation and Cleaning
-- ============================================
-- Strategy: Import to temp tables, validate, clean, then insert valid data

-- 1. Import patient (no dependencies)
CREATE TEMP TABLE temp_patient(patient_id, dob, sex_at_birth, gender_identity, zip_code);
.import dataset/patient.csv temp_patient

-- Count total rows (including header)
SELECT COUNT(*) INTO @total_patient FROM temp_patient;

-- Insert valid data (filter header, null IDs, invalid IDs)
INSERT INTO patient 
SELECT * FROM temp_patient 
WHERE CAST(patient_id AS TEXT) != 'patient_id'
  AND patient_id IS NOT NULL 
  AND patient_id != ''
  AND CAST(patient_id AS INTEGER) > 0
  AND (dob IS NULL OR dob = '' OR dob LIKE '____-__-__')  -- Valid ISO date or empty
  AND (sex_at_birth IS NULL OR sex_at_birth = '' OR sex_at_birth IN ('Male', 'Female'))
  AND (gender_identity IS NULL OR gender_identity = '' OR gender_identity IN ('Male', 'Female', 'Non-binary', 'Prefer not to say'));

-- Log cleaning
INSERT INTO cleaning_log (table_name, issue_type, issue_description, rows_affected)
SELECT 'patient', 'header_row', 'Header row filtered', 1
WHERE EXISTS (SELECT 1 FROM temp_patient WHERE CAST(patient_id AS TEXT) = 'patient_id')
UNION ALL
SELECT 'patient', 'invalid_id', 'Invalid patient_id filtered', 
       (SELECT COUNT(*) FROM temp_patient 
        WHERE CAST(patient_id AS TEXT) != 'patient_id' 
          AND (patient_id IS NULL OR patient_id = '' OR CAST(patient_id AS INTEGER) <= 0))
UNION ALL
SELECT 'patient', 'missing_value', 'Missing required values filtered',
       (SELECT COUNT(*) FROM temp_patient 
        WHERE CAST(patient_id AS TEXT) != 'patient_id'
          AND patient_id IS NOT NULL 
          AND patient_id != ''
          AND CAST(patient_id AS INTEGER) > 0
          AND (dob IS NULL OR dob = '' OR dob NOT LIKE '____-__-__'))
UNION ALL
SELECT 'patient', 'invalid_enum', 'Invalid enum values filtered',
       (SELECT COUNT(*) FROM temp_patient 
        WHERE CAST(patient_id AS TEXT) != 'patient_id'
          AND patient_id IS NOT NULL 
          AND patient_id != ''
          AND CAST(patient_id AS INTEGER) > 0
          AND ((sex_at_birth IS NOT NULL AND sex_at_birth != '' AND sex_at_birth NOT IN ('Male', 'Female'))
            OR (gender_identity IS NOT NULL AND gender_identity != '' AND gender_identity NOT IN ('Male', 'Female', 'Non-binary', 'Prefer not to say'))));

DROP TABLE temp_patient;

-- 2. Import staff (no dependencies)
CREATE TEMP TABLE temp_staff(staff_id, first_name, last_name, role_code, department, is_active);
.import dataset/staff.csv temp_staff

-- Insert valid data
INSERT INTO staff 
SELECT * FROM temp_staff 
WHERE CAST(staff_id AS TEXT) != 'staff_id'
  AND staff_id IS NOT NULL 
  AND staff_id != ''
  AND CAST(staff_id AS INTEGER) > 0
  AND first_name IS NOT NULL AND first_name != ''
  AND last_name IS NOT NULL AND last_name != ''
  AND role_code IS NOT NULL AND role_code != ''
  AND role_code IN ('MD_ATT', 'MD_RES', 'PA', 'NP', 'RN', 'TECH', 'REG')
  AND (is_active IS NULL OR is_active = '' OR CAST(is_active AS INTEGER) IN (0, 1));

-- Log cleaning
INSERT INTO cleaning_log (table_name, issue_type, issue_description, rows_affected)
SELECT 'staff', 'header_row', 'Header row filtered', 1
WHERE EXISTS (SELECT 1 FROM temp_staff WHERE CAST(staff_id AS TEXT) = 'staff_id')
UNION ALL
SELECT 'staff', 'invalid_id', 'Invalid staff_id filtered',
       (SELECT COUNT(*) FROM temp_staff 
        WHERE CAST(staff_id AS TEXT) != 'staff_id' 
          AND (staff_id IS NULL OR staff_id = '' OR CAST(staff_id AS INTEGER) <= 0))
UNION ALL
SELECT 'staff', 'missing_value', 'Missing required values filtered',
       (SELECT COUNT(*) FROM temp_staff 
        WHERE CAST(staff_id AS TEXT) != 'staff_id'
          AND staff_id IS NOT NULL 
          AND staff_id != ''
          AND CAST(staff_id AS INTEGER) > 0
          AND (first_name IS NULL OR first_name = '' OR last_name IS NULL OR last_name = '' OR role_code IS NULL OR role_code = ''))
UNION ALL
SELECT 'staff', 'invalid_enum', 'Invalid role_code or is_active filtered',
       (SELECT COUNT(*) FROM temp_staff 
        WHERE CAST(staff_id AS TEXT) != 'staff_id'
          AND staff_id IS NOT NULL 
          AND staff_id != ''
          AND CAST(staff_id AS INTEGER) > 0
          AND ((role_code IS NOT NULL AND role_code != '' AND role_code NOT IN ('MD_ATT', 'MD_RES', 'PA', 'NP', 'RN', 'TECH', 'REG'))
            OR (is_active IS NOT NULL AND is_active != '' AND CAST(is_active AS INTEGER) NOT IN (0, 1))));

DROP TABLE temp_staff;

-- 3. Import encounter (depends on patient)
CREATE TEMP TABLE temp_encounter(encounter_id, patient_id, arrival_ts, triage_start_ts, triage_end_ts, provider_start_ts, dispo_decision_ts, departure_ts, arrival_mode, chief_complaint, esi_level, disposition_code, referral_code, left_without_being_seen, notes);
.import dataset/encounter.csv temp_encounter

-- Insert valid data
INSERT INTO encounter 
SELECT * FROM temp_encounter 
WHERE CAST(encounter_id AS TEXT) != 'encounter_id'
  AND encounter_id IS NOT NULL 
  AND encounter_id != ''
  AND CAST(encounter_id AS INTEGER) > 0
  AND patient_id IS NOT NULL 
  AND patient_id != ''
  AND CAST(patient_id AS INTEGER) > 0
  AND CAST(patient_id AS INTEGER) IN (SELECT patient_id FROM patient)  -- Valid foreign key
  AND (esi_level IS NULL OR esi_level = '' OR CAST(esi_level AS INTEGER) BETWEEN 1 AND 5)
  AND (arrival_mode IS NULL OR arrival_mode = '' OR arrival_mode IN ('Walk-in', 'EMS', 'Transfer'))
  AND (left_without_being_seen IS NULL OR left_without_being_seen = '' OR CAST(left_without_being_seen AS INTEGER) IN (0, 1))
  AND (disposition_code IS NULL OR disposition_code = '' OR disposition_code IN ('DISCH_HOME', 'ADMIT_INPT', 'ADMIT_OBS', 'TRANSFER', 'LWBS', 'AMA', 'EXPIRED'));

-- Log cleaning
INSERT INTO cleaning_log (table_name, issue_type, issue_description, rows_affected)
SELECT 'encounter', 'header_row', 'Header row filtered', 1
WHERE EXISTS (SELECT 1 FROM temp_encounter WHERE CAST(encounter_id AS TEXT) = 'encounter_id')
UNION ALL
SELECT 'encounter', 'invalid_id', 'Invalid encounter_id filtered',
       (SELECT COUNT(*) FROM temp_encounter 
        WHERE CAST(encounter_id AS TEXT) != 'encounter_id' 
          AND (encounter_id IS NULL OR encounter_id = '' OR CAST(encounter_id AS INTEGER) <= 0))
UNION ALL
SELECT 'encounter', 'invalid_fk', 'Invalid patient_id (foreign key) filtered',
       (SELECT COUNT(*) FROM temp_encounter 
        WHERE CAST(encounter_id AS TEXT) != 'encounter_id'
          AND encounter_id IS NOT NULL 
          AND encounter_id != ''
          AND CAST(encounter_id AS INTEGER) > 0
          AND (patient_id IS NULL OR patient_id = '' OR CAST(patient_id AS INTEGER) <= 0 
            OR CAST(patient_id AS INTEGER) NOT IN (SELECT patient_id FROM patient)))
UNION ALL
SELECT 'encounter', 'out_of_range', 'Invalid ESI level or left_without_being_seen filtered',
       (SELECT COUNT(*) FROM temp_encounter 
        WHERE CAST(encounter_id AS TEXT) != 'encounter_id'
          AND encounter_id IS NOT NULL 
          AND encounter_id != ''
          AND CAST(encounter_id AS INTEGER) > 0
          AND patient_id IS NOT NULL 
          AND patient_id != ''
          AND CAST(patient_id AS INTEGER) > 0
          AND CAST(patient_id AS INTEGER) IN (SELECT patient_id FROM patient)
          AND ((esi_level IS NOT NULL AND esi_level != '' AND CAST(esi_level AS INTEGER) NOT BETWEEN 1 AND 5)
            OR (left_without_being_seen IS NOT NULL AND left_without_being_seen != '' AND CAST(left_without_being_seen AS INTEGER) NOT IN (0, 1))))
UNION ALL
SELECT 'encounter', 'invalid_enum', 'Invalid arrival_mode or disposition_code filtered',
       (SELECT COUNT(*) FROM temp_encounter 
        WHERE CAST(encounter_id AS TEXT) != 'encounter_id'
          AND encounter_id IS NOT NULL 
          AND encounter_id != ''
          AND CAST(encounter_id AS INTEGER) > 0
          AND patient_id IS NOT NULL 
          AND patient_id != ''
          AND CAST(patient_id AS INTEGER) > 0
          AND CAST(patient_id AS INTEGER) IN (SELECT patient_id FROM patient)
          AND (esi_level IS NULL OR esi_level = '' OR CAST(esi_level AS INTEGER) BETWEEN 1 AND 5)
          AND (left_without_being_seen IS NULL OR left_without_being_seen = '' OR CAST(left_without_being_seen AS INTEGER) IN (0, 1))
          AND ((arrival_mode IS NOT NULL AND arrival_mode != '' AND arrival_mode NOT IN ('Walk-in', 'EMS', 'Transfer'))
            OR (disposition_code IS NOT NULL AND disposition_code != '' AND disposition_code NOT IN ('DISCH_HOME', 'ADMIT_INPT', 'ADMIT_OBS', 'TRANSFER', 'LWBS', 'AMA', 'EXPIRED'))));

DROP TABLE temp_encounter;

-- 4. Import encounter_payor (depends on encounter)
CREATE TEMP TABLE temp_encounter_payor(encounter_id, payor_name, payor_type, member_id);
.import dataset/encounter_payor.csv temp_encounter_payor

-- Insert valid data
INSERT INTO encounter_payor 
SELECT * FROM temp_encounter_payor 
WHERE CAST(encounter_id AS TEXT) != 'encounter_id'
  AND encounter_id IS NOT NULL 
  AND encounter_id != ''
  AND CAST(encounter_id AS INTEGER) > 0
  AND CAST(encounter_id AS INTEGER) IN (SELECT encounter_id FROM encounter)  -- Valid foreign key
  AND (payor_type IS NULL OR payor_type = '' OR payor_type IN ('Public', 'Private', 'Self'));

-- Log cleaning
INSERT INTO cleaning_log (table_name, issue_type, issue_description, rows_affected)
SELECT 'encounter_payor', 'header_row', 'Header row filtered', 1
WHERE EXISTS (SELECT 1 FROM temp_encounter_payor WHERE CAST(encounter_id AS TEXT) = 'encounter_id')
UNION ALL
SELECT 'encounter_payor', 'invalid_id', 'Invalid encounter_id filtered',
       (SELECT COUNT(*) FROM temp_encounter_payor 
        WHERE CAST(encounter_id AS TEXT) != 'encounter_id' 
          AND (encounter_id IS NULL OR encounter_id = '' OR CAST(encounter_id AS INTEGER) <= 0))
UNION ALL
SELECT 'encounter_payor', 'invalid_fk', 'Invalid encounter_id (foreign key) filtered',
       (SELECT COUNT(*) FROM temp_encounter_payor 
        WHERE CAST(encounter_id AS TEXT) != 'encounter_id'
          AND encounter_id IS NOT NULL 
          AND encounter_id != ''
          AND CAST(encounter_id AS INTEGER) > 0
          AND CAST(encounter_id AS INTEGER) NOT IN (SELECT encounter_id FROM encounter))
UNION ALL
SELECT 'encounter_payor', 'invalid_enum', 'Invalid payor_type filtered',
       (SELECT COUNT(*) FROM temp_encounter_payor 
        WHERE CAST(encounter_id AS TEXT) != 'encounter_id'
          AND encounter_id IS NOT NULL 
          AND encounter_id != ''
          AND CAST(encounter_id AS INTEGER) > 0
          AND CAST(encounter_id AS INTEGER) IN (SELECT encounter_id FROM encounter)
          AND payor_type IS NOT NULL 
          AND payor_type != ''
          AND payor_type NOT IN ('Public', 'Private', 'Self'));

DROP TABLE temp_encounter_payor;

-- 5. Import vitals (depends on encounter)
CREATE TEMP TABLE temp_vitals(vital_id, encounter_id, taken_ts, heart_rate, systolic_bp, diastolic_bp, respiratory_rate, temperature_c, spo2, pain_score);
.import dataset/vitals.csv temp_vitals

-- Insert valid data
INSERT INTO vitals 
SELECT * FROM temp_vitals 
WHERE CAST(vital_id AS TEXT) != 'vital_id'
  AND vital_id IS NOT NULL 
  AND vital_id != ''
  AND CAST(vital_id AS INTEGER) > 0
  AND encounter_id IS NOT NULL 
  AND encounter_id != ''
  AND CAST(encounter_id AS INTEGER) > 0
  AND CAST(encounter_id AS INTEGER) IN (SELECT encounter_id FROM encounter)  -- Valid foreign key
  AND taken_ts IS NOT NULL AND taken_ts != ''
  AND (heart_rate IS NULL OR heart_rate = '' OR (CAST(heart_rate AS INTEGER) BETWEEN 40 AND 200))
  AND (systolic_bp IS NULL OR systolic_bp = '' OR (CAST(systolic_bp AS INTEGER) BETWEEN 60 AND 220))
  AND (diastolic_bp IS NULL OR diastolic_bp = '' OR (CAST(diastolic_bp AS INTEGER) BETWEEN 40 AND 140))
  AND (respiratory_rate IS NULL OR respiratory_rate = '' OR (CAST(respiratory_rate AS INTEGER) BETWEEN 8 AND 45))
  AND (temperature_c IS NULL OR temperature_c = '' OR (CAST(temperature_c AS REAL) BETWEEN 35.0 AND 41.5))
  AND (spo2 IS NULL OR spo2 = '' OR (CAST(spo2 AS INTEGER) BETWEEN 70 AND 100))
  AND (pain_score IS NULL OR pain_score = '' OR (CAST(pain_score AS INTEGER) BETWEEN 0 AND 10));

-- Log cleaning
INSERT INTO cleaning_log (table_name, issue_type, issue_description, rows_affected)
SELECT 'vitals', 'header_row', 'Header row filtered', 1
WHERE EXISTS (SELECT 1 FROM temp_vitals WHERE CAST(vital_id AS TEXT) = 'vital_id')
UNION ALL
SELECT 'vitals', 'invalid_id', 'Invalid vital_id filtered',
       (SELECT COUNT(*) FROM temp_vitals 
        WHERE CAST(vital_id AS TEXT) != 'vital_id' 
          AND (vital_id IS NULL OR vital_id = '' OR CAST(vital_id AS INTEGER) <= 0))
UNION ALL
SELECT 'vitals', 'missing_value', 'Missing taken_ts filtered',
       (SELECT COUNT(*) FROM temp_vitals 
        WHERE CAST(vital_id AS TEXT) != 'vital_id'
          AND vital_id IS NOT NULL 
          AND vital_id != ''
          AND CAST(vital_id AS INTEGER) > 0
          AND (taken_ts IS NULL OR taken_ts = ''))
UNION ALL
SELECT 'vitals', 'invalid_fk', 'Invalid encounter_id (foreign key) filtered',
       (SELECT COUNT(*) FROM temp_vitals 
        WHERE CAST(vital_id AS TEXT) != 'vital_id'
          AND vital_id IS NOT NULL 
          AND vital_id != ''
          AND CAST(vital_id AS INTEGER) > 0
          AND (encounter_id IS NULL OR encounter_id = '' OR CAST(encounter_id AS INTEGER) <= 0 
            OR CAST(encounter_id AS INTEGER) NOT IN (SELECT encounter_id FROM encounter)))
UNION ALL
SELECT 'vitals', 'out_of_range', 'Out-of-range vital sign values filtered',
       (SELECT COUNT(*) FROM temp_vitals 
        WHERE CAST(vital_id AS TEXT) != 'vital_id'
          AND vital_id IS NOT NULL 
          AND vital_id != ''
          AND CAST(vital_id AS INTEGER) > 0
          AND encounter_id IS NOT NULL 
          AND encounter_id != ''
          AND CAST(encounter_id AS INTEGER) > 0
          AND CAST(encounter_id AS INTEGER) IN (SELECT encounter_id FROM encounter)
          AND taken_ts IS NOT NULL AND taken_ts != ''
          AND ((heart_rate IS NOT NULL AND heart_rate != '' AND CAST(heart_rate AS INTEGER) NOT BETWEEN 40 AND 200)
            OR (systolic_bp IS NOT NULL AND systolic_bp != '' AND CAST(systolic_bp AS INTEGER) NOT BETWEEN 60 AND 220)
            OR (diastolic_bp IS NOT NULL AND diastolic_bp != '' AND CAST(diastolic_bp AS INTEGER) NOT BETWEEN 40 AND 140)
            OR (respiratory_rate IS NOT NULL AND respiratory_rate != '' AND CAST(respiratory_rate AS INTEGER) NOT BETWEEN 8 AND 45)
            OR (temperature_c IS NOT NULL AND temperature_c != '' AND CAST(temperature_c AS REAL) NOT BETWEEN 35.0 AND 41.5)
            OR (spo2 IS NOT NULL AND spo2 != '' AND CAST(spo2 AS INTEGER) NOT BETWEEN 70 AND 100)
            OR (pain_score IS NOT NULL AND pain_score != '' AND CAST(pain_score AS INTEGER) NOT BETWEEN 0 AND 10)));

DROP TABLE temp_vitals;

-- 6. Import diagnosis (depends on encounter)
CREATE TEMP TABLE temp_diagnosis(encounter_id, code, is_primary);
.import dataset/diagnosis.csv temp_diagnosis

-- Insert valid data
INSERT INTO diagnosis 
SELECT * FROM temp_diagnosis 
WHERE CAST(encounter_id AS TEXT) != 'encounter_id'
  AND encounter_id IS NOT NULL 
  AND encounter_id != ''
  AND CAST(encounter_id AS INTEGER) > 0
  AND CAST(encounter_id AS INTEGER) IN (SELECT encounter_id FROM encounter)  -- Valid foreign key
  AND code IS NOT NULL AND code != ''
  AND code NOT LIKE 'INVALID%' AND code != 'XXX' AND code != 'NOT_A_CODE'  -- Filter invalid codes
  AND (is_primary IS NULL OR is_primary = '' OR CAST(is_primary AS INTEGER) IN (0, 1));

-- Log cleaning
INSERT INTO cleaning_log (table_name, issue_type, issue_description, rows_affected)
SELECT 'diagnosis', 'header_row', 'Header row filtered', 1
WHERE EXISTS (SELECT 1 FROM temp_diagnosis WHERE CAST(encounter_id AS TEXT) = 'encounter_id')
UNION ALL
SELECT 'diagnosis', 'invalid_id', 'Invalid encounter_id filtered',
       (SELECT COUNT(*) FROM temp_diagnosis 
        WHERE CAST(encounter_id AS TEXT) != 'encounter_id' 
          AND (encounter_id IS NULL OR encounter_id = '' OR CAST(encounter_id AS INTEGER) <= 0))
UNION ALL
SELECT 'diagnosis', 'invalid_fk', 'Invalid encounter_id (foreign key) filtered',
       (SELECT COUNT(*) FROM temp_diagnosis 
        WHERE CAST(encounter_id AS TEXT) != 'encounter_id'
          AND encounter_id IS NOT NULL 
          AND encounter_id != ''
          AND CAST(encounter_id AS INTEGER) > 0
          AND CAST(encounter_id AS INTEGER) NOT IN (SELECT encounter_id FROM encounter))
UNION ALL
SELECT 'diagnosis', 'missing_value', 'Missing diagnosis code filtered',
       (SELECT COUNT(*) FROM temp_diagnosis 
        WHERE CAST(encounter_id AS TEXT) != 'encounter_id'
          AND encounter_id IS NOT NULL 
          AND encounter_id != ''
          AND CAST(encounter_id AS INTEGER) > 0
          AND CAST(encounter_id AS INTEGER) IN (SELECT encounter_id FROM encounter)
          AND (code IS NULL OR code = ''))
UNION ALL
SELECT 'diagnosis', 'invalid_enum', 'Invalid diagnosis code filtered',
       (SELECT COUNT(*) FROM temp_diagnosis 
        WHERE CAST(encounter_id AS TEXT) != 'encounter_id'
          AND encounter_id IS NOT NULL 
          AND encounter_id != ''
          AND CAST(encounter_id AS INTEGER) > 0
          AND CAST(encounter_id AS INTEGER) IN (SELECT encounter_id FROM encounter)
          AND code IS NOT NULL AND code != ''
          AND (code LIKE 'INVALID%' OR code = 'XXX' OR code = 'NOT_A_CODE'))
UNION ALL
SELECT 'diagnosis', 'out_of_range', 'Invalid is_primary value filtered',
       (SELECT COUNT(*) FROM temp_diagnosis 
        WHERE CAST(encounter_id AS TEXT) != 'encounter_id'
          AND encounter_id IS NOT NULL 
          AND encounter_id != ''
          AND CAST(encounter_id AS INTEGER) > 0
          AND CAST(encounter_id AS INTEGER) IN (SELECT encounter_id FROM encounter)
          AND code IS NOT NULL AND code != ''
          AND code NOT LIKE 'INVALID%' AND code != 'XXX' AND code != 'NOT_A_CODE'
          AND (is_primary IS NOT NULL AND is_primary != '' AND CAST(is_primary AS INTEGER) NOT IN (0, 1)));

DROP TABLE temp_diagnosis;

-- 7. Import staff_assignment (depends on encounter and staff)
CREATE TEMP TABLE temp_staff_assignment(encounter_id, staff_id, assignment_role, assigned_ts, released_ts);
.import dataset/staff_assignment.csv temp_staff_assignment

-- Insert valid data
INSERT INTO staff_assignment 
SELECT * FROM temp_staff_assignment 
WHERE CAST(encounter_id AS TEXT) != 'encounter_id'
  AND encounter_id IS NOT NULL 
  AND encounter_id != ''
  AND CAST(encounter_id AS INTEGER) > 0
  AND CAST(encounter_id AS INTEGER) IN (SELECT encounter_id FROM encounter)  -- Valid foreign key
  AND staff_id IS NOT NULL 
  AND staff_id != ''
  AND CAST(staff_id AS INTEGER) > 0
  AND CAST(staff_id AS INTEGER) IN (SELECT staff_id FROM staff)  -- Valid foreign key
  AND assignment_role IS NOT NULL AND assignment_role != ''
  AND assignment_role IN ('Attending', 'Resident', 'PA', 'NP', 'RN', 'Tech', 'Registration')  -- Valid roles
  AND assigned_ts IS NOT NULL AND assigned_ts != '';

-- Log cleaning
INSERT INTO cleaning_log (table_name, issue_type, issue_description, rows_affected)
SELECT 'staff_assignment', 'header_row', 'Header row filtered', 1
WHERE EXISTS (SELECT 1 FROM temp_staff_assignment WHERE CAST(encounter_id AS TEXT) = 'encounter_id')
UNION ALL
SELECT 'staff_assignment', 'invalid_id', 'Invalid encounter_id or staff_id filtered',
       (SELECT COUNT(*) FROM temp_staff_assignment 
        WHERE CAST(encounter_id AS TEXT) != 'encounter_id' 
          AND ((encounter_id IS NULL OR encounter_id = '' OR CAST(encounter_id AS INTEGER) <= 0)
            OR (staff_id IS NULL OR staff_id = '' OR CAST(staff_id AS INTEGER) <= 0)))
UNION ALL
SELECT 'staff_assignment', 'invalid_fk', 'Invalid foreign keys filtered',
       (SELECT COUNT(*) FROM temp_staff_assignment 
        WHERE CAST(encounter_id AS TEXT) != 'encounter_id'
          AND encounter_id IS NOT NULL 
          AND encounter_id != ''
          AND CAST(encounter_id AS INTEGER) > 0
          AND staff_id IS NOT NULL 
          AND staff_id != ''
          AND CAST(staff_id AS INTEGER) > 0
          AND (CAST(encounter_id AS INTEGER) NOT IN (SELECT encounter_id FROM encounter)
            OR CAST(staff_id AS INTEGER) NOT IN (SELECT staff_id FROM staff)))
UNION ALL
SELECT 'staff_assignment', 'missing_value', 'Missing required values filtered',
       (SELECT COUNT(*) FROM temp_staff_assignment 
        WHERE CAST(encounter_id AS TEXT) != 'encounter_id'
          AND encounter_id IS NOT NULL 
          AND encounter_id != ''
          AND CAST(encounter_id AS INTEGER) > 0
          AND CAST(encounter_id AS INTEGER) IN (SELECT encounter_id FROM encounter)
          AND staff_id IS NOT NULL 
          AND staff_id != ''
          AND CAST(staff_id AS INTEGER) > 0
          AND CAST(staff_id AS INTEGER) IN (SELECT staff_id FROM staff)
          AND (assignment_role IS NULL OR assignment_role = '' OR assigned_ts IS NULL OR assigned_ts = ''))
UNION ALL
SELECT 'staff_assignment', 'invalid_enum', 'Invalid assignment_role filtered',
       (SELECT COUNT(*) FROM temp_staff_assignment 
        WHERE CAST(encounter_id AS TEXT) != 'encounter_id'
          AND encounter_id IS NOT NULL 
          AND encounter_id != ''
          AND CAST(encounter_id AS INTEGER) > 0
          AND CAST(encounter_id AS INTEGER) IN (SELECT encounter_id FROM encounter)
          AND staff_id IS NOT NULL 
          AND staff_id != ''
          AND CAST(staff_id AS INTEGER) > 0
          AND CAST(staff_id AS INTEGER) IN (SELECT staff_id FROM staff)
          AND assignment_role IS NOT NULL AND assignment_role != ''
          AND assigned_ts IS NOT NULL AND assigned_ts != ''
          AND assignment_role NOT IN ('Attending', 'Resident', 'PA', 'NP', 'RN', 'Tech', 'Registration'));

DROP TABLE temp_staff_assignment;

-- Re-enable foreign keys
PRAGMA foreign_keys = ON;

-- ============================================
-- Verify imports and show cleaning summary
-- ============================================

SELECT '=== Import Summary ===' as summary;
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

SELECT '' as separator;
SELECT '=== Cleaning Log Summary ===' as summary;
SELECT table_name, issue_type, issue_description, SUM(rows_affected) as total_rows_affected
FROM cleaning_log
GROUP BY table_name, issue_type, issue_description
ORDER BY table_name, issue_type;

SELECT '' as separator;
SELECT '=== Total Rows Cleaned by Table ===' as summary;
SELECT table_name, SUM(rows_affected) as total_cleaned
FROM cleaning_log
GROUP BY table_name
ORDER BY total_cleaned DESC;
