PRAGMA foreign_keys = ON;

-- =========
-- Entities
-- =========
CREATE TABLE IF NOT EXISTS patient (
  patient_id       INTEGER PRIMARY KEY,
  dob              TEXT,         -- ISO date "YYYY-MM-DD"
  sex_at_birth     TEXT,
  gender_identity  TEXT,
  zip_code         TEXT
);

CREATE TABLE IF NOT EXISTS staff (
  staff_id    INTEGER PRIMARY KEY,
  first_name  TEXT NOT NULL,
  last_name   TEXT NOT NULL,
  role_code   TEXT NOT NULL,     -- MD_ATT, RN, etc.
  department  TEXT,
  is_active   INTEGER NOT NULL   -- 1/0
);

CREATE TABLE IF NOT EXISTS encounter (
  encounter_id         INTEGER PRIMARY KEY,
  patient_id           INTEGER NOT NULL,
  arrival_ts           TEXT,     -- ISO datetime
  triage_start_ts      TEXT,
  triage_end_ts        TEXT,
  provider_start_ts    TEXT,
  dispo_decision_ts    TEXT,
  departure_ts         TEXT,
  arrival_mode         TEXT,     -- Walk-in, EMS, Transfer
  chief_complaint      TEXT,
  esi_level            INTEGER,  -- 1–5
  disposition_code     TEXT,     -- DISCH_HOME, ADMIT_INPT, etc.
  referral_code        TEXT,     -- PCP, CARD, ...
  left_without_being_seen INTEGER DEFAULT 0, -- 1/0
  notes                TEXT,
  FOREIGN KEY (patient_id) REFERENCES patient(patient_id)
);

CREATE TABLE IF NOT EXISTS encounter_payor (
  encounter_id  INTEGER PRIMARY KEY,
  payor_name    TEXT,            -- Medicare, Blue Cross, Self-pay...
  payor_type    TEXT,            -- Public, Private, Self
  member_id     TEXT,
  FOREIGN KEY (encounter_id) REFERENCES encounter(encounter_id)
);

CREATE TABLE IF NOT EXISTS vitals (
  vital_id          INTEGER PRIMARY KEY,
  encounter_id      INTEGER NOT NULL,
  taken_ts          TEXT NOT NULL,
  heart_rate        INTEGER,
  systolic_bp       INTEGER,
  diastolic_bp      INTEGER,
  respiratory_rate  INTEGER,
  temperature_c     REAL,
  spo2              INTEGER,
  pain_score        INTEGER,
  FOREIGN KEY (encounter_id) REFERENCES encounter(encounter_id)
);

CREATE TABLE IF NOT EXISTS diagnosis (
  encounter_id  INTEGER NOT NULL,
  code          TEXT NOT NULL,      -- ICD-10 (e.g., R07.9)
  is_primary    INTEGER NOT NULL,   -- 1/0
  PRIMARY KEY (encounter_id, code),
  FOREIGN KEY (encounter_id) REFERENCES encounter(encounter_id)
);

CREATE TABLE IF NOT EXISTS staff_assignment (
  encounter_id    INTEGER NOT NULL,
  staff_id        INTEGER NOT NULL,
  assignment_role TEXT NOT NULL,    -- Attending, RN, Tech, ...
  assigned_ts     TEXT NOT NULL,
  released_ts     TEXT,
  PRIMARY KEY (encounter_id, staff_id, assignment_role, assigned_ts),
  FOREIGN KEY (encounter_id) REFERENCES encounter(encounter_id),
  FOREIGN KEY (staff_id)     REFERENCES staff(staff_id)
);

-- =========
-- Indexes
-- =========
CREATE INDEX IF NOT EXISTS idx_enc_patient   ON encounter(patient_id);
CREATE INDEX IF NOT EXISTS idx_enc_times     ON encounter(arrival_ts, departure_ts);
CREATE INDEX IF NOT EXISTS idx_enc_esi       ON encounter(esi_level);
CREATE INDEX IF NOT EXISTS idx_enc_dispo     ON encounter(disposition_code);
CREATE INDEX IF NOT EXISTS idx_vitals_enc_ts ON vitals(encounter_id, taken_ts);
CREATE INDEX IF NOT EXISTS idx_staff_assign  ON staff_assignment(staff_id, assigned_ts);

-- =========
-- Cleaning Log (for ETL process tracking)
-- =========
CREATE TABLE IF NOT EXISTS cleaning_log (
  log_id            INTEGER PRIMARY KEY AUTOINCREMENT,
  table_name        TEXT NOT NULL,
  issue_type        TEXT NOT NULL,  -- 'header_row', 'missing_value', 'out_of_range', 'invalid_fk', 'invalid_enum'
  issue_description TEXT,
  rows_affected     INTEGER,
  created_at        TEXT DEFAULT (datetime('now'))
);

-- =========
-- Analytics Views
-- =========
-- Wait time (minutes) from arrival -> provider
CREATE VIEW IF NOT EXISTS v_wait_time AS
SELECT e.encounter_id,
       e.esi_level,
       e.disposition_code,
       CAST((julianday(e.provider_start_ts) - julianday(e.arrival_ts)) * 1440 AS INTEGER) AS wait_min
FROM encounter e
WHERE e.provider_start_ts IS NOT NULL AND e.provider_start_ts <> ''
  AND e.arrival_ts        IS NOT NULL AND e.arrival_ts        <> '';

-- Length of stay (minutes) from arrival -> departure
CREATE VIEW IF NOT EXISTS v_los AS
SELECT e.encounter_id,
       e.esi_level,
       e.disposition_code,
       CAST((julianday(e.departure_ts) - julianday(e.arrival_ts)) * 1440 AS INTEGER) AS los_min
FROM encounter e
WHERE e.departure_ts IS NOT NULL AND e.departure_ts <> ''
  AND e.arrival_ts   IS NOT NULL AND e.arrival_ts   <> '';
