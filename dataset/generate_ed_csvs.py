#!/usr/bin/env python3
"""
Generate synthetic ED CSVs (first 7 datasets):
  1) patient.csv
  2) staff.csv
  3) encounter.csv
  4) encounter_payor.csv
  5) vitals.csv
  6) diagnosis.csv
  7) staff_assignment.csv

Edit the CONFIG section to change sizes or year.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import string

# -----------------------
# CONFIG
# -----------------------
SEED = 7
YEAR = 2025

N_PATIENTS = 4000
N_STAFF = 200
N_ENCOUNTERS = 8000

AVG_VITALS_PER_ENC = 1.6      # Poisson mean
AVG_DIAG_PER_ENC = 1.4        # Poisson mean
AVG_ASSIGN_PER_ENC = 1.6      # Poisson mean

OUT_DIR = "."  # current folder

np.random.seed(SEED)
random.seed(SEED)

# -----------------------
# Helpers
# -----------------------
def rand_date(start, end):
    delta = end - start
    rsec = random.randrange(int(delta.total_seconds()))
    return start + timedelta(seconds=rsec)

def weighted_choice(items, weights):
    return random.choices(items, weights=weights, k=1)[0]

def random_string(n=10, alphabet=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(alphabet) for _ in range(n))

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def dt_to_str(dt):
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def date_to_str(dt):
    return dt.strftime("%Y-%m-%d")

def birthdate_from_age(age):
    today = datetime(YEAR, 12, 31)
    bd = today - timedelta(days=int(age * 365.25)) - timedelta(days=random.randint(0, 364))
    return date_to_str(bd)

# Global windows
year_start = datetime(YEAR, 1, 1, 0, 0, 0)
year_end   = datetime(YEAR, 12, 31, 23, 59, 59)

# Distributions
ESI_DIST = {1: 0.02, 2: 0.13, 3: 0.55, 4: 0.22, 5: 0.08}
ARRIVAL_MODE_DIST = {"Walk-in": 0.75, "EMS": 0.20, "Transfer": 0.05}
DISPO_DIST = {
    "DISCH_HOME": 0.75,
    "ADMIT_INPT": 0.15,
    "ADMIT_OBS": 0.05,
    "TRANSFER": 0.02,
    "LWBS": 0.02,
    "AMA": 0.009,
    "EXPIRED": 0.001,
}
REFERRAL_DIST = {"PCP": 0.50, "CARD": 0.08, "GENSURG": 0.05, "ORTHO": 0.06, "SOCWORK": 0.05, "NONE": 0.26}
CHIEF_COMPLAINTS = [
    "Chest pain","Shortness of breath","Abdominal pain","Head injury","Fever",
    "Cough","Back pain","Fall","Laceration","Weakness","Dizziness","Nausea/Vomiting"
]
PAYOR_NAME_DIST = {
    "Medicare": 0.25, "Medicaid": 0.20, "Blue Cross": 0.18, "Aetna": 0.10,
    "United": 0.10, "Cigna": 0.07, "Self-pay": 0.07, "Other": 0.03
}
PAYOR_TYPE_MAP = {"Medicare": "Public","Medicaid":"Public","Blue Cross":"Private","Aetna":"Private",
                  "United":"Private","Cigna":"Private","Self-pay":"Self","Other":"Private"}
DIAG_CODES = ["R07.9","J06.9","S09.90XA","N39.0","I10"]

ESI_WAIT_MEAN = {1: 5, 2: 15, 3: 45, 4: 90, 5: 120}

def random_arrival_time():
    d = rand_date(year_start, year_end)
    # slight evening / weekend bias
    if random.random() < 0.5:
        hour = weighted_choice(list(range(24)),
                               [2,2,2,2,2,2,3,4,6,6,6,6,6,6,8,9,9,9,8,6,5,4,3,2])
        d = d.replace(hour=hour, minute=random.randint(0,59), second=random.randint(0,59))
    if random.random() < 0.2:
        weekday = d.weekday()
        if weekday < 5:
            d = d + timedelta(days=random.choice([5 - weekday, 6 - weekday]))
    return d

def save_df(df, name):
    path = f"{OUT_DIR}/{name}.csv"
    df.to_csv(path, index=False)
    print(f"wrote {path}")
    return path

def main():
    # 1) patient.csv
    patients = []
    for pid in range(1, N_PATIENTS+1):
        # age buckets
        bucket = weighted_choice(["child","young_adult","middle","senior"], [0.22,0.34,0.24,0.20])
        age = {"child": (0,18), "young_adult": (19,44), "middle": (45,64), "senior": (65,95)}[bucket]
        age = random.randint(*age)
        patients.append({
            "patient_id": pid,
            "dob": birthdate_from_age(age),
            "sex_at_birth": weighted_choice(["Male","Female"], [0.49, 0.51]),
            "gender_identity": weighted_choice(["Male","Female","Non-binary","Prefer not to say"], [0.47,0.47,0.03,0.03]),
            "zip_code": f"{random.randint(10000, 99999)}"
        })
    df_patient = pd.DataFrame(patients)

    # 2) staff.csv
    STAFF_ROLES = ["MD_ATT","MD_RES","PA","NP","RN","TECH","REG"]
    staff = [{
        "staff_id": sid,
        "first_name": f"F{sid}",
        "last_name": f"L{sid}",
        "role_code": random.choice(STAFF_ROLES),
        "department": "ED",
        "is_active": 1 if random.random() < 0.92 else 0
    } for sid in range(1, N_STAFF+1)]
    df_staff = pd.DataFrame(staff)

    # 3) encounter.csv
    encounters = []
    for eid in range(1, N_ENCOUNTERS+1):
        pid = random.randint(1, N_PATIENTS)
        arr = random_arrival_time()
        triage_start = arr + timedelta(minutes=random.randint(0, 20))
        triage_end = triage_start + timedelta(minutes=random.randint(5, 20))

        esi = weighted_choice([1,2,3,4,5], list(ESI_DIST.values()))
        provider_start = triage_end + timedelta(minutes=max(0, int(np.random.normal(ESI_WAIT_MEAN[esi], 10))))

        base_eval = int(np.random.normal(60 + (5 - esi) * 20, 20))
        dispo_decision = provider_start + timedelta(minutes=max(10, base_eval))

        base_dep = int(np.random.normal(30 + (5 - esi) * 10, 15))
        departure = dispo_decision + timedelta(minutes=max(5, base_dep))

        dispo = weighted_choice(list(DISPO_DIST.keys()), list(DISPO_DIST.values()))
        lwbs = 1 if dispo == "LWBS" else 0
        if lwbs:
            provider_start = None
            dispo_decision = None
            departure = triage_end + timedelta(minutes=random.randint(10, 60))

        encounters.append({
            "encounter_id": eid,
            "patient_id": pid,
            "arrival_ts": dt_to_str(arr),
            "triage_start_ts": dt_to_str(triage_start),
            "triage_end_ts": dt_to_str(triage_end),
            "provider_start_ts": dt_to_str(provider_start) if provider_start else "",
            "dispo_decision_ts": dt_to_str(dispo_decision) if dispo_decision else "",
            "departure_ts": dt_to_str(departure),
            "arrival_mode": weighted_choice(list(ARRIVAL_MODE_DIST.keys()), list(ARRIVAL_MODE_DIST.values())),
            "chief_complaint": random.choice(CHIEF_COMPLAINTS),
            "esi_level": esi,
            "disposition_code": dispo,
            "referral_code": weighted_choice(list(REFERRAL_DIST.keys()), list(REFERRAL_DIST.values())),
            "left_without_being_seen": lwbs,
            "notes": ""
        })
    df_encounter = pd.DataFrame(encounters)

    # 4) encounter_payor.csv
    payors = []
    for eid in range(1, N_ENCOUNTERS+1):
        name = weighted_choice(list(PAYOR_NAME_DIST.keys()), list(PAYOR_NAME_DIST.values()))
        payors.append({
            "encounter_id": eid,
            "payor_name": name,
            "payor_type": PAYOR_TYPE_MAP[name],
            "member_id": "" if name == "Self-pay" else random_string(random.randint(8,12))
        })
    df_payor = pd.DataFrame(payors)

    # 5) vitals.csv
    vitals_rows = []
    vital_id = 1
    for _, row in df_encounter.iterrows():
        eid = int(row["encounter_id"])
        arr = datetime.strptime(row["arrival_ts"], "%Y-%m-%d %H:%M:%S")
        dep = datetime.strptime(row["departure_ts"], "%Y-%m-%d %H:%M:%S")
        n = max(0, np.random.poisson(AVG_VITALS_PER_ENC))

        for _ in range(n):
            if (dep - arr).total_seconds() <= 0:
                taken = arr
            else:
                offset_min = random.randint(0, max(1, int((dep - arr).total_seconds() // 60)))
                taken = arr + timedelta(minutes=offset_min)

            vitals_rows.append({
                "vital_id": vital_id,
                "encounter_id": eid,
                "taken_ts": dt_to_str(taken),
                "heart_rate": clamp(int(np.random.normal(85, 18)), 40, 200),
                "systolic_bp": clamp(int(np.random.normal(128, 22)), 90, 200),
                "diastolic_bp": clamp(int(np.random.normal(78, 12)), 50, 120),
                "respiratory_rate": clamp(int(np.random.normal(17, 4)), 10, 30),
                "temperature_c": round(clamp(np.random.normal(37.0, 0.6), 35.5, 40.5), 1),
                "spo2": clamp(int(np.random.normal(97, 3)), 80, 100),
                "pain_score": clamp(int(np.random.normal(4, 3)), 0, 10)
            })
            vital_id += 1
    df_vitals = pd.DataFrame(vitals_rows)

    # 6) diagnosis.csv
    diag_rows = []
    for _, row in df_encounter.iterrows():
        eid = int(row["encounter_id"])
        lwbs = int(row["left_without_being_seen"])
        base_n = np.random.poisson(AVG_DIAG_PER_ENC)
        n = 0 if lwbs == 1 else max(1, base_n)
        chosen = random.sample(DIAG_CODES, k=min(n, len(DIAG_CODES)))
        for j, code in enumerate(chosen):
            diag_rows.append({"encounter_id": eid, "code": code, "is_primary": 1 if j == 0 else 0})
    df_diagnosis = pd.DataFrame(diag_rows)

    # 7) staff_assignment.csv
    assign_rows = []
    for _, row in df_encounter.iterrows():
        eid = int(row["encounter_id"])
        arr = datetime.strptime(row["arrival_ts"], "%Y-%m-%d %H:%M:%S")
        dep = datetime.strptime(row["departure_ts"], "%Y-%m-%d %H:%M:%S")

        n = max(0, np.random.poisson(AVG_ASSIGN_PER_ENC))
        if row["left_without_being_seen"] == 0 and n == 0:
            n = 1

        for _ in range(n):
            sid = random.randint(1, N_STAFF)
            role = df_staff.loc[df_staff["staff_id"] == sid, "role_code"].values[0]
            role_map = {"MD_ATT":"Attending","MD_RES":"Resident","PA":"PA","NP":"NP","RN":"RN","TECH":"Tech","REG":"Registration"}
            arole = role_map.get(role, "RN")

            if (dep - arr).total_seconds() <= 0:
                assigned_ts = arr
                released_ts = dep
            else:
                start_offset = random.randint(0, max(1, int((dep - arr).total_seconds() // 60) - 5))
                end_offset = random.randint(start_offset+1, max(start_offset+2, int((dep - arr).total_seconds() // 60)))
                assigned_ts = arr + timedelta(minutes=start_offset)
                released_ts = arr + timedelta(minutes=end_offset)

            assign_rows.append({
                "encounter_id": eid,
                "staff_id": sid,
                "assignment_role": arole,
                "assigned_ts": dt_to_str(assigned_ts),
                "released_ts": dt_to_str(released_ts)
            })
    df_staff_assignment = pd.DataFrame(assign_rows)

    # Save
    def save_df(df, name):
        path = f"{OUT_DIR}/{name}.csv"
        df.to_csv(path, index=False)
        print(f"wrote {path}")
        return path

    save_df(df_patient, "patient")
    save_df(df_staff, "staff")
    save_df(df_encounter, "encounter")
    save_df(df_payor, "encounter_payor")
    save_df(df_vitals, "vitals")
    save_df(df_diagnosis, "diagnosis")
    save_df(df_staff_assignment, "staff_assignment")

if __name__ == "__main__":
    main()
