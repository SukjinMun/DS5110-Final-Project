#!/usr/bin/env python3
# Stdlib-only ED CSV generator (no numpy/pandas required)
# Produces:
# 1) patient.csv  2) staff.csv  3) encounter.csv  4) encounter_payor.csv
# 5) vitals.csv   6) diagnosis.csv  7) staff_assignment.csv

import csv, random, string
from datetime import datetime, timedelta

# ---------- CONFIG ----------
SEED = 7
YEAR = 2025
N_PATIENTS = 4000
N_STAFF = 200
N_ENCOUNTERS = 8000
AVG_VITALS_PER_ENC = 1.6
AVG_DIAG_PER_ENC = 1.4
AVG_ASSIGN_PER_ENC = 1.6
OUT_DIR = "."

random.seed(SEED)

# ---------- Helpers ----------
def dt_to_str(dt): return dt.strftime("%Y-%m-%d %H:%M:%S")
def date_to_str(dt): return dt.strftime("%Y-%m-%d")
def weighted_choice(items, weights): return random.choices(items, weights=weights, k=1)[0]
def random_string(n=10, alphabet=string.ascii_uppercase+string.digits): return ''.join(random.choice(alphabet) for _ in range(n))
def clamp(x, lo, hi): return max(lo, min(hi, x))

def poisson_knuth(lam):
    # Knuth algorithm for Poisson(lam)
    from math import exp
    L = exp(-lam)
    k = 0
    p = 1.0
    while p > L:
        k += 1
        p *= random.random()
    return k - 1

def rand_datetime(start, end):
    delta = end - start
    rsec = random.randrange(int(delta.total_seconds()))
    return start + timedelta(seconds=rsec)

def birthdate_from_age(year, age):
    today = datetime(year, 12, 31)
    return date_to_str(today - timedelta(days=int(age*365.25) + random.randint(0, 364)))

# windows & dists
year_start = datetime(YEAR, 1, 1, 0, 0, 0)
year_end   = datetime(YEAR, 12, 31, 23, 59, 59)

ESI_VALUES, ESI_WEIGHTS = [1,2,3,4,5], [0.02,0.13,0.55,0.22,0.08]
ARR_KEYS, ARR_W = ["Walk-in","EMS","Transfer"], [0.75,0.20,0.05]
DISPO_KEYS, DISPO_W = ["DISCH_HOME","ADMIT_INPT","ADMIT_OBS","TRANSFER","LWBS","AMA","EXPIRED"], [0.75,0.15,0.05,0.02,0.02,0.009,0.001]
REF_KEYS, REF_W = ["PCP","CARD","GENSURG","ORTHO","SOCWORK","NONE"], [0.50,0.08,0.05,0.06,0.05,0.26]
CHIEF = ["Chest pain","Shortness of breath","Abdominal pain","Head injury","Fever","Cough","Back pain","Fall","Laceration","Weakness","Dizziness","Nausea/Vomiting"]
PAYOR_KEYS, PAYOR_W = ["Medicare","Medicaid","Blue Cross","Aetna","United","Cigna","Self-pay","Other"], [0.25,0.20,0.18,0.10,0.10,0.07,0.07,0.03]
PAYOR_TYPE = {"Medicare":"Public","Medicaid":"Public","Blue Cross":"Private","Aetna":"Private","United":"Private","Cigna":"Private","Self-pay":"Self","Other":"Private"}
DIAG_CODES = ["R07.9","J06.9","S09.90XA","N39.0","I10"]
STAFF_ROLES = ["MD_ATT","MD_RES","PA","NP","RN","TECH","REG"]
ASSIGN_ROLE_MAP = {"MD_ATT":"Attending","MD_RES":"Resident","PA":"PA","NP":"NP","RN":"RN","TECH":"Tech","REG":"Registration"}
ESI_WAIT_MEAN = {1:5, 2:15, 3:45, 4:90, 5:120}

def random_arrival_time():
    d = rand_datetime(year_start, year_end)
    if random.random() < 0.5:  # evening-ish bias
        hour = weighted_choice(list(range(24)),
                               [2,2,2,2,2,2,3,4,6,6,6,6,6,6,8,9,9,9,8,6,5,4,3,2])
        d = d.replace(hour=hour, minute=random.randint(0,59), second=random.randint(0,59))
    if random.random() < 0.2:   # weekend bias
        wk = d.weekday()
        if wk < 5:
            d = d + timedelta(days=random.choice([5 - wk, 6 - wk]))
    return d

# ---------- 1) patient.csv ----------
patients = []
for pid in range(1, N_PATIENTS+1):
    bucket = weighted_choice(["child","young_adult","middle","senior"], [0.22,0.34,0.24,0.20])
    lo, hi = {"child":(0,18),"young_adult":(19,44),"middle":(45,64),"senior":(65,95)}[bucket]
    age = random.randint(lo, hi)
    patients.append({
        "patient_id": pid,
        "dob": birthdate_from_age(YEAR, age),
        "sex_at_birth": weighted_choice(["Male","Female"], [0.49,0.51]),
        "gender_identity": weighted_choice(["Male","Female","Non-binary","Prefer not to say"], [0.47,0.47,0.03,0.03]),
        "zip_code": f"{random.randint(10000,99999)}"
    })

# ---------- 2) staff.csv ----------
staff = []
for sid in range(1, N_STAFF+1):
    staff.append({
        "staff_id": sid,
        "first_name": f"F{sid}",
        "last_name": f"L{sid}",
        "role_code": random.choice(STAFF_ROLES),
        "department": "ED",
        "is_active": 1 if random.random() < 0.92 else 0
    })

# ---------- 3) encounter.csv ----------
encounters = []
for eid in range(1, N_ENCOUNTERS+1):
    pid = random.randint(1, N_PATIENTS)
    arr = random_arrival_time()
    triage_start = arr + timedelta(minutes=random.randint(0, 20))
    triage_end = triage_start + timedelta(minutes=random.randint(5, 20))
    esi = weighted_choice(ESI_VALUES, ESI_WEIGHTS)

    # provider start with Gaussian around mean
    wait = max(0, int(random.gauss(ESI_WAIT_MEAN[esi], 10)))
    provider_start = triage_end + timedelta(minutes=wait)

    # decision and departure
    base_eval = max(10, int(random.gauss(60 + (5 - esi) * 20, 20)))
    dispo_decision = provider_start + timedelta(minutes=base_eval)
    base_dep = max(5, int(random.gauss(30 + (5 - esi) * 10, 15)))
    departure = dispo_decision + timedelta(minutes=base_dep)

    dispo = weighted_choice(DISPO_KEYS, DISPO_W)
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
        "arrival_mode": weighted_choice(ARR_KEYS, ARR_W),
        "chief_complaint": random.choice(CHIEF),
        "esi_level": esi,
        "disposition_code": dispo,
        "referral_code": weighted_choice(REF_KEYS, REF_W),
        "left_without_being_seen": lwbs,
        "notes": ""
    })

# ---------- 4) encounter_payor.csv ----------
payors = []
for eid in range(1, N_ENCOUNTERS+1):
    name = weighted_choice(PAYOR_KEYS, PAYOR_W)
    payors.append({
        "encounter_id": eid,
        "payor_name": name,
        "payor_type": PAYOR_TYPE[name],
        "member_id": "" if name == "Self-pay" else random_string(random.randint(8,12))
    })

# ---------- 5) vitals.csv ----------
vitals = []
vital_id = 1
for e in encounters:
    eid = e["encounter_id"]
    arr = datetime.strptime(e["arrival_ts"], "%Y-%m-%d %H:%M:%S")
    dep = datetime.strptime(e["departure_ts"], "%Y-%m-%d %H:%M:%S")
    n = max(0, poisson_knuth(AVG_VITALS_PER_ENC))
    for _ in range(n):
        if dep <= arr:
            taken = arr
        else:
            minutes = int((dep - arr).total_seconds() // 60)
            off = random.randint(0, max(1, minutes))
            taken = arr + timedelta(minutes=off)
        vitals.append({
            "vital_id": vital_id,
            "encounter_id": eid,
            "taken_ts": dt_to_str(taken),
            "heart_rate": clamp(int(random.gauss(85, 18)), 40, 200),
            "systolic_bp": clamp(int(random.gauss(128, 22)), 90, 200),
            "diastolic_bp": clamp(int(random.gauss(78, 12)), 50, 120),
            "respiratory_rate": clamp(int(random.gauss(17, 4)), 10, 30),
            "temperature_c": round(clamp(random.gauss(37.0, 0.6), 35.5, 40.5), 1),
            "spo2": clamp(int(random.gauss(97, 3)), 80, 100),
            "pain_score": clamp(int(random.gauss(4, 3)), 0, 10)
        })
        vital_id += 1

# ---------- 6) diagnosis.csv ----------
diagnosis = []
for e in encounters:
    eid = e["encounter_id"]
    lwbs = e["left_without_being_seen"]
    base_n = poisson_knuth(AVG_DIAG_PER_ENC)
    n = 0 if lwbs == 1 else max(1, base_n)
    chosen = random.sample(DIAG_CODES, k=min(n, len(DIAG_CODES)))
    for j, code in enumerate(chosen):
        diagnosis.append({"encounter_id": eid, "code": code, "is_primary": 1 if j == 0 else 0})

# ---------- 7) staff_assignment.csv ----------
staff_assign = []
for e in encounters:
    eid = e["encounter_id"]
    arr = datetime.strptime(e["arrival_ts"], "%Y-%m-%d %H:%M:%S")
    dep = datetime.strptime(e["departure_ts"], "%Y-%m-%d %H:%M:%S")
    n = max(0, poisson_knuth(AVG_ASSIGN_PER_ENC))
    if e["left_without_being_seen"] == 0 and n == 0: n = 1
    for _ in range(n):
        sid = random.randint(1, N_STAFF)
        arole = ASSIGN_ROLE_MAP.get(random.choice(STAFF_ROLES), "RN")
        if dep <= arr:
            assigned_ts, released_ts = arr, dep
        else:
            total = int((dep - arr).total_seconds() // 60)
            start = random.randint(0, max(1, total - 5))
            end = random.randint(start+1, max(start+2, total))
            assigned_ts = arr + timedelta(minutes=start)
            released_ts = arr + timedelta(minutes=end)
        staff_assign.append({
            "encounter_id": eid,
            "staff_id": sid,
            "assignment_role": arole,
            "assigned_ts": dt_to_str(assigned_ts),
            "released_ts": dt_to_str(released_ts)
        })

# ---------- Write CSVs ----------
def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

write_csv(f"{OUT_DIR}/patient.csv",
          ["patient_id","dob","sex_at_birth","gender_identity","zip_code"], patients)
write_csv(f"{OUT_DIR}/staff.csv",
          ["staff_id","first_name","last_name","role_code","department","is_active"], staff)
write_csv(f"{OUT_DIR}/encounter.csv",
          ["encounter_id","patient_id","arrival_ts","triage_start_ts","triage_end_ts","provider_start_ts","dispo_decision_ts","departure_ts","arrival_mode","chief_complaint","esi_level","disposition_code","referral_code","left_without_being_seen","notes"], encounters)
write_csv(f"{OUT_DIR}/encounter_payor.csv",
          ["encounter_id","payor_name","payor_type","member_id"], payors)
write_csv(f"{OUT_DIR}/vitals.csv",
          ["vital_id","encounter_id","taken_ts","heart_rate","systolic_bp","diastolic_bp","respiratory_rate","temperature_c","spo2","pain_score"], vitals)
write_csv(f"{OUT_DIR}/diagnosis.csv",
          ["encounter_id","code","is_primary"], diagnosis)
write_csv(f"{OUT_DIR}/staff_assignment.csv",
          ["encounter_id","staff_id","assignment_role","assigned_ts","released_ts"], staff_assign)

print("CSV generation complete in:", OUT_DIR)
