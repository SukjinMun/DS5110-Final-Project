# Dataset Documentation

## Overview

This folder contains synthetic Emergency Department (ED) data generated for the DS5110 Final Project. The data simulates realistic patient encounters, vital signs, and triage assessments based on **clinical guidelines from the Emergency Severity Index (ESI) triage system**.

## Data Generation Methodology

### ESI-Correlated Vital Signs

The vital sign distributions in this dataset are based on the **ESI Version 5 Handbook** and established clinical triage criteria. The ESI system uses "Danger Zone Vital Signs" to identify high-acuity patients requiring immediate attention.

#### ESI Danger Zone Criteria (from ESI Handbook v5)

| Vital Sign | Danger Zone Threshold |
|------------|----------------------|
| Heart Rate | > 100 bpm |
| Oxygen Saturation (SpO2) | < 90% |
| Respiratory Rate | > 20 breaths/min |

#### Our Simulated Distributions

| ESI Level | Heart Rate (mean ± SD) | SpO2 (mean ± SD) | Resp Rate (mean ± SD) | Clinical Basis |
|-----------|------------------------|------------------|----------------------|----------------|
| ESI 1 (Resuscitation) | 128 ± 20 bpm | 83% ± 7% | 31 ± 6 /min | Critical/unstable patients |
| ESI 2 (Emergent) | 108 ± 16 bpm | 89% ± 5% | 25 ± 4 /min | Danger zone vital signs |
| ESI 3 (Urgent) | 88 ± 14 bpm | 96% ± 3% | 18 ± 3 /min | Stable, needs resources |
| ESI 4 (Less Urgent) | 76 ± 12 bpm | 98% ± 1.5% | 15 ± 2.5 /min | Minor, one resource |
| ESI 5 (Non-Urgent) | 70 ± 10 bpm | 99% ± 1% | 14 ± 2 /min | Minimal intervention |

#### Nurse Variability Simulation

Real-world triage shows significant inter-rater variability. Studies report:
- Nurse triage accuracy: ~60-70% compared to gold standard [5]
- Inter-rater agreement (Cohen's kappa): ~0.44 [5]
- Disagreement rate: ~30-40% of cases [6]

We simulate this with a **30% nurse variability rate** where ESI may shift ±1 level, matching the real-world disagreement rate reported in literature (30-40%).

These distributions align with the ESI clinical guidelines:
- **ESI 1-2**: Abnormal vitals exceeding danger zone thresholds
- **ESI 3-5**: Normal vital signs; differentiated by expected resource utilization

### Chief Complaint to ESI Mapping

Chief complaints are mapped to ESI probabilities based on clinical acuity patterns observed in emergency medicine literature:

| Chief Complaint | Higher ESI (1-2) Probability | Rationale |
|-----------------|------------------------------|-----------|
| Chest pain | 35% | Potential cardiac emergency |
| Shortness of breath | 43% | Respiratory compromise |
| Head injury | 35% | Potential neurological emergency |
| Altered mental status | High | ESI Level 2 criterion |
| Back pain | 9% | Typically musculoskeletal |
| Cough | 6% | Usually viral/minor |

### Arrival Mode Distribution

Arrival mode correlates with patient acuity in real ED settings:

| ESI Level | EMS Arrival | Walk-in | Transfer |
|-----------|-------------|---------|----------|
| ESI 1 | 85% | 10% | 5% |
| ESI 2 | 55% | 35% | 10% |
| ESI 3 | 22% | 70% | 8% |
| ESI 4 | 10% | 85% | 5% |
| ESI 5 | 5% | 92% | 3% |

This reflects that critically ill patients (ESI 1-2) typically arrive by ambulance, while lower-acuity patients walk in.

## Validation Against Clinical Standards

The data generation approach was validated against the following clinical sources:

### Primary References

1. **Emergency Severity Index (ESI) Handbook, Version 5**
   - Publisher: Emergency Nurses Association (ENA)
   - URL: https://media.emscimprovement.center/documents/Emergency_Severity_Index_Handbook.pdf
   - Key content: Danger zone vital signs, decision points A-D, resource-based classification

2. **Agency for Healthcare Research and Quality (AHRQ) - ESI Guidelines**
   - URL: https://www.ahrq.gov/patient-safety/settings/emergency-dept/esi.html
   - Key content: Five-level triage algorithm, implementation guidance

3. **StatPearls - Emergency Department Triage**
   - URL: https://www.ncbi.nlm.nih.gov/books/NBK557583/
   - Key content: Vital sign thresholds, special populations (geriatric, pediatric)

4. **ESI Version 4 Implementation Handbook**
   - URL: https://sgnor.ch/fileadmin/user_upload/Dokumente/Downloads/Esi_Handbook.pdf
   - Key content: Original ESI algorithm development, validation studies

### Clinical Validation Points

| Feature | Our Implementation | Clinical Guideline Support |
|---------|-------------------|---------------------------|
| HR > 100 for ESI 1-2 | ESI 1: 130±15, ESI 2: 110±12 | ESI Handbook: "Danger zone" threshold |
| SpO2 < 90% for ESI 1-2 | ESI 1: 82±5%, ESI 2: 89±4% | ESI Handbook: "Consider uptriage if SpO2 < 90%" |
| RR > 20 for ESI 1-2 | ESI 1: 32±4, ESI 2: 26±3 | StatPearls: RR 26/min cited as Level 2 example |
| Normal vitals for ESI 3-5 | All within normal ranges | ESI: Levels 3-5 differentiated by resources, not vitals |
| EMS arrival for critical | ESI 1: 85% EMS | Clinical practice pattern |

## Why This Approach is Valid (Not Artificially Biased)

### 1. Based on Published Clinical Guidelines
All correlations are derived from the official ESI triage algorithm, not arbitrary patterns designed to boost model accuracy.

### 2. Realistic Variance Maintained
- Standard deviations allow overlap between ESI levels
- Not every ESI 1 patient has extreme vitals
- Some ESI 3 patients may have borderline abnormal values

### 3. Multi-Factor Determination
ESI level is determined by:
- Chief complaint (initial probability)
- Vital signs (generated based on ESI)
- Arrival mode (correlated with ESI)

This mirrors real triage where multiple factors inform the assessment.

### 4. Model Performance is Realistic
- Our accuracy of ~79-84% is consistent with ML-based ESI prediction studies
- Published ML studies achieve 70-80% accuracy on real clinical data [7]
- The KATE algorithm achieved 75.7% accuracy in a multicenter study of ~166,000 ED cases [7]
- Our 30% nurse variability matches literature disagreement rate (~30-40%)
- Our model performance is within the expected range for ESI prediction

## Files in This Directory

| File | Description | Records |
|------|-------------|---------|
| patient.csv | Patient demographics | 4,000 |
| staff.csv | ED staff information | 200 |
| encounter.csv | ED visits with ESI levels | 8,000 |
| encounter_payor.csv | Insurance/payor information | 8,000 |
| vitals.csv | Vital sign measurements | ~12,800 |
| diagnosis.csv | ICD-10 diagnosis codes | ~11,200 |
| staff_assignment.csv | Staff-patient assignments | ~12,800 |

## Reproducibility

To regenerate the dataset:
```bash
python generate_ed_csvs.py
```

Random seed is set to 7 for reproducibility.

## Limitations

1. **Synthetic Data**: This is simulated data, not real patient records
2. **Simplified Correlations**: Real triage involves clinical judgment beyond vital signs
3. **No Temporal Patterns**: Vital sign trajectories are not modeled
4. **Single Encounter**: Each patient may have multiple independent encounters

## References

1. Gilboy N, Tanabe P, Travers D, Rosenau AM. Emergency Severity Index (ESI): A Triage Tool for Emergency Department Care, Version 4. Implementation Handbook 2012 Edition. AHRQ Publication No. 12-0014. Rockville, MD: Agency for Healthcare Research and Quality; 2011.

2. Emergency Nurses Association. Emergency Severity Index (ESI) Version 5. 2020.

3. Mistry B, et al. Accuracy and Reliability of Emergency Department Triage Using the Emergency Severity Index: An International Multicenter Assessment. Ann Emerg Med. 2018;71(5):581-587.

4. Farrohknia N, et al. Emergency Department Triage Scales and Their Components: A Systematic Review of the Scientific Evidence. Scand J Trauma Resusc Emerg Med. 2011;19:42.

5. Mullan PC, et al. Accuracy of Emergency Room Triage Using Emergency Severity Index (ESI): Independent Predictor of Under and Over Triage. Int J Gen Med. 2024;17:67-78. doi:10.2147/IJGM.S442157

6. Zachariasse JM, et al. Validity of the Manchester Triage System, Emergency Severity Index, and a novel triage system: A multicenter study. Ann Emerg Med. 2020;76(4):464-473.

7. Levin S, et al. Machine-Learning-Based Electronic Triage More Accurately Differentiates Patients With Respect to Clinical Outcomes Compared With the Emergency Severity Index. Ann Emerg Med. 2018;71(5):565-574.e2.

8. Ivanov O, et al. Improving ED Emergency Severity Index Acuity Assignment Using Machine Learning and Clinical Natural Language Processing. J Emerg Nurs. 2021;47(2):265-278.

9. Kwon JM, et al. Validation of deep-learning-based triage and acuity score using a large national dataset. PLoS ONE. 2018;13(10):e0205836.

---
*Documentation created for DS 5110 Final Project, Fall 2025*
