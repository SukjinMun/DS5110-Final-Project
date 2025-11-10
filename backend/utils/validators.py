"""
Business logic validation and helper functions

Provides validation for API inputs and business rules.
"""

def validate_esi_level(esi_level):
    """Validate ESI level is between 1-5"""
    if not isinstance(esi_level, int):
        return False, "ESI level must be an integer"

    if esi_level < 1 or esi_level > 5:
        return False, "ESI level must be between 1 and 5"

    return True, None

def validate_disposition_code(code):
    """Validate disposition code"""
    valid_codes = [
        'DISCH_HOME', 'ADMIT_INPT', 'ADMIT_OBS',
        'TRANSFER', 'LWBS', 'AMA', 'EXPIRED'
    ]

    if code not in valid_codes:
        return False, f"Invalid disposition code. Must be one of: {', '.join(valid_codes)}"

    return True, None

def validate_arrival_mode(mode):
    """Validate arrival mode"""
    valid_modes = ['Walk-in', 'EMS', 'Transfer']

    if mode not in valid_modes:
        return False, f"Invalid arrival mode. Must be one of: {', '.join(valid_modes)}"

    return True, None

def validate_vital_signs(vitals):
    """Validate vital sign ranges"""
    errors = []

    if 'heart_rate' in vitals:
        hr = vitals['heart_rate']
        if hr < 30 or hr > 250:
            errors.append("Heart rate must be between 30-250 bpm")

    if 'systolic_bp' in vitals:
        sbp = vitals['systolic_bp']
        if sbp < 50 or sbp > 250:
            errors.append("Systolic BP must be between 50-250 mmHg")

    if 'diastolic_bp' in vitals:
        dbp = vitals['diastolic_bp']
        if dbp < 30 or dbp > 150:
            errors.append("Diastolic BP must be between 30-150 mmHg")

    if 'temperature_c' in vitals:
        temp = vitals['temperature_c']
        if temp < 30.0 or temp > 45.0:
            errors.append("Temperature must be between 30-45°C")

    if 'spo2' in vitals:
        spo2 = vitals['spo2']
        if spo2 < 50 or spo2 > 100:
            errors.append("SpO2 must be between 50-100%")

    if 'pain_score' in vitals:
        pain = vitals['pain_score']
        if pain < 0 or pain > 10:
            errors.append("Pain score must be between 0-10")

    if errors:
        return False, "; ".join(errors)

    return True, None

def calculate_admission_rate(encounters):
    """Calculate admission rate from encounters"""
    if not encounters:
        return 0.0

    admitted = sum(1 for enc in encounters
                  if enc.disposition_code in ['ADMIT_INPT', 'ADMIT_OBS'])

    return admitted / len(encounters)

def calculate_lwbs_rate(encounters):
    """Calculate left without being seen rate"""
    if not encounters:
        return 0.0

    lwbs = sum(1 for enc in encounters if enc.left_without_being_seen == 1)

    return lwbs / len(encounters)

def get_esi_description(esi_level):
    """Get text description for ESI level"""
    descriptions = {
        1: "Resuscitation - Immediate life-saving intervention required",
        2: "Emergent - High risk, severe pain/distress, or confusion/lethargy",
        3: "Urgent - Stable with multiple resources needed",
        4: "Less Urgent - Stable with one resource needed",
        5: "Non-Urgent - Fast track/clinic visit appropriate"
    }

    return descriptions.get(esi_level, "Unknown ESI level")
