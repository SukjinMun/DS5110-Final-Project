"""
API endpoints for Emergency Department Analysis System

Provides RESTful API for frontend and statistical analysis.
"""

from flask import Blueprint, jsonify, request
from sqlalchemy import func, desc
from functools import wraps
from config.database import get_session, db_session
from models.orm_models import Patient, Staff, Encounter, EncounterPayor, Vitals, Diagnosis, StaffAssignment

api_bp = Blueprint('api', __name__)

def with_session(f):
    """Decorator to ensure database session is properly closed"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        finally:
            # Ensure session is removed/closed (if db_session is initialized)
            if db_session is not None:
                try:
                    db_session.remove()
                except Exception:
                    pass  # Ignore errors during cleanup
    return decorated_function

@api_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'API is running'
    })

@api_bp.route('/encounters', methods=['GET'])
@with_session
def get_encounters():
    """Get all encounters with optional filtering"""
    session = get_session()
    try:
        # Query parameters for filtering
        esi_level = request.args.get('esi_level', type=int)
        disposition = request.args.get('disposition')
        limit = request.args.get('limit', default=100, type=int)
        offset = request.args.get('offset', default=0, type=int)

        query = session.query(Encounter)

        # Apply filters
        if esi_level:
            query = query.filter(Encounter.esi_level == esi_level)
        if disposition:
            query = query.filter(Encounter.disposition_code == disposition)

        # Pagination
        total = query.count()
        encounters = query.limit(limit).offset(offset).all()

        return jsonify({
            'total': total,
            'limit': limit,
            'offset': offset,
            'data': [enc.to_dict() for enc in encounters]
        })
    finally:
        session.close()

@api_bp.route('/encounters/<int:encounter_id>', methods=['GET'])
@with_session
def get_encounter_detail(encounter_id):
    """Get detailed encounter information with related data"""
    session = get_session()
    try:
        encounter = session.query(Encounter).filter(Encounter.encounter_id == encounter_id).first()

        if not encounter:
            return jsonify({'error': 'Encounter not found'}), 404

        # Get related data
        payor = session.query(EncounterPayor).filter(EncounterPayor.encounter_id == encounter_id).first()
        vitals = session.query(Vitals).filter(Vitals.encounter_id == encounter_id).all()
        diagnoses = session.query(Diagnosis).filter(Diagnosis.encounter_id == encounter_id).all()
        staff = session.query(StaffAssignment).filter(StaffAssignment.encounter_id == encounter_id).all()

        return jsonify({
            'encounter': encounter.to_dict(),
            'payor': payor.to_dict() if payor else None,
            'vitals': [v.to_dict() for v in vitals],
            'diagnoses': [d.to_dict() for d in diagnoses],
            'staff_assignments': [s.to_dict() for s in staff]
        })
    finally:
        session.close()

@api_bp.route('/patients/<int:patient_id>', methods=['GET'])
@with_session
def get_patient(patient_id):
    """Get patient information and encounter history"""
    session = get_session()
    try:
        patient = session.query(Patient).filter(Patient.patient_id == patient_id).first()

        if not patient:
            return jsonify({'error': 'Patient not found'}), 404

        encounters = session.query(Encounter).filter(Encounter.patient_id == patient_id).all()

        return jsonify({
            'patient': patient.to_dict(),
            'encounter_count': len(encounters),
            'encounters': [enc.to_dict() for enc in encounters]
        })
    finally:
        session.close()

@api_bp.route('/statistics/overview', methods=['GET'])
@with_session
def get_statistics_overview():
    """Get overall ED statistics"""
    session = get_session()

    # Basic counts
    total_encounters = session.query(func.count(Encounter.encounter_id)).scalar()
    total_patients = session.query(func.count(Patient.patient_id)).scalar()
    total_staff = session.query(func.count(Staff.staff_id)).scalar()

    # ESI distribution
    esi_dist = session.query(
        Encounter.esi_level,
        func.count(Encounter.encounter_id).label('count')
    ).group_by(Encounter.esi_level).all()

    # Disposition distribution
    dispo_dist = session.query(
        Encounter.disposition_code,
        func.count(Encounter.encounter_id).label('count')
    ).group_by(Encounter.disposition_code).all()

    # Arrival mode distribution
    arrival_dist = session.query(
        Encounter.arrival_mode,
        func.count(Encounter.encounter_id).label('count')
    ).group_by(Encounter.arrival_mode).all()

    return jsonify({
        'totals': {
            'encounters': total_encounters,
            'patients': total_patients,
            'staff': total_staff
        },
        'esi_distribution': [{'level': level, 'count': count} for level, count in esi_dist],
        'disposition_distribution': [{'code': code, 'count': count} for code, count in dispo_dist],
        'arrival_mode_distribution': [{'mode': mode, 'count': count} for mode, count in arrival_dist]
    })

@api_bp.route('/statistics/esi', methods=['GET'])
def get_esi_statistics():
    """Get ESI-level statistics"""
    session = get_session()

    esi_stats = []

    for esi in range(1, 6):
        count = session.query(func.count(Encounter.encounter_id)).filter(
            Encounter.esi_level == esi
        ).scalar()

        # Disposition breakdown for this ESI level
        dispositions = session.query(
            Encounter.disposition_code,
            func.count(Encounter.encounter_id).label('count')
        ).filter(Encounter.esi_level == esi).group_by(Encounter.disposition_code).all()

        # LWBS rate for this ESI level
        lwbs_count = session.query(func.count(Encounter.encounter_id)).filter(
            Encounter.esi_level == esi,
            Encounter.left_without_being_seen == 1
        ).scalar()

        esi_stats.append({
            'esi_level': esi,
            'total_count': count,
            'lwbs_count': lwbs_count,
            'lwbs_rate': lwbs_count / count if count > 0 else 0,
            'dispositions': [{'code': code, 'count': cnt} for code, cnt in dispositions]
        })

    return jsonify({'esi_statistics': esi_stats})

@api_bp.route('/statistics/vitals', methods=['GET'])
@with_session
def get_vitals_statistics():
    """Get vital signs statistics"""
    session = get_session()

    vitals_stats = session.query(
        func.avg(Vitals.heart_rate).label('avg_hr'),
        func.avg(Vitals.systolic_bp).label('avg_sbp'),
        func.avg(Vitals.diastolic_bp).label('avg_dbp'),
        func.avg(Vitals.respiratory_rate).label('avg_rr'),
        func.avg(Vitals.temperature_c).label('avg_temp'),
        func.avg(Vitals.spo2).label('avg_spo2'),
        func.avg(Vitals.pain_score).label('avg_pain')
    ).first()

    return jsonify({
        'average_vitals': {
            'heart_rate': round(vitals_stats.avg_hr, 1) if vitals_stats.avg_hr else None,
            'systolic_bp': round(vitals_stats.avg_sbp, 1) if vitals_stats.avg_sbp else None,
            'diastolic_bp': round(vitals_stats.avg_dbp, 1) if vitals_stats.avg_dbp else None,
            'respiratory_rate': round(vitals_stats.avg_rr, 1) if vitals_stats.avg_rr else None,
            'temperature_c': round(vitals_stats.avg_temp, 1) if vitals_stats.avg_temp else None,
            'spo2': round(vitals_stats.avg_spo2, 1) if vitals_stats.avg_spo2 else None,
            'pain_score': round(vitals_stats.avg_pain, 1) if vitals_stats.avg_pain else None
        }
    })

@api_bp.route('/statistics/payor', methods=['GET'])
@with_session
def get_payor_statistics():
    """Get payor distribution statistics"""
    session = get_session()

    # Payor name distribution
    payor_name_dist = session.query(
        EncounterPayor.payor_name,
        func.count(EncounterPayor.encounter_id).label('count')
    ).group_by(EncounterPayor.payor_name).all()

    # Payor type distribution
    payor_type_dist = session.query(
        EncounterPayor.payor_type,
        func.count(EncounterPayor.encounter_id).label('count')
    ).group_by(EncounterPayor.payor_type).all()

    return jsonify({
        'payor_name_distribution': [{'name': name, 'count': count} for name, count in payor_name_dist],
        'payor_type_distribution': [{'type': ptype, 'count': count} for ptype, count in payor_type_dist]
    })

@api_bp.route('/statistics/diagnoses', methods=['GET'])
@with_session
def get_diagnosis_statistics():
    """Get diagnosis code statistics"""
    session = get_session()

    # Top diagnoses
    diagnosis_dist = session.query(
        Diagnosis.code,
        func.count(Diagnosis.encounter_id).label('count')
    ).group_by(Diagnosis.code).order_by(desc('count')).limit(10).all()

    # Primary diagnosis count
    primary_count = session.query(func.count(Diagnosis.encounter_id)).filter(
        Diagnosis.is_primary == 1
    ).scalar()

    total_diagnoses = session.query(func.count(Diagnosis.encounter_id)).scalar()

    return jsonify({
        'total_diagnoses': total_diagnoses,
        'primary_diagnosis_count': primary_count,
        'top_diagnoses': [{'code': code, 'count': count} for code, count in diagnosis_dist]
    })

@api_bp.route('/chief-complaints', methods=['GET'])
@with_session
def get_chief_complaints():
    """Get chief complaint distribution"""
    session = get_session()

    complaints = session.query(
        Encounter.chief_complaint,
        func.count(Encounter.encounter_id).label('count')
    ).group_by(Encounter.chief_complaint).order_by(desc('count')).all()

    return jsonify({
        'chief_complaints': [{'complaint': cc, 'count': count} for cc, count in complaints]
    })

@api_bp.route('/staff', methods=['GET'])
@with_session
def get_staff():
    """Get staff list"""
    session = get_session()

    active_only = request.args.get('active_only', default='false').lower() == 'true'

    query = session.query(Staff)

    if active_only:
        query = query.filter(Staff.is_active == 1)

    staff = query.all()

    return jsonify({
        'total': len(staff),
        'staff': [s.to_dict() for s in staff]
    })

# Note: Wait time endpoints will be implemented after date format is fixed
# Endpoints to be added:
# - /api/wait-times
# - /api/statistics/wait-times-by-esi
# - /api/statistics/length-of-stay
