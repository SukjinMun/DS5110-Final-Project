"""
API endpoints for Emergency Department Analysis System

Provides RESTful API for frontend and statistical analysis.
"""

from flask import Blueprint, jsonify, request
from sqlalchemy import func, desc, text
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
    """Get all encounters with optional filtering, search, and sorting"""
    session = get_session()
    try:
        # Query parameters for filtering
        esi_level = request.args.get('esi_level', type=int)
        disposition = request.args.get('disposition')
        limit = request.args.get('limit', default=100, type=int)
        offset = request.args.get('offset', default=0, type=int)
        
        # Search parameter
        search = request.args.get('search', type=str)
        
        # Sort parameters
        sort_by = request.args.get('sort_by', default='encounter_id', type=str)
        sort_order = request.args.get('sort_order', default='asc', type=str)

        query = session.query(Encounter)

        # Apply filters
        if esi_level:
            query = query.filter(Encounter.esi_level == esi_level)
        if disposition:
            query = query.filter(Encounter.disposition_code == disposition)
        
        # Apply search (searches across multiple fields)
        if search:
            search_term = f'%{search}%'
            # Use text() for casting to string in SQLite
            query = query.filter(
                (Encounter.chief_complaint.like(search_term)) |
                (Encounter.disposition_code.like(search_term)) |
                (Encounter.arrival_mode.like(search_term)) |
                (Encounter.referral_code.like(search_term)) |
                (Encounter.notes.like(search_term)) |
                (text(f"CAST(encounter.encounter_id AS TEXT)").like(search_term)) |
                (text(f"CAST(encounter.patient_id AS TEXT)").like(search_term))
            )

        # Apply sorting
        sort_column = None
        valid_sort_columns = {
            'encounter_id': Encounter.encounter_id,
            'patient_id': Encounter.patient_id,
            'arrival_ts': Encounter.arrival_ts,
            'esi_level': Encounter.esi_level,
            'chief_complaint': Encounter.chief_complaint,
            'disposition_code': Encounter.disposition_code,
            'departure_ts': Encounter.departure_ts,
        }
        
        if sort_by in valid_sort_columns:
            sort_column = valid_sort_columns[sort_by]
        else:
            sort_column = Encounter.encounter_id  # Default sort
        
        if sort_order.lower() == 'desc':
            query = query.order_by(desc(sort_column))
        else:
            query = query.order_by(sort_column)

        # Get total count before pagination
        total = query.count()
        
        # Apply pagination
        encounters = query.limit(limit).offset(offset).all()

        return jsonify({
            'total': total,
            'limit': limit,
            'offset': offset,
            'sort_by': sort_by,
            'sort_order': sort_order,
            'search': search,
            'data': [enc.to_dict() for enc in encounters]
        })
    finally:
        session.close()

@api_bp.route('/encounters', methods=['POST'])
@with_session
def create_encounter():
    """Create a new encounter"""
    session = get_session()
    try:
        data = request.get_json()
        
        # Validate required fields
        if 'patient_id' not in data:
            return jsonify({'error': 'patient_id is required'}), 400
        
        # Create new encounter
        encounter = Encounter(
            patient_id=data['patient_id'],
            arrival_ts=data.get('arrival_ts'),
            triage_start_ts=data.get('triage_start_ts'),
            triage_end_ts=data.get('triage_end_ts'),
            provider_start_ts=data.get('provider_start_ts'),
            dispo_decision_ts=data.get('dispo_decision_ts'),
            departure_ts=data.get('departure_ts'),
            arrival_mode=data.get('arrival_mode'),
            chief_complaint=data.get('chief_complaint'),
            esi_level=data.get('esi_level'),
            disposition_code=data.get('disposition_code'),
            referral_code=data.get('referral_code'),
            left_without_being_seen=data.get('left_without_being_seen', 0),
            notes=data.get('notes')
        )
        
        session.add(encounter)
        session.commit()
        session.refresh(encounter)
        
        return jsonify({
            'message': 'Encounter created successfully',
            'encounter': encounter.to_dict()
        }), 201
    except Exception as e:
        session.rollback()
        return jsonify({'error': str(e)}), 400
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

@api_bp.route('/encounters/<int:encounter_id>', methods=['PUT'])
@with_session
def update_encounter(encounter_id):
    """Update an existing encounter"""
    session = get_session()
    try:
        encounter = session.query(Encounter).filter(Encounter.encounter_id == encounter_id).first()
        
        if not encounter:
            return jsonify({'error': 'Encounter not found'}), 404
        
        data = request.get_json()
        
        # Update fields if provided
        if 'patient_id' in data:
            encounter.patient_id = data['patient_id']
        if 'arrival_ts' in data:
            encounter.arrival_ts = data['arrival_ts']
        if 'triage_start_ts' in data:
            encounter.triage_start_ts = data['triage_start_ts']
        if 'triage_end_ts' in data:
            encounter.triage_end_ts = data['triage_end_ts']
        if 'provider_start_ts' in data:
            encounter.provider_start_ts = data['provider_start_ts']
        if 'dispo_decision_ts' in data:
            encounter.dispo_decision_ts = data['dispo_decision_ts']
        if 'departure_ts' in data:
            encounter.departure_ts = data['departure_ts']
        if 'arrival_mode' in data:
            encounter.arrival_mode = data['arrival_mode']
        if 'chief_complaint' in data:
            encounter.chief_complaint = data['chief_complaint']
        if 'esi_level' in data:
            encounter.esi_level = data['esi_level']
        if 'disposition_code' in data:
            encounter.disposition_code = data['disposition_code']
        if 'referral_code' in data:
            encounter.referral_code = data['referral_code']
        if 'left_without_being_seen' in data:
            encounter.left_without_being_seen = data['left_without_being_seen']
        if 'notes' in data:
            encounter.notes = data['notes']
        
        session.commit()
        session.refresh(encounter)
        
        return jsonify({
            'message': 'Encounter updated successfully',
            'encounter': encounter.to_dict()
        })
    except Exception as e:
        session.rollback()
        return jsonify({'error': str(e)}), 400
    finally:
        session.close()

@api_bp.route('/encounters/<int:encounter_id>', methods=['DELETE'])
@with_session
def delete_encounter(encounter_id):
    """Delete an encounter"""
    session = get_session()
    try:
        encounter = session.query(Encounter).filter(Encounter.encounter_id == encounter_id).first()
        
        if not encounter:
            return jsonify({'error': 'Encounter not found'}), 404
        
        # Delete related records first (if cascade delete is not set up)
        session.query(EncounterPayor).filter(EncounterPayor.encounter_id == encounter_id).delete()
        session.query(Vitals).filter(Vitals.encounter_id == encounter_id).delete()
        session.query(Diagnosis).filter(Diagnosis.encounter_id == encounter_id).delete()
        session.query(StaffAssignment).filter(StaffAssignment.encounter_id == encounter_id).delete()
        
        session.delete(encounter)
        session.commit()
        
        return jsonify({
            'message': 'Encounter deleted successfully'
        })
    except Exception as e:
        session.rollback()
        return jsonify({'error': str(e)}), 400
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
    """Get diagnosis code statistics with optional date filtering

    Query parameters:
    - start_date: Filter encounters from this date (ISO format: YYYY-MM-DD)
    - end_date: Filter encounters until this date (ISO format: YYYY-MM-DD)
    - date: Single date filter (ISO format: YYYY-MM-DD) - shorthand for start_date=end_date

    POST-PRESENTATION UPDATE: Added date filtering support for cross-validation
    with SQL queries on specific date ranges (e.g., January 14, 2025).
    """
    session = get_session()
    try:
        # Get optional date filtering parameters
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        single_date = request.args.get('date')

        # If single date provided, use it for both start and end
        if single_date:
            start_date = single_date
            end_date = single_date

        # Build query with optional date filtering
        base_query = """
            SELECT d.code, COUNT(*) as frequency
            FROM diagnosis d
            JOIN encounter e ON d.encounter_id = e.encounter_id
            WHERE 1=1
        """
        params = {}

        # Add date filtering if provided
        if start_date:
            base_query += " AND date(e.arrival_ts) >= :start_date"
            params['start_date'] = start_date
        if end_date:
            base_query += " AND date(e.arrival_ts) <= :end_date"
            params['end_date'] = end_date

        base_query += " GROUP BY d.code ORDER BY frequency DESC LIMIT 10"

        # Execute top diagnoses query
        diagnosis_dist = session.execute(text(base_query), params).fetchall()

        # Build count queries with same date filters
        count_query = """
            SELECT COUNT(*) FROM diagnosis d
            JOIN encounter e ON d.encounter_id = e.encounter_id
            WHERE 1=1
        """
        primary_query = """
            SELECT COUNT(*) FROM diagnosis d
            JOIN encounter e ON d.encounter_id = e.encounter_id
            WHERE d.is_primary = 1
        """

        if start_date:
            count_query += " AND date(e.arrival_ts) >= :start_date"
            primary_query += " AND date(e.arrival_ts) >= :start_date"
        if end_date:
            count_query += " AND date(e.arrival_ts) <= :end_date"
            primary_query += " AND date(e.arrival_ts) <= :end_date"

        total_diagnoses = session.execute(text(count_query), params).scalar()
        primary_count = session.execute(text(primary_query), params).scalar()

        return jsonify({
            'total_diagnoses': total_diagnoses,
            'primary_diagnosis_count': primary_count,
            'date_range': {
                'start_date': start_date if start_date else 'all',
                'end_date': end_date if end_date else 'all'
            },
            'top_diagnoses': [{'code': row[0], 'count': row[1]} for row in diagnosis_dist]
        })
    finally:
        session.close()

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
    """Get staff list with optional filtering, search, and sorting"""
    session = get_session()
    try:
        # Query parameters for filtering
        active_only = request.args.get('active_only', default='false').lower() == 'true'
        limit = request.args.get('limit', default=100, type=int)
        offset = request.args.get('offset', default=0, type=int)
        
        # Search parameter
        search = request.args.get('search', type=str)
        
        # Sort parameters
        sort_by = request.args.get('sort_by', default='staff_id', type=str)
        sort_order = request.args.get('sort_order', default='asc', type=str)

        query = session.query(Staff)

        # Apply filters
        if active_only:
            query = query.filter(Staff.is_active == 1)
        
        # Apply search (searches across multiple fields)
        if search:
            search_term = f'%{search}%'
            query = query.filter(
                (Staff.first_name.like(search_term)) |
                (Staff.last_name.like(search_term)) |
                (Staff.role_code.like(search_term)) |
                (Staff.department.like(search_term)) |
                (text(f"CAST(staff.staff_id AS TEXT)").like(search_term))
            )

        # Apply sorting
        sort_column = None
        valid_sort_columns = {
            'staff_id': Staff.staff_id,
            'first_name': Staff.first_name,
            'last_name': Staff.last_name,
            'role_code': Staff.role_code,
            'department': Staff.department,
            'is_active': Staff.is_active,
        }
        
        if sort_by in valid_sort_columns:
            sort_column = valid_sort_columns[sort_by]
        else:
            sort_column = Staff.staff_id  # Default sort
        
        if sort_order.lower() == 'desc':
            query = query.order_by(desc(sort_column))
        else:
            query = query.order_by(sort_column)

        # Get total count before pagination
        total = query.count()
        
        # Apply pagination
        staff = query.limit(limit).offset(offset).all()

        return jsonify({
            'total': total,
            'limit': limit,
            'offset': offset,
            'sort_by': sort_by,
            'sort_order': sort_order,
            'search': search,
            'staff': [s.to_dict() for s in staff]
        })
    finally:
        session.close()

@api_bp.route('/staff', methods=['POST'])
@with_session
def create_staff():
    """Create a new staff member"""
    session = get_session()
    try:
        data = request.get_json()
        
        # Validate required fields
        if 'first_name' not in data or 'last_name' not in data or 'role_code' not in data:
            return jsonify({'error': 'first_name, last_name, and role_code are required'}), 400
        
        # Create new staff member
        staff = Staff(
            first_name=data['first_name'],
            last_name=data['last_name'],
            role_code=data['role_code'],
            department=data.get('department'),
            is_active=data.get('is_active', 1)
        )
        
        session.add(staff)
        session.commit()
        session.refresh(staff)
        
        return jsonify({
            'message': 'Staff member created successfully',
            'staff': staff.to_dict()
        }), 201
    except Exception as e:
        session.rollback()
        return jsonify({'error': str(e)}), 400
    finally:
        session.close()

@api_bp.route('/staff/<int:staff_id>', methods=['PUT'])
@with_session
def update_staff(staff_id):
    """Update an existing staff member"""
    session = get_session()
    try:
        staff = session.query(Staff).filter(Staff.staff_id == staff_id).first()
        
        if not staff:
            return jsonify({'error': 'Staff member not found'}), 404
        
        data = request.get_json()
        
        # Update fields if provided
        if 'first_name' in data:
            staff.first_name = data['first_name']
        if 'last_name' in data:
            staff.last_name = data['last_name']
        if 'role_code' in data:
            staff.role_code = data['role_code']
        if 'department' in data:
            staff.department = data['department']
        if 'is_active' in data:
            staff.is_active = 1 if data['is_active'] else 0
        
        session.commit()
        session.refresh(staff)
        
        return jsonify({
            'message': 'Staff member updated successfully',
            'staff': staff.to_dict()
        })
    except Exception as e:
        session.rollback()
        return jsonify({'error': str(e)}), 400
    finally:
        session.close()

@api_bp.route('/staff/<int:staff_id>', methods=['DELETE'])
@with_session
def delete_staff(staff_id):
    """Delete a staff member"""
    session = get_session()
    try:
        staff = session.query(Staff).filter(Staff.staff_id == staff_id).first()
        
        if not staff:
            return jsonify({'error': 'Staff member not found'}), 404
        
        # Check if staff has assignments (optional: prevent deletion if assigned)
        assignments = session.query(StaffAssignment).filter(StaffAssignment.staff_id == staff_id).count()
        if assignments > 0:
            return jsonify({
                'error': f'Cannot delete staff member with {assignments} active assignment(s). Please remove assignments first.'
            }), 400
        
        session.delete(staff)
        session.commit()
        
        return jsonify({
            'message': 'Staff member deleted successfully'
        })
    except Exception as e:
        session.rollback()
        return jsonify({'error': str(e)}), 400
    finally:
        session.close()

@api_bp.route('/wait-times', methods=['GET'])
@with_session
def get_wait_times():
    """Get wait time data for encounters
    
    Returns wait times (arrival to provider) for encounters with valid timestamps.
    Supports optional filtering by ESI level.
    """
    session = get_session()
    try:
        esi_level = request.args.get('esi_level', type=int)
        
        # Build query to calculate wait times using raw SQL with julianday
        # Wait time = (provider_start_ts - arrival_ts) in minutes
        base_query = """
            SELECT 
                encounter_id,
                esi_level,
                arrival_ts,
                provider_start_ts,
                CAST((julianday(provider_start_ts) - julianday(arrival_ts)) * 1440 AS INTEGER) AS wait_time_minutes
            FROM encounter
            WHERE arrival_ts IS NOT NULL AND arrival_ts != ''
              AND provider_start_ts IS NOT NULL AND provider_start_ts != ''
        """
        
        if esi_level:
            base_query += f" AND esi_level = {esi_level}"
        
        results = session.execute(text(base_query)).fetchall()
        
        wait_times = []
        for row in results:
            wait_times.append({
                'encounter_id': row[0],
                'esi_level': row[1],
                'arrival_ts': row[2],
                'provider_start_ts': row[3],
                'wait_time_minutes': row[4]
            })
        
        # Calculate statistics
        if wait_times:
            wait_time_values = [wt['wait_time_minutes'] for wt in wait_times if wt['wait_time_minutes'] is not None]
            avg_wait = sum(wait_time_values) / len(wait_time_values) if wait_time_values else 0
            min_wait = min(wait_time_values) if wait_time_values else 0
            max_wait = max(wait_time_values) if wait_time_values else 0
        else:
            avg_wait = min_wait = max_wait = 0
        
        return jsonify({
            'total': len(wait_times),
            'statistics': {
                'average_wait_minutes': round(avg_wait, 2),
                'min_wait_minutes': min_wait,
                'max_wait_minutes': max_wait
            },
            'wait_times': wait_times
        })
    finally:
        session.close()

@api_bp.route('/statistics/wait-times-by-esi', methods=['GET'])
@with_session
def get_wait_times_by_esi():
    """Get wait time statistics grouped by ESI level with optional date filtering

    Query parameters:
    - start_date: Filter encounters from this date (ISO format: YYYY-MM-DD)
    - end_date: Filter encounters until this date (ISO format: YYYY-MM-DD)

    POST-PRESENTATION UPDATE: Added date filtering support for cross-validation
    with SQL queries on specific date ranges (e.g., January 14, 2025).
    """
    session = get_session()
    try:
        # Get optional date filtering parameters (POST-PRESENTATION UPDATE)
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')

        esi_stats = []

        for esi in range(1, 6):
            # Build query with optional date filtering
            base_query = """
                SELECT CAST((julianday(provider_start_ts) - julianday(arrival_ts)) * 1440 AS INTEGER) AS wait_time_minutes
                FROM encounter
                WHERE esi_level = :esi_level
                  AND arrival_ts IS NOT NULL AND arrival_ts != ''
                  AND provider_start_ts IS NOT NULL AND provider_start_ts != ''
            """
            params = {'esi_level': esi}

            # Add date filtering if provided (POST-PRESENTATION UPDATE)
            if start_date:
                base_query += " AND date(arrival_ts) >= :start_date"
                params['start_date'] = start_date
            if end_date:
                base_query += " AND date(arrival_ts) <= :end_date"
                params['end_date'] = end_date

            query = text(base_query)
            results = session.execute(query, params).fetchall()
            
            if results:
                wait_times = [r[0] for r in results if r[0] is not None]
                if wait_times:
                    esi_stats.append({
                        'esi_level': esi,
                        'count': len(wait_times),
                        'average_wait_minutes': round(sum(wait_times) / len(wait_times), 2),
                        'min_wait_minutes': min(wait_times),
                        'max_wait_minutes': max(wait_times),
                        'median_wait_minutes': round(sorted(wait_times)[len(wait_times) // 2], 2)
                    })
                else:
                    esi_stats.append({
                        'esi_level': esi,
                        'count': 0,
                        'average_wait_minutes': 0,
                        'min_wait_minutes': 0,
                        'max_wait_minutes': 0,
                        'median_wait_minutes': 0
                    })
            else:
                esi_stats.append({
                    'esi_level': esi,
                    'count': 0,
                    'average_wait_minutes': 0,
                    'min_wait_minutes': 0,
                    'max_wait_minutes': 0,
                    'median_wait_minutes': 0
                })
        
        return jsonify({
            'wait_times_by_esi': esi_stats
        })
    finally:
        session.close()

@api_bp.route('/statistics/length-of-stay', methods=['GET'])
@with_session
def get_length_of_stay():
    """Get length of stay statistics
    
    Length of stay = (departure_ts - arrival_ts) in minutes
    """
    session = get_session()
    try:
        # Calculate LOS for all encounters with valid timestamps using raw SQL
        query = text("""
            SELECT 
                encounter_id,
                esi_level,
                disposition_code,
                arrival_ts,
                departure_ts,
                CAST((julianday(departure_ts) - julianday(arrival_ts)) * 1440 AS INTEGER) AS los_minutes
            FROM encounter
            WHERE arrival_ts IS NOT NULL AND arrival_ts != ''
              AND departure_ts IS NOT NULL AND departure_ts != ''
        """)
        results = session.execute(query).fetchall()
        
        los_data = []
        for row in results:
            los_data.append({
                'encounter_id': row[0],
                'esi_level': row[1],
                'disposition_code': row[2],
                'arrival_ts': row[3],
                'departure_ts': row[4],
                'los_minutes': row[5]
            })
        
        # Calculate overall statistics
        if los_data:
            los_values = [los['los_minutes'] for los in los_data if los['los_minutes'] is not None]
            if los_values:
                avg_los = sum(los_values) / len(los_values)
                min_los = min(los_values)
                max_los = max(los_values)
                sorted_los = sorted(los_values)
                median_los = sorted_los[len(sorted_los) // 2]
            else:
                avg_los = min_los = max_los = median_los = 0
        else:
            avg_los = min_los = max_los = median_los = 0
        
        # Calculate LOS by ESI level
        los_by_esi = []
        for esi in range(1, 6):
            esi_los = [los['los_minutes'] for los in los_data 
                      if los['esi_level'] == esi and los['los_minutes'] is not None]
            if esi_los:
                los_by_esi.append({
                    'esi_level': esi,
                    'count': len(esi_los),
                    'average_los_minutes': round(sum(esi_los) / len(esi_los), 2),
                    'min_los_minutes': min(esi_los),
                    'max_los_minutes': max(esi_los),
                    'median_los_minutes': round(sorted(esi_los)[len(esi_los) // 2], 2)
                })
            else:
                los_by_esi.append({
                    'esi_level': esi,
                    'count': 0,
                    'average_los_minutes': 0,
                    'min_los_minutes': 0,
                    'max_los_minutes': 0,
                    'median_los_minutes': 0
                })
        
        # Calculate LOS by disposition
        los_by_disposition = {}
        for los in los_data:
            if los['disposition_code'] and los['los_minutes'] is not None:
                if los['disposition_code'] not in los_by_disposition:
                    los_by_disposition[los['disposition_code']] = []
                los_by_disposition[los['disposition_code']].append(los['los_minutes'])
        
        disposition_stats = []
        for disposition, values in los_by_disposition.items():
            disposition_stats.append({
                'disposition_code': disposition,
                'count': len(values),
                'average_los_minutes': round(sum(values) / len(values), 2),
                'min_los_minutes': min(values),
                'max_los_minutes': max(values)
            })
        
        return jsonify({
            'total_encounters': len(los_data),
            'overall_statistics': {
                'average_los_minutes': round(avg_los, 2),
                'min_los_minutes': min_los,
                'max_los_minutes': max_los,
                'median_los_minutes': round(median_los, 2)
            },
            'los_by_esi': los_by_esi,
            'los_by_disposition': disposition_stats
        })
    finally:
        session.close()

@api_bp.route('/statistics/staff-workload', methods=['GET'])
@with_session
def get_staff_workload():
    """Get staff workload statistics with optional date filtering

    Query parameters:
    - start_date: Filter encounters from this date (ISO format: YYYY-MM-DD)
    - end_date: Filter encounters until this date (ISO format: YYYY-MM-DD)

    Returns:
    - Per-staff metrics: patient count, average encounter duration, ESI distribution
    - Workload distribution by role
    - Overall summary statistics
    """
    session = get_session()
    try:
        # Get optional date filtering parameters
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')

        # Build base query to get staff assignments with encounter details
        base_query = """
            SELECT
                sa.staff_id,
                s.first_name || ' ' || s.last_name as staff_name,
                s.role_code as staff_role,
                sa.assignment_role,
                sa.encounter_id,
                e.esi_level,
                e.arrival_ts,
                e.departure_ts,
                sa.assigned_ts,
                sa.released_ts,
                CAST((julianday(e.departure_ts) - julianday(e.arrival_ts)) * 1440 AS INTEGER) AS encounter_duration_minutes,
                CAST((julianday(sa.released_ts) - julianday(sa.assigned_ts)) * 1440 AS INTEGER) AS assignment_duration_minutes
            FROM staff_assignment sa
            JOIN staff s ON sa.staff_id = s.staff_id
            JOIN encounter e ON sa.encounter_id = e.encounter_id
            WHERE e.arrival_ts IS NOT NULL AND e.arrival_ts != ''
              AND e.departure_ts IS NOT NULL AND e.departure_ts != ''
        """

        # Add date filtering if provided
        params = {}
        if start_date:
            base_query += " AND date(e.arrival_ts) >= :start_date"
            params['start_date'] = start_date
        if end_date:
            base_query += " AND date(e.arrival_ts) <= :end_date"
            params['end_date'] = end_date

        base_query += " ORDER BY sa.staff_id, e.arrival_ts"

        # Execute query
        results = session.execute(text(base_query), params).fetchall()

        # Process results into staff-level statistics
        staff_workload = {}
        role_workload = {}

        for row in results:
            staff_id = row[0]
            staff_name = row[1]
            staff_role = row[2]
            assignment_role = row[3]
            encounter_id = row[4]
            esi_level = row[5]
            encounter_duration = row[10]
            assignment_duration = row[11]

            # Initialize staff entry if not exists
            if staff_id not in staff_workload:
                staff_workload[staff_id] = {
                    'staff_id': staff_id,
                    'staff_name': staff_name,
                    'staff_role': staff_role,
                    'patient_count': 0,
                    'encounter_durations': [],
                    'assignment_durations': [],
                    'esi_distribution': {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
                    'assignment_roles': {}
                }

            # Update staff metrics
            staff_workload[staff_id]['patient_count'] += 1
            if encounter_duration is not None:
                staff_workload[staff_id]['encounter_durations'].append(encounter_duration)
            if assignment_duration is not None:
                staff_workload[staff_id]['assignment_durations'].append(assignment_duration)
            if esi_level:
                staff_workload[staff_id]['esi_distribution'][esi_level] += 1

            # Track assignment roles
            if assignment_role not in staff_workload[staff_id]['assignment_roles']:
                staff_workload[staff_id]['assignment_roles'][assignment_role] = 0
            staff_workload[staff_id]['assignment_roles'][assignment_role] += 1

            # Track role-level workload
            if staff_role not in role_workload:
                role_workload[staff_role] = {
                    'role': staff_role,
                    'staff_count': set(),
                    'patient_count': 0,
                    'encounter_durations': []
                }
            role_workload[staff_role]['staff_count'].add(staff_id)
            role_workload[staff_role]['patient_count'] += 1
            if encounter_duration is not None:
                role_workload[staff_role]['encounter_durations'].append(encounter_duration)

        # Calculate final statistics for each staff member
        staff_stats = []
        for staff_id, data in staff_workload.items():
            encounter_durations = data['encounter_durations']
            assignment_durations = data['assignment_durations']

            staff_stats.append({
                'staff_id': data['staff_id'],
                'staff_name': data['staff_name'],
                'staff_role': data['staff_role'],
                'patient_count': data['patient_count'],
                'average_encounter_duration_minutes': round(sum(encounter_durations) / len(encounter_durations), 2) if encounter_durations else 0,
                'average_assignment_duration_minutes': round(sum(assignment_durations) / len(assignment_durations), 2) if assignment_durations else 0,
                'total_assignment_time_minutes': sum(assignment_durations) if assignment_durations else 0,
                'esi_distribution': data['esi_distribution'],
                'assignment_roles': data['assignment_roles']
            })

        # Calculate role-level statistics
        role_stats = []
        for role, data in role_workload.items():
            encounter_durations = data['encounter_durations']

            role_stats.append({
                'role': role,
                'staff_count': len(data['staff_count']),
                'total_patient_count': data['patient_count'],
                'average_patients_per_staff': round(data['patient_count'] / len(data['staff_count']), 2) if data['staff_count'] else 0,
                'average_encounter_duration_minutes': round(sum(encounter_durations) / len(encounter_durations), 2) if encounter_durations else 0
            })

        # Sort results
        staff_stats.sort(key=lambda x: x['patient_count'], reverse=True)
        role_stats.sort(key=lambda x: x['total_patient_count'], reverse=True)

        # Calculate overall statistics
        total_assignments = len(results)
        total_unique_staff = len(staff_workload)
        total_patients = sum(s['patient_count'] for s in staff_stats)

        return jsonify({
            'total_assignments': total_assignments,
            'total_unique_staff': total_unique_staff,
            'total_patients_served': total_patients,
            'date_range': {
                'start_date': start_date if start_date else 'all',
                'end_date': end_date if end_date else 'all'
            },
            'staff_workload': staff_stats,
            'workload_by_role': role_stats
        })
    finally:
        session.close()
