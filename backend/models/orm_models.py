"""
SQLAlchemy ORM models for Emergency Department database

Maps to the schema created by Shaobo in db_setup.sql.
"""

from sqlalchemy import Column, Integer, String, Float, ForeignKey, Text
from sqlalchemy.orm import relationship
from config.database import Base

class Patient(Base):
    __tablename__ = 'patient'

    patient_id = Column(Integer, primary_key=True)
    dob = Column(Text)
    sex_at_birth = Column(Text)
    gender_identity = Column(Text)
    zip_code = Column(Text)

    # Relationships
    encounters = relationship('Encounter', back_populates='patient')

    def to_dict(self):
        return {
            'patient_id': self.patient_id,
            'dob': self.dob,
            'sex_at_birth': self.sex_at_birth,
            'gender_identity': self.gender_identity,
            'zip_code': self.zip_code
        }

class Staff(Base):
    __tablename__ = 'staff'

    staff_id = Column(Integer, primary_key=True)
    first_name = Column(Text, nullable=False)
    last_name = Column(Text, nullable=False)
    role_code = Column(Text, nullable=False)
    department = Column(Text)
    is_active = Column(Integer, nullable=False)

    # Relationships
    assignments = relationship('StaffAssignment', back_populates='staff')

    def to_dict(self):
        return {
            'staff_id': self.staff_id,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'role_code': self.role_code,
            'department': self.department,
            'is_active': bool(self.is_active)
        }

class Encounter(Base):
    __tablename__ = 'encounter'

    encounter_id = Column(Integer, primary_key=True)
    patient_id = Column(Integer, ForeignKey('patient.patient_id'), nullable=False)
    arrival_ts = Column(Text)
    triage_start_ts = Column(Text)
    triage_end_ts = Column(Text)
    provider_start_ts = Column(Text)
    dispo_decision_ts = Column(Text)
    departure_ts = Column(Text)
    arrival_mode = Column(Text)
    chief_complaint = Column(Text)
    esi_level = Column(Integer)
    disposition_code = Column(Text)
    referral_code = Column(Text)
    left_without_being_seen = Column(Integer, default=0)
    notes = Column(Text)

    # Relationships
    patient = relationship('Patient', back_populates='encounters')
    payor = relationship('EncounterPayor', uselist=False, back_populates='encounter')
    vitals = relationship('Vitals', back_populates='encounter')
    diagnoses = relationship('Diagnosis', back_populates='encounter')
    staff_assignments = relationship('StaffAssignment', back_populates='encounter')

    def to_dict(self):
        return {
            'encounter_id': self.encounter_id,
            'patient_id': self.patient_id,
            'arrival_ts': self.arrival_ts,
            'triage_start_ts': self.triage_start_ts,
            'triage_end_ts': self.triage_end_ts,
            'provider_start_ts': self.provider_start_ts,
            'dispo_decision_ts': self.dispo_decision_ts,
            'departure_ts': self.departure_ts,
            'arrival_mode': self.arrival_mode,
            'chief_complaint': self.chief_complaint,
            'esi_level': self.esi_level,
            'disposition_code': self.disposition_code,
            'referral_code': self.referral_code,
            'left_without_being_seen': bool(self.left_without_being_seen),
            'notes': self.notes
        }

class EncounterPayor(Base):
    __tablename__ = 'encounter_payor'

    encounter_id = Column(Integer, ForeignKey('encounter.encounter_id'), primary_key=True)
    payor_name = Column(Text)
    payor_type = Column(Text)
    member_id = Column(Text)

    # Relationships
    encounter = relationship('Encounter', back_populates='payor')

    def to_dict(self):
        return {
            'encounter_id': self.encounter_id,
            'payor_name': self.payor_name,
            'payor_type': self.payor_type,
            'member_id': self.member_id
        }

class Vitals(Base):
    __tablename__ = 'vitals'

    vital_id = Column(Integer, primary_key=True)
    encounter_id = Column(Integer, ForeignKey('encounter.encounter_id'), nullable=False)
    taken_ts = Column(Text, nullable=False)
    heart_rate = Column(Integer)
    systolic_bp = Column(Integer)
    diastolic_bp = Column(Integer)
    respiratory_rate = Column(Integer)
    temperature_c = Column(Float)
    spo2 = Column(Integer)
    pain_score = Column(Integer)

    # Relationships
    encounter = relationship('Encounter', back_populates='vitals')

    def to_dict(self):
        return {
            'vital_id': self.vital_id,
            'encounter_id': self.encounter_id,
            'taken_ts': self.taken_ts,
            'heart_rate': self.heart_rate,
            'systolic_bp': self.systolic_bp,
            'diastolic_bp': self.diastolic_bp,
            'respiratory_rate': self.respiratory_rate,
            'temperature_c': self.temperature_c,
            'spo2': self.spo2,
            'pain_score': self.pain_score
        }

class Diagnosis(Base):
    __tablename__ = 'diagnosis'

    encounter_id = Column(Integer, ForeignKey('encounter.encounter_id'), primary_key=True)
    code = Column(Text, primary_key=True)
    is_primary = Column(Integer, nullable=False)

    # Relationships
    encounter = relationship('Encounter', back_populates='diagnoses')

    def to_dict(self):
        return {
            'encounter_id': self.encounter_id,
            'code': self.code,
            'is_primary': bool(self.is_primary)
        }

class StaffAssignment(Base):
    __tablename__ = 'staff_assignment'

    encounter_id = Column(Integer, ForeignKey('encounter.encounter_id'), primary_key=True)
    staff_id = Column(Integer, ForeignKey('staff.staff_id'), primary_key=True)
    assignment_role = Column(Text, primary_key=True, nullable=False)
    assigned_ts = Column(Text, primary_key=True, nullable=False)
    released_ts = Column(Text)

    # Relationships
    encounter = relationship('Encounter', back_populates='staff_assignments')
    staff = relationship('Staff', back_populates='assignments')

    def to_dict(self):
        return {
            'encounter_id': self.encounter_id,
            'staff_id': self.staff_id,
            'assignment_role': self.assignment_role,
            'assigned_ts': self.assigned_ts,
            'released_ts': self.released_ts
        }
