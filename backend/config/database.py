"""
Database configuration and connection management

Handles SQLAlchemy setup and database initialization.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()
db_session = None

def init_db(app):
    """Initialize database connection"""
    global db_session

    db_path = app.config['DATABASE_PATH']
    engine = create_engine(f'sqlite:///{db_path}',
                          connect_args={'check_same_thread': False})

    # Create scoped session
    session_factory = sessionmaker(bind=engine)
    db_session = scoped_session(session_factory)

    # Import models to register them
    from models.orm_models import Patient, Staff, Encounter, EncounterPayor, Vitals, Diagnosis, StaffAssignment

    Base.query = db_session.query_property()

    return db_session

def get_session():
    """Get database session"""
    return db_session
