"""
Database configuration and connection management

Handles SQLAlchemy setup and database initialization.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()
db_session = None
engine = None

def init_db(app):
    """Initialize database connection"""
    global db_session, engine

    db_path = app.config['DATABASE_PATH']
    
    # Create engine with increased pool size and proper configuration
    engine = create_engine(
        f'sqlite:///{db_path}',
        connect_args={'check_same_thread': False},
        pool_size=10,  # Increased from default 5
        max_overflow=20,  # Increased from default 10
        pool_pre_ping=True,  # Verify connections before using
        pool_recycle=3600,  # Recycle connections after 1 hour
        echo=False  # Set to True for SQL query logging
    )

    # Create scoped session
    session_factory = sessionmaker(bind=engine)
    db_session = scoped_session(session_factory)

    # Import models to register them
    from models.orm_models import Patient, Staff, Encounter, EncounterPayor, Vitals, Diagnosis, StaffAssignment

    Base.query = db_session.query_property()

    # Add teardown handler to close sessions after each request
    @app.teardown_appcontext
    def shutdown_session(exception=None):
        """Remove database session after each request"""
        db_session.remove()

    return db_session

def get_session():
    """Get database session"""
    return db_session
