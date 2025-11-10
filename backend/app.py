"""
Flask API for Emergency Department Analysis System

Main application entry point for the backend API.
Author: Suk Jin Mun
Course: DS 5110, Fall 2025
"""

from flask import Flask
from flask_cors import CORS
from config.database import init_db
from routes.api import api_bp
from routes.predictions import predictions_bp

def create_app():
    """Create and configure Flask application"""
    app = Flask(__name__)

    # Configuration
    app.config['DATABASE_PATH'] = '../ed_database.db'
    app.config['JSON_SORT_KEYS'] = False

    # Enable CORS for frontend communication
    CORS(app)

    # Initialize database
    init_db(app)

    # Register blueprints
    app.register_blueprint(api_bp, url_prefix='/api')
    app.register_blueprint(predictions_bp, url_prefix='/api/predictions')

    @app.route('/')
    def index():
        return {
            "message": "Emergency Department API",
            "version": "1.1",
            "endpoints": {
                "health": "/api/health",
                "encounters": "/api/encounters",
                "patients": "/api/patients",
                "wait_times": "/api/wait-times",
                "statistics": "/api/statistics",
                "predictions": {
                    "models_info": "/api/predictions/models/info",
                    "predict_esi": "/api/predictions/esi (POST)",
                    "predict_wait_time": "/api/predictions/wait-time (POST)",
                    "predict_volume": "/api/predictions/volume (GET)"
                }
            }
        }

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)
