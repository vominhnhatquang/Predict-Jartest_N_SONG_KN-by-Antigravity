"""
Configuration file for the Flask application
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Base configuration"""
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    DEBUG = os.getenv('DEBUG', 'True') == 'True'
    
    # Model settings
    MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'trained_model.pkl')
    SCALER_PATH = os.path.join(os.path.dirname(__file__), 'model', 'scaler.pkl')
    IMPUTER_PATH = os.path.join(os.path.dirname(__file__), 'model', 'imputer.pkl')
    
    # API settings
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max request size
    
    # CORS settings
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', '*').split(',')


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    SECRET_KEY = os.getenv('SECRET_KEY')  # Must be set in production


# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}
