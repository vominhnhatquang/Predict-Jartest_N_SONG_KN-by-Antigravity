"""
Flask Application - AI Model Web Integration
Main API server for model predictions
"""
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import sys
import traceback
import csv
import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

from config import config
from utils import create_response, validate_numeric_input
from model.model import ModelPredictor

# Initialize Flask app
app = Flask(__name__, static_folder='../frontend', static_url_path='')

# Load configuration
env = os.getenv('FLASK_ENV', 'development')
app.config.from_object(config[env])

# Enable CORS
CORS(app, origins=app.config['CORS_ORIGINS'])

# Initialize model predictor (will be loaded on first request or startup)
model_predictor = None



# Validating model initialization
def init_model():
    """Initialize and load the model"""
    global model_predictor
    
    if model_predictor is None:
        try:
            model_path = app.config['MODEL_PATH']
            scaler_path = app.config['SCALER_PATH']
            imputer_path = app.config['IMPUTER_PATH']
            
            # Check if model file exists
            if not os.path.exists(model_path):
                print(f"[WARN] Model file not found: {model_path}")
                print("[INFO] Please train and save the model first.")
                return False
            
            model_predictor = ModelPredictor(model_path, scaler_path, imputer_path)
            model_predictor.load_model()
            print("[OK] Model initialized successfully")
            return True
            
        except Exception as e:
            print(f"[ERROR] Error initializing model: {str(e)}")
            traceback.print_exc()
            return False
    
    return True


@app.before_request
def log_request_info():
    """Log connection info to CSV"""
    log_file = os.path.join(os.path.dirname(app.root_path), 'connection_logs.csv')
    file_exists = os.path.isfile(log_file)
    
    try:
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            # Write header if file doesn't exist
            if not file_exists:
                writer.writerow(['Timestamp', 'IP Address', 'Endpoint', 'Method', 'User Agent'])
            
            # Write log entry
            writer.writerow([
                datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                request.remote_addr,
                request.path,
                request.method,
                request.user_agent.string
            ])
    except Exception as e:
        print(f"Error logging request: {str(e)}")



# Routes

@app.route('/')
def index():
    """Serve the main frontend page"""
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    model_loaded = model_predictor is not None and model_predictor.is_loaded
    
    return create_response(
        success=True,
        data={
            'status': 'healthy',
            'model_loaded': model_loaded,
            'environment': env
        },
        message='API is running'
    )


@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get model information"""
    if not init_model():
        return create_response(
            success=False,
            message='Model not available',
            status_code=503
        )
    
    info = model_predictor.get_model_info()
    
    return create_response(
        success=True,
        data=info,
        message='Model information retrieved'
    )


@app.route('/api/performance', methods=['GET'])
def model_performance():
    """Get model performance metrics"""
    try:
        import json
        metrics_path = os.path.join(os.path.dirname(app.config['MODEL_PATH']), 'metrics.json')
        
        if not os.path.exists(metrics_path):
            return create_response(
                success=False,
                message='Metrics not found. Please re-train model.',
                status_code=404
            )
            
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
            
        return create_response(
            success=True,
            data=metrics,
            message='Model performance metrics retrieved'
        )
    except Exception as e:
        print(f"Error loading metrics: {str(e)}")
        return create_response(
            success=False,
            message=f'Error loading metrics: {str(e)}',
            status_code=500
        )


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Make predictions
    
    Expected JSON format:
    {
        "features": [value1, value2, ...] or
        "feature_name_1": value1,
        "feature_name_2": value2,
        ...
    }
    """
    try:
        # Initialize model if not loaded
        if not init_model():
            return create_response(
                success=False,
                message='Model not loaded. Please ensure model files exist.',
                status_code=503
            )
        
        # Get request data
        data = request.get_json()
        
        if not data:
            return create_response(
                success=False,
                message='No data provided',
                status_code=400
            )
        
        # Extract features
        if 'features' in data:
            # Array format
            features = data['features']
            if not isinstance(features, list):
                return create_response(
                    success=False,
                    message='Features must be a list of numbers',
                    status_code=400
                )
        else:
            # Dictionary format - extract values
            # Note: You need to define the expected feature names
            # For now, we'll accept any numeric fields
            features = [float(v) for k, v in data.items() if k != 'features']
        
        # Validate features
        if not features:
            return create_response(
                success=False,
                message='No features provided',
                status_code=400
            )
        
        # Make prediction
        result = model_predictor.predict_with_confidence(features)
        
        return create_response(
            success=True,
            data=result,
            message='Prediction successful'
        )
    
    except ValueError as e:
        return create_response(
            success=False,
            message=f'Invalid input: {str(e)}',
            status_code=400
        )
    
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        traceback.print_exc()
        return create_response(
            success=False,
            message=f'Prediction error: {str(e)}',
            status_code=500
        )


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return create_response(
        success=False,
        message='Endpoint not found',
        status_code=404
    )


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return create_response(
        success=False,
        message='Internal server error',
        status_code=500
    )


if __name__ == '__main__':
    # Try to initialize model on startup
    init_model()
    
    # Run the app
    port = int(os.getenv('PORT', 5000))
    debug = app.config['DEBUG']
    
    print(f"\n{'='*50}")
    print(f"[INFO] Starting Flask server...")
    print(f"[INFO] Environment: {env}")
    print(f"[INFO] Port: {port}")
    print(f"[INFO] Debug: {debug}")
    print(f"{'='*50}\n")
    
    app.run(host='0.0.0.0', port=port, debug=debug)
