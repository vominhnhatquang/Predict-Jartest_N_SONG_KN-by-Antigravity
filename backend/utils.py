"""
Utility functions for the application
"""
from flask import jsonify
import numpy as np


def create_response(success: bool, data=None, message: str = "", status_code: int = 200):
    """
    Create a standardized JSON response
    
    Args:
        success (bool): Whether the request was successful
        data: The data to return
        message (str): Message to include in response
        status_code (int): HTTP status code
    
    Returns:
        Flask response object
    """
    response = {
        'success': success,
        'message': message
    }
    
    if data is not None:
        response['data'] = data
    
    return jsonify(response), status_code


def validate_numeric_input(data: dict, required_fields: list) -> tuple:
    """
    Validate that input data contains required numeric fields
    
    Args:
        data (dict): Input data dictionary
        required_fields (list): List of required field names
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if not data:
        return False, "No data provided"
    
    for field in required_fields:
        if field not in data:
            return False, f"Missing required field: {field}"
        
        try:
            float(data[field])
        except (ValueError, TypeError):
            return False, f"Field '{field}' must be a number"
    
    return True, ""


def prepare_features(data: dict, feature_names: list) -> np.ndarray:
    """
    Convert input dictionary to numpy array in the correct order
    
    Args:
        data (dict): Input data dictionary
        feature_names (list): Ordered list of feature names
    
    Returns:
        np.ndarray: Features array
    """
    features = []
    for feature in feature_names:
        features.append(float(data[feature]))
    
    return np.array(features).reshape(1, -1)
