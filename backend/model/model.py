"""
Model loading and inference module
"""
import joblib
import numpy as np
from typing import Union, Optional, Dict
import os


class ModelPredictor:
    """Load and use trained model for predictions"""
    
    def __init__(self, model_path: str, scaler_path: Optional[str] = None, 
                 imputer_path: Optional[str] = None):
        """
        Initialize model predictor
        
        Args:
            model_path (str): Path to trained model file
            scaler_path (str, optional): Path to scaler file
            imputer_path (str, optional): Path to imputer file
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.imputer_path = imputer_path
        
        self.model = None
        self.scaler = None
        self.imputer = None
        
        self.is_loaded = False
        self.model_info = {}
    
    def load_model(self):
        """Load model and preprocessing objects"""
        try:
            # Load model
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                print(f"[OK] Model loaded from {self.model_path}")
            else:
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            # Load scaler if provided
            if self.scaler_path and os.path.exists(self.scaler_path):
                self.scaler = joblib.load(self.scaler_path)
                print(f"[OK] Scaler loaded from {self.scaler_path}")
            
            # Load imputer if provided
            if self.imputer_path and os.path.exists(self.imputer_path):
                self.imputer = joblib.load(self.imputer_path)
                print(f"[OK] Imputer loaded from {self.imputer_path}")
            
            self.is_loaded = True
            self._extract_model_info()
            
        except Exception as e:
            print(f"[ERROR] Error loading model: {str(e)}")
            raise
    
    def _extract_model_info(self):
        """Extract information about the model"""
        if self.model is not None:
            self.model_info = {
                'model_type': type(self.model).__name__,
                'has_scaler': self.scaler is not None,
                'has_imputer': self.imputer is not None,
            }
            
            # Add model-specific info
            if hasattr(self.model, 'coef_'):
                self.model_info['n_features'] = len(self.model.coef_)
            
            if hasattr(self.model, 'intercept_'):
                self.model_info['has_intercept'] = True
    
    def preprocess(self, X: np.ndarray) -> np.ndarray:
        """
        Preprocess input data
        
        Args:
            X (np.ndarray): Input data
        
        Returns:
            np.ndarray: Preprocessed data
        """
        # Handle missing values
        if self.imputer is not None:
            X = self.imputer.transform(X)
        
        # Scale features
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        return X
    
    def predict(self, X: Union[np.ndarray, list]) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Input features (numpy array or list)
        
        Returns:
            np.ndarray: Predictions
        """
        if not self.is_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Convert to numpy array if needed
        if isinstance(X, list):
            X = np.array(X)
        
        # Ensure 2D array
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Preprocess
        X_processed = self.preprocess(X)
        
        # Predict
        predictions = self.model.predict(X_processed)
        
        return predictions
    
    def predict_with_confidence(self, X: Union[np.ndarray, list]) -> Dict:
        """
        Make predictions with additional information
        
        Args:
            X: Input features
        
        Returns:
            dict: Predictions with metadata
        """
        predictions = self.predict(X)
        
        result = {
            'predictions': predictions.tolist(),
            'model_type': self.model_info.get('model_type', 'Unknown'),
        }
        
        # Add prediction interval if available (for some models)
        # This is a simplified version - expand based on model type
        if hasattr(self.model, 'predict') and len(predictions) > 0:
            result['prediction_value'] = float(predictions[0])
        
        return result
    
    def get_model_info(self) -> Dict:
        """
        Get model information
        
        Returns:
            dict: Model metadata
        """
        if not self.is_loaded:
            return {'error': 'Model not loaded'}
        
        return self.model_info
