"""
Feature scaling module
Scales features to a standard range
"""
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
from typing import Optional


class FeatureScaler:
    """Scale features for model input"""
    
    def __init__(self, scaler_type: str = 'standard'):
        """
        Initialize scaler
        
        Args:
            scaler_type (str): Type of scaler ('standard' or 'minmax')
        """
        self.scaler_type = scaler_type
        
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}. Use 'standard' or 'minmax'")
        
        self.is_fitted = False
    
    def fit(self, X: np.ndarray) -> 'FeatureScaler':
        """
        Fit scaler on training data
        
        Args:
            X (np.ndarray): Training data
        
        Returns:
            self
        """
        self.scaler.fit(X)
        self.is_fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data by scaling features
        
        Args:
            X (np.ndarray): Data to transform
        
        Returns:
            np.ndarray: Scaled data
        """
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before transform")
        
        return self.scaler.transform(X)
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit and transform data
        
        Args:
            X (np.ndarray): Data to fit and transform
        
        Returns:
            np.ndarray: Scaled data
        """
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Reverse the scaling transformation
        
        Args:
            X (np.ndarray): Scaled data
        
        Returns:
            np.ndarray: Original scale data
        """
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before inverse_transform")
        
        return self.scaler.inverse_transform(X)
    
    def save(self, filepath: str):
        """
        Save scaler to file
        
        Args:
            filepath (str): Path to save scaler
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted scaler")
        
        joblib.dump(self.scaler, filepath)
        print(f"Scaler saved to {filepath}")
    
    @staticmethod
    def load(filepath: str, scaler_type: str = 'standard') -> 'FeatureScaler':
        """
        Load scaler from file
        
        Args:
            filepath (str): Path to scaler file
            scaler_type (str): Type of scaler
        
        Returns:
            FeatureScaler: Loaded scaler
        """
        scaler_obj = FeatureScaler(scaler_type)
        scaler_obj.scaler = joblib.load(filepath)
        scaler_obj.is_fitted = True
        print(f"Scaler loaded from {filepath}")
        return scaler_obj
    
    def get_parameters(self) -> dict:
        """
        Get scaler parameters
        
        Returns:
            dict: Scaler parameters (mean, std for StandardScaler; min, max for MinMaxScaler)
        """
        if not self.is_fitted:
            return {}
        
        params = {'type': self.scaler_type}
        
        if self.scaler_type == 'standard':
            params['mean'] = self.scaler.mean_.tolist() if hasattr(self.scaler, 'mean_') else None
            params['scale'] = self.scaler.scale_.tolist() if hasattr(self.scaler, 'scale_') else None
        elif self.scaler_type == 'minmax':
            params['min'] = self.scaler.min_.tolist() if hasattr(self.scaler, 'min_') else None
            params['scale'] = self.scaler.scale_.tolist() if hasattr(self.scaler, 'scale_') else None
        
        return params
