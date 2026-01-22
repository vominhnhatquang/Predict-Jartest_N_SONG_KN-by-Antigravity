"""
Input validation module
Validates user input data before preprocessing
"""
import numpy as np
import pandas as pd
from typing import Union, Tuple, List


class InputValidator:
    """Validates input data for the model"""
    
    def __init__(self, expected_features: List[str], expected_dtypes: str = 'float64'):
        """
        Initialize validator
        
        Args:
            expected_features (List[str]): List of expected feature names
            expected_dtypes (str): Expected data type (default: 'float64')
        """
        self.expected_features = expected_features
        self.expected_dtypes = expected_dtypes
        self.n_features = len(expected_features)
    
    def validate_input(self, data: Union[dict, pd.DataFrame, np.ndarray]) -> Tuple[bool, str, np.ndarray]:
        """
        Validate input data
        
        Args:
            data: Input data (dict, DataFrame, or ndarray)
        
        Returns:
            Tuple[bool, str, np.ndarray]: (is_valid, error_message, processed_data)
        """
        try:
            # Convert to numpy array
            if isinstance(data, dict):
                processed_data = self._validate_dict(data)
            elif isinstance(data, pd.DataFrame):
                processed_data = self._validate_dataframe(data)
            elif isinstance(data, np.ndarray):
                processed_data = self._validate_array(data)
            else:
                return False, f"Unsupported data type: {type(data)}", None
            
            # Check for non-numeric values
            if not np.issubdtype(processed_data.dtype, np.number):
                return False, "All values must be numeric", None
            
            # Check shape
            if processed_data.shape[1] != self.n_features:
                return False, f"Expected {self.n_features} features, got {processed_data.shape[1]}", None
            
            return True, "", processed_data
            
        except Exception as e:
            return False, f"Validation error: {str(e)}", None
    
    def _validate_dict(self, data: dict) -> np.ndarray:
        """Validate and convert dictionary input"""
        # Check for missing features
        missing_features = set(self.expected_features) - set(data.keys())
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        
        # Extract values in the correct order
        values = [float(data[feature]) for feature in self.expected_features]
        return np.array(values).reshape(1, -1)
    
    def _validate_dataframe(self, data: pd.DataFrame) -> np.ndarray:
        """Validate and convert DataFrame input"""
        # Check for missing columns
        missing_cols = set(self.expected_features) - set(data.columns)
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
        
        # Select columns in the correct order
        return data[self.expected_features].values
    
    def _validate_array(self, data: np.ndarray) -> np.ndarray:
        """Validate numpy array input"""
        if data.ndim == 1:
            data = data.reshape(1, -1)
        elif data.ndim != 2:
            raise ValueError(f"Array must be 1D or 2D, got {data.ndim}D")
        
        return data
    
    def get_feature_names(self) -> List[str]:
        """Get list of expected feature names"""
        return self.expected_features
