"""
Test cases for preprocessing module
"""
import pytest
import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from preprocessing.validator import InputValidator
from preprocessing.imputer import DataImputer
from preprocessing.scaler import FeatureScaler


class TestInputValidator:
    """Test InputValidator class"""
    
    def test_validate_dict_input(self):
        """Test dictionary input validation"""
        features = ['feature1', 'feature2', 'feature3']
        validator = InputValidator(features)
        
        data = {'feature1': 1.0, 'feature2': 2.0, 'feature3': 3.0}
        is_valid, error, result = validator.validate_input(data)
        
        assert is_valid == True
        assert error == ""
        assert result.shape == (1, 3)
    
    def test_missing_features(self):
        """Test missing features detection"""
        features = ['feature1', 'feature2', 'feature3']
        validator = InputValidator(features)
        
        data = {'feature1': 1.0, 'feature2': 2.0}
        is_valid, error, result = validator.validate_input(data)
        
        assert is_valid == False
        assert "Missing features" in error


class TestDataImputer:
    """Test DataImputer class"""
    
    def test_fit_transform(self):
        """Test fit and transform"""
        imputer = DataImputer(strategy='mean')
        
        X = np.array([[1, 2, 3], [4, np.nan, 6], [7, 8, 9]])
        X_transformed = imputer.fit_transform(X)
        
        assert imputer.is_fitted == True
        assert not np.isnan(X_transformed).any()
    
    def test_transform_without_fit(self):
        """Test transform before fit raises error"""
        imputer = DataImputer()
        X = np.array([[1, 2, 3]])
        
        with pytest.raises(ValueError):
            imputer.transform(X)


class TestFeatureScaler:
    """Test FeatureScaler class"""
    
    def test_standard_scaler(self):
        """Test StandardScaler"""
        scaler = FeatureScaler(scaler_type='standard')
        
        X = np.array([[1, 2], [3, 4], [5, 6]])
        X_scaled = scaler.fit_transform(X)
        
        assert scaler.is_fitted == True
        assert X_scaled.shape == X.shape
        # Check mean is approximately 0
        np.testing.assert_almost_equal(X_scaled.mean(axis=0), 0, decimal=10)
    
    def test_minmax_scaler(self):
        """Test MinMaxScaler"""
        scaler = FeatureScaler(scaler_type='minmax')
        
        X = np.array([[1, 2], [3, 4], [5, 6]])
        X_scaled = scaler.fit_transform(X)
        
        assert scaler.is_fitted == True
        # Check range is [0, 1]
        assert X_scaled.min() >= 0
        assert X_scaled.max() <= 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
