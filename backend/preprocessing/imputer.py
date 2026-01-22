"""
Missing value imputation module
Handles missing values in input data
"""
import numpy as np
from sklearn.impute import SimpleImputer
import joblib
from typing import Optional


class DataImputer:
    """Handle missing values in data"""
    
    def __init__(self, strategy: str = 'mean'):
        """
        Initialize imputer
        
        Args:
            strategy (str): Imputation strategy ('mean', 'median', 'most_frequent')
        """
        self.strategy = strategy
        self.imputer = SimpleImputer(strategy=strategy, missing_values=np.nan)
        self.is_fitted = False
    
    def fit(self, X: np.ndarray) -> 'DataImputer':
        """
        Fit imputer on training data
        
        Args:
            X (np.ndarray): Training data
        
        Returns:
            self
        """
        self.imputer.fit(X)
        self.is_fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data by imputing missing values
        
        Args:
            X (np.ndarray): Data to transform
        
        Returns:
            np.ndarray: Transformed data
        """
        if not self.is_fitted:
            raise ValueError("Imputer must be fitted before transform")
        
        return self.imputer.transform(X)
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit and transform data
        
        Args:
            X (np.ndarray): Data to fit and transform
        
        Returns:
            np.ndarray: Transformed data
        """
        return self.fit(X).transform(X)
    
    def save(self, filepath: str):
        """
        Save imputer to file
        
        Args:
            filepath (str): Path to save imputer
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted imputer")
        
        joblib.dump(self.imputer, filepath)
        print(f"Imputer saved to {filepath}")
    
    @staticmethod
    def load(filepath: str) -> 'DataImputer':
        """
        Load imputer from file
        
        Args:
            filepath (str): Path to imputer file
        
        Returns:
            DataImputer: Loaded imputer
        """
        imputer_obj = DataImputer()
        imputer_obj.imputer = joblib.load(filepath)
        imputer_obj.is_fitted = True
        imputer_obj.strategy = imputer_obj.imputer.strategy
        print(f"Imputer loaded from {filepath}")
        return imputer_obj
    
    def get_statistics(self) -> Optional[np.ndarray]:
        """
        Get imputation statistics (mean, median, etc.)
        
        Returns:
            np.ndarray: Statistics used for imputation
        """
        if not self.is_fitted:
            return None
        return self.imputer.statistics_
