"""Data preprocessing transformers."""

import pandas as pd
from typing import Self, Dict, Any
from ...domain.contracts import Preprocessor
from ...domain.entities import DatasetSpec
from ...domain.errors import PreprocessingError


class TabularPreprocessor:
    """
    Basic tabular data preprocessor implementing the Preprocessor protocol.
    
    TODO: Implement comprehensive preprocessing including:
    - Automatic type inference and conversion
    - Missing value imputation strategies
    - Categorical encoding (one-hot, label encoding, etc.)
    - Numerical scaling and normalization
    - Feature selection and dimensionality reduction
    - Outlier detection and handling
    """
    
    def __init__(self):
        self._is_fitted = False
        self._transformations = {}
        self._spec = None
    
    def fit(self, data: pd.DataFrame, spec: DatasetSpec) -> Self:
        """
        Fit preprocessor to data and dataset specification.
        
        Args:
            data: Training data to fit preprocessor
            spec: Dataset specification with constraints and dtypes
            
        Returns:
            Self for method chaining
        """
        try:
            self._spec = spec
            self._transformations = {}
            
            # TODO: Implement actual fitting logic
            # For now, just record basic statistics
            for col in data.columns:
                if pd.api.types.is_numeric_dtype(data[col]):
                    self._transformations[col] = {
                        "type": "numeric",
                        "mean": data[col].mean(),
                        "std": data[col].std(),
                        "min": data[col].min(),
                        "max": data[col].max()
                    }
                else:
                    self._transformations[col] = {
                        "type": "categorical",
                        "unique_values": data[col].unique().tolist(),
                        "mode": data[col].mode().iloc[0] if len(data[col].mode()) > 0 else None
                    }
            
            self._is_fitted = True
            return self
            
        except Exception as e:
            raise PreprocessingError(f"Failed to fit preprocessor: {e}", step="fit")
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted preprocessor.
        
        Args:
            data: Data to transform
            
        Returns:
            Transformed data
        """
        if not self._is_fitted:
            raise PreprocessingError("Preprocessor must be fitted before transform", step="transform")
        
        try:
            # TODO: Implement actual transformation logic
            # For now, just return a copy of the data
            transformed = data.copy()
            
            return transformed
            
        except Exception as e:
            raise PreprocessingError(f"Failed to transform data: {e}", step="transform", original_error=e)
    
    def metadata(self) -> Dict[str, Any]:
        """Return preprocessing metadata and transformations applied."""
        if not self._is_fitted:
            return {"fitted": False}
        
        return {
            "fitted": True,
            "transformations": self._transformations,
            "n_columns": len(self._transformations),
            "spec": {
                "target": self._spec.target if self._spec else None,
                "dtypes": self._spec.dtypes if self._spec else {},
                "constraints": self._spec.constraints if self._spec else {}
            }
        }


class IdentityPreprocessor:
    """Preprocessor that applies no transformations (identity function)."""
    
    def __init__(self):
        self._is_fitted = False
        self._spec = None
    
    def fit(self, data: pd.DataFrame, spec: DatasetSpec) -> Self:
        """Fit preprocessor (no-op for identity)."""
        self._spec = spec
        self._is_fitted = True
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data (no-op for identity)."""
        if not self._is_fitted:
            raise PreprocessingError("Preprocessor must be fitted before transform")
        return data.copy()
    
    def metadata(self) -> Dict[str, Any]:
        """Return metadata."""
        return {
            "fitted": self._is_fitted,
            "transformations": "identity",
            "spec": {
                "target": self._spec.target if self._spec else None
            }
        }


class StandardScaler:
    """Standard scaling preprocessor for numerical columns."""
    
    def __init__(self, columns: list[str] | None = None):
        self.columns = columns  # If None, will auto-detect numeric columns
        self._means = {}
        self._stds = {}
        self._is_fitted = False
        self._spec = None
    
    def fit(self, data: pd.DataFrame, spec: DatasetSpec) -> Self:
        """Fit scaler to data."""
        self._spec = spec
        
        # Auto-detect numeric columns if not specified
        if self.columns is None:
            self.columns = data.select_dtypes(include=['number']).columns.tolist()
        
        # Calculate means and standard deviations
        for col in self.columns:
            if col in data.columns:
                self._means[col] = data[col].mean()
                self._stds[col] = data[col].std()
        
        self._is_fitted = True
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply standard scaling."""
        if not self._is_fitted:
            raise PreprocessingError("StandardScaler must be fitted before transform")
        
        transformed = data.copy()
        
        for col in self.columns:
            if col in transformed.columns and col in self._means:
                mean = self._means[col]
                std = self._stds[col]
                if std > 0:  # Avoid division by zero
                    transformed[col] = (transformed[col] - mean) / std
        
        return transformed
    
    def metadata(self) -> Dict[str, Any]:
        """Return scaler metadata."""
        return {
            "fitted": self._is_fitted,
            "transformations": "standard_scaling",
            "columns": self.columns,
            "means": self._means,
            "stds": self._stds
        }