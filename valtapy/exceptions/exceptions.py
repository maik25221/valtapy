"""
Custom exceptions for ML validation processes.
"""


class ValidationError(Exception):
    """Base exception for validation errors."""


class DataPreparationError(ValidationError):
    """Raised when data preparation fails."""


class ModelTrainingError(ValidationError):
    """Raised when model training fails."""


class MetricsCalculationError(ValidationError):
    """Raised when metrics calculation fails."""


class InvalidDataError(ValidationError):
    """Raised when input data is invalid."""
