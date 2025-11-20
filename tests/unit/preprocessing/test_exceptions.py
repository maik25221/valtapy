"""Tests for preprocessing exceptions."""

import pytest

from valtapy.preprocessing.exceptions import (
    PreprocessingError,
    IncompatibleDatasetsError,
    InvalidConfigurationError,
    PreprocessingFailedError,
    UnsupportedDataTypeError
)


class TestPreprocessingExceptions:
    """Test cases for custom preprocessing exceptions."""
    
    def test_base_preprocessing_error(self):
        """Test base PreprocessingError."""
        error = PreprocessingError("Base preprocessing error")
        
        assert str(error) == "Base preprocessing error"
        assert isinstance(error, Exception)
    
    def test_incompatible_datasets_error(self):
        """Test IncompatibleDatasetsError with dataset info."""
        real_info = {"columns": ["a", "b"], "shape": (100, 2)}
        synth_info = {"columns": ["a", "c"], "shape": (50, 2)}
        
        error = IncompatibleDatasetsError(
            "Different column names",
            real_info,
            synth_info
        )
        
        assert error.reason == "Different column names"
        assert error.real_info == real_info
        assert error.synth_info == synth_info
        assert "Incompatible datasets: Different column names" in str(error)
        assert isinstance(error, PreprocessingError)
    
    def test_invalid_configuration_error(self):
        """Test InvalidConfigurationError with config details."""
        error = InvalidConfigurationError(
            "na_strategy",
            "invalid_strategy",
            ["keep", "drop", "fill"]
        )
        
        assert error.config_field == "na_strategy"
        assert error.value == "invalid_strategy"
        assert error.valid_options == ["keep", "drop", "fill"]
        assert "Invalid na_strategy: 'invalid_strategy'" in str(error)
        assert "keep, drop, fill" in str(error)
        assert isinstance(error, PreprocessingError)
    
    def test_preprocessing_failed_error(self):
        """Test PreprocessingFailedError with operation details."""
        error = PreprocessingFailedError("type_conversion", "Cannot convert string to int")
        
        assert error.operation == "type_conversion"
        assert error.reason == "Cannot convert string to int"
        assert "Preprocessing failed during type_conversion" in str(error)
        assert "Cannot convert string to int" in str(error)
        assert isinstance(error, PreprocessingError)
    
    def test_unsupported_data_type_error(self):
        """Test UnsupportedDataTypeError with type details."""
        error = UnsupportedDataTypeError(
            "my_column",
            "complex128",
            ["int64", "float64", "object"]
        )
        
        assert error.column_name == "my_column"
        assert error.data_type == "complex128"
        assert error.supported_types == ["int64", "float64", "object"]
        assert "Unsupported data type for column 'my_column': complex128" in str(error)
        assert "int64, float64, object" in str(error)
        assert isinstance(error, PreprocessingError)
    
    def test_exception_inheritance_chain(self):
        """Test that all exceptions inherit correctly."""
        # All custom exceptions should inherit from PreprocessingError
        exceptions = [
            IncompatibleDatasetsError("reason", {}, {}),
            InvalidConfigurationError("field", "value", ["option1"]),
            PreprocessingFailedError("operation", "reason"),
            UnsupportedDataTypeError("column", "type", ["supported"])
        ]
        
        for exc in exceptions:
            assert isinstance(exc, PreprocessingError)
            assert isinstance(exc, Exception)