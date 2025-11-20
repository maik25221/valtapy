"""Tests for ingestion exceptions."""

import pytest

from valtapy.ingestion.exceptions import (
    IngestionError,
    FileNotFoundError,
    UnsupportedFormatError,
    CorruptedFileError,
    EncodingError,
    EmptyFileError,
    DatasetMismatchError
)


class TestIngestionExceptions:
    """Test cases for custom ingestion exceptions."""
    
    def test_base_ingestion_error(self):
        """Test base IngestionError."""
        error = IngestionError("Base error message")
        
        assert str(error) == "Base error message"
        assert isinstance(error, Exception)
        
    def test_file_not_found_error(self):
        """Test FileNotFoundError with file path."""
        error = FileNotFoundError("/path/to/missing.csv")
        
        assert error.file_path == "/path/to/missing.csv"
        assert "Data file not found: /path/to/missing.csv" in str(error)
        assert isinstance(error, IngestionError)
        
    def test_unsupported_format_error(self):
        """Test UnsupportedFormatError with format info."""
        error = UnsupportedFormatError("xlsx", ["csv", "json", "parquet"])
        
        assert error.format_name == "xlsx"
        assert error.supported_formats == ["csv", "json", "parquet"]
        assert "Unsupported format 'xlsx'" in str(error)
        assert "csv, json, parquet" in str(error)
        assert isinstance(error, IngestionError)
        
    def test_corrupted_file_error(self):
        """Test CorruptedFileError with file and reason."""
        error = CorruptedFileError("/path/to/file.csv", "Invalid CSV format")
        
        assert error.file_path == "/path/to/file.csv"
        assert error.reason == "Invalid CSV format"
        assert "Corrupted file /path/to/file.csv: Invalid CSV format" in str(error)
        assert isinstance(error, IngestionError)
        
    def test_encoding_error(self):
        """Test EncodingError with encoding details."""
        error = EncodingError("/path/to/file.csv", "utf-8", "invalid start byte")
        
        assert error.file_path == "/path/to/file.csv"
        assert error.encoding == "utf-8"
        assert error.reason == "invalid start byte"
        assert "Encoding error in /path/to/file.csv with utf-8" in str(error)
        assert "invalid start byte" in str(error)
        assert isinstance(error, IngestionError)
        
    def test_empty_file_error(self):
        """Test EmptyFileError with file path."""
        error = EmptyFileError("/path/to/empty.csv")
        
        assert error.file_path == "/path/to/empty.csv"
        assert "Data file is empty: /path/to/empty.csv" in str(error)
        assert isinstance(error, IngestionError)
        
    def test_dataset_mismatch_error(self):
        """Test DatasetMismatchError with dataset info."""
        real_info = {"columns": ["a", "b"], "shape": (100, 2)}
        synth_info = {"columns": ["a", "c"], "shape": (50, 2)}
        
        error = DatasetMismatchError(
            "Different column names", 
            real_info, 
            synth_info
        )
        
        assert error.reason == "Different column names"
        assert error.real_info == real_info
        assert error.synth_info == synth_info
        assert "Dataset mismatch: Different column names" in str(error)
        assert isinstance(error, IngestionError)
        
    def test_exception_inheritance_chain(self):
        """Test that all exceptions inherit correctly."""
        # All custom exceptions should inherit from IngestionError
        exceptions = [
            FileNotFoundError("test.csv"),
            UnsupportedFormatError("xlsx", ["csv"]),
            CorruptedFileError("test.csv", "reason"),
            EncodingError("test.csv", "utf-8", "reason"),
            EmptyFileError("test.csv"),
            DatasetMismatchError("reason", {}, {})
        ]
        
        for exc in exceptions:
            assert isinstance(exc, IngestionError)
            assert isinstance(exc, Exception)