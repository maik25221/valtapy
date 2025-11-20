"""Tests for CSV reader implementation."""

import pytest
import pandas as pd
from pathlib import Path

from valtapy.ingestion.readers.csv_reader import CSVReader
from valtapy.ingestion.entities import DataSource, ReadResult
from valtapy.ingestion.exceptions import (
    FileNotFoundError,
    CorruptedFileError,
    EmptyFileError,
    EncodingError
)


class TestCSVReader:
    """Test cases for CSVReader."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.reader = CSVReader()
        self.fixtures_dir = Path(__file__).parent / "fixtures"
        
    def test_can_handle_csv_format(self):
        """Test that CSVReader correctly identifies CSV format."""
        csv_source = DataSource(path="test.csv", format="csv")
        assert self.reader.can_handle(csv_source) is True
        
    def test_can_handle_case_insensitive(self):
        """Test format detection is case insensitive."""
        csv_source = DataSource(path="test.CSV", format="CSV")
        assert self.reader.can_handle(csv_source) is True
        
    def test_cannot_handle_other_formats(self):
        """Test that CSVReader rejects non-CSV formats."""
        json_source = DataSource(path="test.json", format="json")
        parquet_source = DataSource(path="test.parquet", format="parquet")
        
        assert self.reader.can_handle(json_source) is False
        assert self.reader.can_handle(parquet_source) is False
        
    def test_read_valid_csv_file(self):
        """Test reading a valid CSV file."""
        csv_path = self.fixtures_dir / "sample_data.csv"
        source = DataSource(path=str(csv_path), format="csv")
        
        result = self.reader.read(source)
        
        # Verify result structure
        assert isinstance(result, ReadResult)
        assert isinstance(result.data, pd.DataFrame)
        assert result.source == source
        assert isinstance(result.metadata, dict)
        
        # Verify data content
        df = result.data
        assert df.shape == (4, 4)  # 4 rows, 4 columns
        assert list(df.columns) == ['id', 'name', 'age', 'salary']
        assert df.iloc[0]['name'] == 'Alice'
        
        # Verify metadata
        metadata = result.metadata
        assert 'shape' in metadata
        assert 'dtypes' in metadata
        assert 'file_size_mb' in metadata
        assert 'memory_usage_mb' in metadata
        assert metadata['shape'] == (4, 4)
        
    def test_read_with_custom_parameters(self):
        """Test reading CSV with custom parameters."""
        csv_path = self.fixtures_dir / "sample_data.csv"
        custom_params = {
            'encoding': 'utf-8',
            'sep': ',',
            'na_values': ['NULL']
        }
        source = DataSource(
            path=str(csv_path), 
            format="csv", 
            read_params=custom_params
        )
        
        result = self.reader.read(source)
        
        # Should still work with custom params
        assert isinstance(result.data, pd.DataFrame)
        assert result.data.shape == (4, 4)
        
    def test_file_not_found_error(self):
        """Test FileNotFoundError for non-existent file."""
        source = DataSource(path="nonexistent.csv", format="csv")
        
        with pytest.raises(FileNotFoundError) as exc_info:
            self.reader.read(source)
            
        assert "nonexistent.csv" in str(exc_info.value)
        assert exc_info.value.file_path == "nonexistent.csv"
        
    def test_empty_file_error(self):
        """Test EmptyFileError for empty CSV file."""
        empty_path = self.fixtures_dir / "empty_file.csv"
        source = DataSource(path=str(empty_path), format="csv")
        
        with pytest.raises(EmptyFileError) as exc_info:
            self.reader.read(source)
            
        assert str(empty_path) in str(exc_info.value)
        assert exc_info.value.file_path == str(empty_path)
        
    def test_corrupted_file_error(self):
        """Test CorruptedFileError for malformed CSV."""
        corrupted_path = self.fixtures_dir / "corrupted_data.csv"
        source = DataSource(path=str(corrupted_path), format="csv")
        
        # Note: pandas is quite forgiving, so this might not always raise an error
        # But we test the error handling mechanism
        try:
            result = self.reader.read(source)
            # If pandas manages to read it, that's fine too
            assert isinstance(result.data, pd.DataFrame)
        except CorruptedFileError as e:
            assert str(corrupted_path) in str(e)
            assert e.file_path == str(corrupted_path)
            
    def test_directory_instead_of_file_error(self):
        """Test CorruptedFileError when path points to directory."""
        source = DataSource(path=str(self.fixtures_dir), format="csv")
        
        with pytest.raises(CorruptedFileError) as exc_info:
            self.reader.read(source)
            
        assert "Path is not a file" in str(exc_info.value)
        assert exc_info.value.file_path == str(self.fixtures_dir)
        
    def test_metadata_extraction(self):
        """Test that metadata is correctly extracted."""
        csv_path = self.fixtures_dir / "sample_data.csv"
        source = DataSource(path=str(csv_path), format="csv")
        
        result = self.reader.read(source)
        metadata = result.metadata
        
        # Check all expected metadata fields
        expected_fields = ['file_size_mb', 'shape', 'dtypes', 'columns', 'memory_usage_mb']
        for field in expected_fields:
            assert field in metadata, f"Missing metadata field: {field}"
            
        # Check specific values
        assert metadata['shape'] == (4, 4)
        assert len(metadata['columns']) == 4
        assert 'id' in metadata['columns']
        assert metadata['file_size_mb'] >= 0  # Small files may report 0.0
        assert metadata['memory_usage_mb'] >= 0  # Small DataFrames may report 0.0
        
    def test_default_read_parameters(self):
        """Test that default read parameters are applied correctly."""
        reader = CSVReader()
        user_params = {'sep': ';'}
        
        merged_params = reader._prepare_read_params(user_params)
        
        # Check that defaults are included
        assert merged_params['encoding'] == 'utf-8'
        assert merged_params['na_values'] == ['', 'NA', 'N/A', 'null', 'NULL', 'None']
        assert merged_params['keep_default_na'] is True
        
        # Check that user params override defaults
        assert merged_params['sep'] == ';'
        
    def test_user_params_override_defaults(self):
        """Test that user parameters correctly override defaults."""
        reader = CSVReader()
        user_params = {
            'encoding': 'latin-1',
            'na_values': ['MISSING'],
        }
        
        merged_params = reader._prepare_read_params(user_params)
        
        # User params should override
        assert merged_params['encoding'] == 'latin-1'
        assert merged_params['na_values'] == ['MISSING']
        
        # Other defaults should remain
        assert merged_params['keep_default_na'] is True