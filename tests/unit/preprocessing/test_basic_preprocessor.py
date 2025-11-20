"""Tests for basic preprocessor implementation."""

import pytest
import pandas as pd
from pathlib import Path

from valtapy.preprocessing.processors.basic_preprocessor import BasicDataPreprocessor
from valtapy.preprocessing.entities import PreprocessingConfig, PreprocessingResult
from valtapy.preprocessing.exceptions import PreprocessingFailedError


class TestBasicDataPreprocessor:
    """Test cases for BasicDataPreprocessor."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.preprocessor = BasicDataPreprocessor()
        self.fixtures_dir = Path(__file__).parent / "fixtures"
    
    def test_can_handle_valid_dataframes(self):
        """Test that preprocessor can handle valid DataFrames."""
        real_df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
        synth_df = pd.DataFrame({"col1": [3, 4], "col2": ["c", "d"]})
        
        assert self.preprocessor.can_handle(real_df, synth_df) is True
    
    def test_cannot_handle_empty_dataframes(self):
        """Test that preprocessor rejects empty DataFrames."""
        real_df = pd.DataFrame({"col": [1, 2]})
        empty_df = pd.DataFrame()
        
        assert self.preprocessor.can_handle(real_df, empty_df) is False
        assert self.preprocessor.can_handle(empty_df, real_df) is False
    
    def test_cannot_handle_non_dataframes(self):
        """Test that preprocessor rejects non-DataFrame inputs."""
        real_df = pd.DataFrame({"col": [1, 2]})
        
        assert self.preprocessor.can_handle(real_df, "not_a_dataframe") is False
        assert self.preprocessor.can_handle("not_a_dataframe", real_df) is False
        assert self.preprocessor.can_handle(None, real_df) is False
    
    def test_basic_preprocessing_with_identical_datasets(self):
        """Test preprocessing with identical datasets."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35]
        })
        real_df = df.copy()
        synth_df = df.copy()
        config = PreprocessingConfig()
        
        result = self.preprocessor.preprocess(real_df, synth_df, config)
        
        # Verify result structure
        assert isinstance(result, PreprocessingResult)
        assert isinstance(result.real_data, pd.DataFrame)
        assert isinstance(result.synth_data, pd.DataFrame)
        assert result.config == config
        assert result.processing_time > 0
        
        # Data should be unchanged for identical datasets
        assert result.real_data.equals(real_df)
        assert result.synth_data.equals(synth_df)
    
    def test_preprocessing_detects_column_mismatches(self):
        """Test that preprocessing detects column mismatches."""
        real_df = pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]})
        synth_df = pd.DataFrame({"id": [1, 2], "age": [25, 30]})
        config = PreprocessingConfig()
        
        result = self.preprocessor.preprocess(real_df, synth_df, config)
        
        # Should detect compatibility issues
        issues = result.report.compatibility_issues
        assert len(issues) > 0
        assert any("missing" in issue.lower() for issue in issues)
    
    def test_empty_data_cleaning_disabled(self):
        """Test preprocessing with empty data cleaning disabled."""
        real_df = pd.DataFrame({
            "id": [1, None, 3],
            "name": ["Alice", None, "Charlie"]
        })
        synth_df = real_df.copy()
        config = PreprocessingConfig(
            drop_empty_rows=False,
            drop_empty_columns=False
        )
        
        result = self.preprocessor.preprocess(real_df, synth_df, config)
        
        # Data should be unchanged when cleaning is disabled
        assert result.real_data.equals(real_df)
        assert result.synth_data.equals(synth_df)
        assert "empty_data_cleaning" not in result.report.operations_applied
    
    def test_empty_data_cleaning_enabled(self):
        """Test preprocessing with empty data cleaning enabled."""
        # Create DataFrames with completely empty rows/columns
        real_df = pd.DataFrame({
            "id": [1, None, 3],
            "name": ["Alice", None, "Charlie"],
            "empty_col": [None, None, None]  # Completely empty column
        })
        # Add a completely empty row
        real_df.loc[len(real_df)] = [None, None, None]
        
        synth_df = real_df.copy()
        config = PreprocessingConfig(
            drop_empty_rows=True,
            drop_empty_columns=True
        )
        
        result = self.preprocessor.preprocess(real_df, synth_df, config)
        
        # Should have applied cleaning
        assert "empty_data_cleaning" in result.report.operations_applied
        
        # Should have removed empty column and row
        assert "empty_col" not in result.real_data.columns
        assert "empty_col" not in result.synth_data.columns
        assert len(result.real_data) < len(real_df)
        assert len(result.synth_data) < len(synth_df)
    
    def test_preprocessing_preserves_original_data(self):
        """Test that preprocessing doesn't modify original DataFrames."""
        original_real = pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]})
        original_synth = pd.DataFrame({"id": [3, 4], "name": ["Charlie", "Diana"]})
        
        # Make copies to compare later
        real_copy = original_real.copy()
        synth_copy = original_synth.copy()
        
        config = PreprocessingConfig(drop_empty_rows=True)
        
        self.preprocessor.preprocess(original_real, original_synth, config)
        
        # Original DataFrames should be unchanged
        assert original_real.equals(real_copy)
        assert original_synth.equals(synth_copy)
    
    def test_preprocessing_report_contains_expected_fields(self):
        """Test that preprocessing report contains all expected fields."""
        real_df = pd.DataFrame({"col": [1, 2]})
        synth_df = pd.DataFrame({"col": [3, 4]})
        config = PreprocessingConfig()
        
        result = self.preprocessor.preprocess(real_df, synth_df, config)
        report = result.report
        
        # Check report structure
        assert hasattr(report, 'real_changes')
        assert hasattr(report, 'synth_changes')
        assert hasattr(report, 'compatibility_issues')
        assert hasattr(report, 'operations_applied')
        
        assert isinstance(report.real_changes, dict)
        assert isinstance(report.synth_changes, dict)
        assert isinstance(report.compatibility_issues, list)
        assert isinstance(report.operations_applied, list)
    
    def test_preprocessing_with_different_shapes(self):
        """Test preprocessing with datasets of different shapes."""
        real_df = pd.DataFrame({"id": [1, 2, 3], "name": ["A", "B", "C"]})
        synth_df = pd.DataFrame({"id": [1, 2], "name": ["X", "Y"]})  # Fewer rows
        config = PreprocessingConfig()
        
        result = self.preprocessor.preprocess(real_df, synth_df, config)
        
        # Should handle different row counts gracefully
        assert isinstance(result, PreprocessingResult)
        # No explicit compatibility issue for different row counts (that's expected)
        # Column count difference would be flagged though
        assert result.real_data.shape[1] == result.synth_data.shape[1]