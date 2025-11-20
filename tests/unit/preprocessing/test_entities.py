"""Tests for preprocessing entities."""

import pytest
import pandas as pd

from valtapy.preprocessing.entities import (
    PreprocessingConfig,
    PreprocessingReport,
    PreprocessingResult
)


class TestPreprocessingConfig:
    """Test cases for PreprocessingConfig entity."""
    
    def test_default_config_creation(self):
        """Test creating config with default values."""
        config = PreprocessingConfig()
        
        assert config.drop_empty_rows is True
        assert config.drop_empty_columns is True
        assert config.na_strategy == "keep"
        assert config.na_fill_value is None
        assert config.enforce_consistent_types is True
        assert config.auto_detect_types is True
        assert config.normalize_numeric is False
        assert config.normalize_text is False
        assert config.encode_categorical is False
    
    def test_custom_config_creation(self):
        """Test creating config with custom values."""
        config = PreprocessingConfig(
            drop_empty_rows=False,
            na_strategy="drop",
            normalize_numeric=True,
            na_fill_value=0
        )
        
        assert config.drop_empty_rows is False
        assert config.na_strategy == "drop"
        assert config.normalize_numeric is True
        assert config.na_fill_value == 0
    
    def test_invalid_na_strategy_raises_error(self):
        """Test that invalid na_strategy raises ValueError."""
        with pytest.raises(ValueError, match="na_strategy must be one of"):
            PreprocessingConfig(na_strategy="invalid")
    
    def test_valid_na_strategies(self):
        """Test all valid na_strategy values."""
        valid_strategies = ["keep", "drop", "fill"]
        
        for strategy in valid_strategies:
            config = PreprocessingConfig(na_strategy=strategy)
            assert config.na_strategy == strategy


class TestPreprocessingReport:
    """Test cases for PreprocessingReport entity."""
    
    def test_default_report_creation(self):
        """Test creating report with default values."""
        report = PreprocessingReport()
        
        assert report.real_changes == {}
        assert report.synth_changes == {}
        assert report.compatibility_issues == []
        assert report.operations_applied == []
    
    def test_custom_report_creation(self):
        """Test creating report with custom values."""
        real_changes = {"rows_dropped": 2}
        synth_changes = {"columns_dropped": 1}
        issues = ["Column mismatch"]
        operations = ["cleaning", "typing"]
        
        report = PreprocessingReport(
            real_changes=real_changes,
            synth_changes=synth_changes,
            compatibility_issues=issues,
            operations_applied=operations
        )
        
        assert report.real_changes == real_changes
        assert report.synth_changes == synth_changes
        assert report.compatibility_issues == issues
        assert report.operations_applied == operations
    
    def test_invalid_changes_type_raises_error(self):
        """Test that non-dict changes raise ValueError."""
        with pytest.raises(ValueError, match="real_changes must be a dictionary"):
            PreprocessingReport(real_changes="not_a_dict")
            
        with pytest.raises(ValueError, match="synth_changes must be a dictionary"):
            PreprocessingReport(synth_changes="not_a_dict")


class TestPreprocessingResult:
    """Test cases for PreprocessingResult entity."""
    
    def test_valid_result_creation(self):
        """Test creating a valid PreprocessingResult."""
        real_df = pd.DataFrame({"col": [1, 2]})
        synth_df = pd.DataFrame({"col": [3, 4]})
        config = PreprocessingConfig()
        report = PreprocessingReport()
        
        result = PreprocessingResult(
            real_data=real_df,
            synth_data=synth_df,
            config=config,
            report=report,
            processing_time=1.5
        )
        
        assert result.real_data.equals(real_df)
        assert result.synth_data.equals(synth_df)
        assert result.config == config
        assert result.report == report
        assert result.processing_time == 1.5
    
    def test_result_with_default_processing_time(self):
        """Test PreprocessingResult with default processing time."""
        real_df = pd.DataFrame({"col": [1]})
        synth_df = pd.DataFrame({"col": [2]})
        config = PreprocessingConfig()
        report = PreprocessingReport()
        
        result = PreprocessingResult(
            real_data=real_df,
            synth_data=synth_df,
            config=config,
            report=report
        )
        
        assert result.processing_time == 0.0
    
    def test_none_real_data_raises_error(self):
        """Test that None real_data raises ValueError."""
        synth_df = pd.DataFrame({"col": [1]})
        config = PreprocessingConfig()
        report = PreprocessingReport()
        
        with pytest.raises(ValueError, match="real_data cannot be None"):
            PreprocessingResult(
                real_data=None,
                synth_data=synth_df,
                config=config,
                report=report
            )
    
    def test_none_synth_data_raises_error(self):
        """Test that None synth_data raises ValueError."""
        real_df = pd.DataFrame({"col": [1]})
        config = PreprocessingConfig()
        report = PreprocessingReport()
        
        with pytest.raises(ValueError, match="synth_data cannot be None"):
            PreprocessingResult(
                real_data=real_df,
                synth_data=None,
                config=config,
                report=report
            )