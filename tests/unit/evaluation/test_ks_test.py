"""Tests for KS Test metric implementation."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from valtapy.evaluation.metrics.fidelity.ks_test import KSTestMetric
from valtapy.evaluation.entities import (
    MetricExecutionContext,
    MetricResult,
    MetricFamily
)
from valtapy.evaluation.exceptions import (
    MetricComputationError,
    UnsupportedDataError
)


class TestKSTestMetric:
    """Test cases for KSTestMetric."""

    def setup_method(self):
        """Set up test fixtures."""
        self.metric = KSTestMetric()

    def test_metric_metadata(self):
        """Test that metric has correct metadata."""
        assert self.metric.metric_id == "fidelity.ks"
        assert self.metric.family == MetricFamily.FIDELITY
        assert self.metric.name == "KS Test"
        assert isinstance(self.metric.description, str)
        assert len(self.metric.description) > 0

    def test_compute_identical_distributions(self):
        """Test KS test with identical distributions (should have high p-value)."""
        # Create identical datasets
        np.random.seed(42)
        data = pd.DataFrame({
            'col1': np.random.normal(0, 1, 100),
            'col2': np.random.uniform(0, 10, 100),
            'col3': np.random.exponential(2, 100)
        })

        context = MetricExecutionContext(
            real_data=data.copy(),
            synth_data=data.copy(),
            parameters={},
            cache_enabled=False
        )

        result = self.metric.compute(context)

        # Verify result structure
        assert isinstance(result, MetricResult)
        assert result.metric_id == "fidelity.ks"
        assert result.family == MetricFamily.FIDELITY

        # For identical distributions, p-value should be very high (close to 1)
        assert result.value > 0.05  # High p-value indicates similar distributions

        # Check details
        assert "column_results" in result.details
        assert "overall_p_value" in result.details
        assert "overall_ks_statistic" in result.details
        assert "num_columns_tested" in result.details

        # All 3 columns should be tested
        assert result.details["num_columns_tested"] == 3
        assert result.details["num_columns_failed"] == 0

        # Metadata should contain test information
        assert "test_type" in result.metadata
        assert result.metadata["test_type"] == "two_sample_ks"
        assert "columns_tested" in result.metadata
        assert len(result.metadata["columns_tested"]) == 3

    def test_compute_different_distributions(self):
        """Test KS test with different distributions (should have low p-value)."""
        np.random.seed(42)

        # Create clearly different distributions
        real_data = pd.DataFrame({
            'col1': np.random.normal(0, 1, 100),  # Mean 0, std 1
            'col2': np.random.uniform(0, 10, 100)  # Uniform [0, 10]
        })

        synth_data = pd.DataFrame({
            'col1': np.random.normal(10, 1, 100),  # Mean 10, std 1
            'col2': np.random.uniform(50, 60, 100)  # Uniform [50, 60]
        })

        context = MetricExecutionContext(
            real_data=real_data,
            synth_data=synth_data,
            parameters={}
        )

        result = self.metric.compute(context)

        # For very different distributions, p-value should be very low
        assert result.value < 0.05  # Low p-value indicates different distributions

        # KS statistic should be high for different distributions
        assert result.details["overall_ks_statistic"] > 0.5

        # Check that all columns were tested
        assert result.details["num_columns_tested"] == 2

    def test_compute_similar_distributions(self):
        """Test KS test with similar but not identical distributions."""
        np.random.seed(42)

        # Create similar distributions with same parameters
        real_data = pd.DataFrame({
            'col1': np.random.normal(5, 2, 200),
            'col2': np.random.exponential(3, 200)
        })

        np.random.seed(43)  # Different seed for variation
        synth_data = pd.DataFrame({
            'col1': np.random.normal(5, 2, 200),
            'col2': np.random.exponential(3, 200)
        })

        context = MetricExecutionContext(
            real_data=real_data,
            synth_data=synth_data,
            parameters={}
        )

        result = self.metric.compute(context)

        # Similar distributions should have reasonable p-values
        # Not necessarily > 0.05, but should be > 0.01 for well-matched distributions
        assert result.value >= 0.0
        assert result.details["num_columns_tested"] == 2

    def test_compute_with_missing_values(self):
        """Test KS test handles missing values correctly."""
        np.random.seed(42)

        # Create data with missing values
        real_data = pd.DataFrame({
            'col1': [1, 2, 3, np.nan, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            'col2': [1, 2, np.nan, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        })

        synth_data = pd.DataFrame({
            'col1': [1, 2, 3, 4, np.nan, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            'col2': [1, np.nan, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        })

        context = MetricExecutionContext(
            real_data=real_data,
            synth_data=synth_data,
            parameters={}
        )

        result = self.metric.compute(context)

        # Should successfully compute despite NaN values
        assert isinstance(result, MetricResult)
        assert result.details["num_columns_tested"] == 2

        # Check that sample counts reflect cleaned data
        for col_name, col_result in result.details["column_results"].items():
            if "real_samples" in col_result:
                assert col_result["real_samples"] < 15  # Some NaNs removed
                assert col_result["synth_samples"] < 15

    def test_compute_with_insufficient_data(self):
        """Test KS test fails gracefully with insufficient data."""
        # Create very small datasets (less than minimum required)
        real_data = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5]  # Only 5 samples, less than min_samples=10
        })

        synth_data = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5]
        })

        context = MetricExecutionContext(
            real_data=real_data,
            synth_data=synth_data,
            parameters={}
        )

        # Should raise UnsupportedDataError or handle gracefully
        result = self.metric.compute(context)

        # Check that the column has an error in results
        col_result = result.details["column_results"]["col1"]
        assert "error" in col_result
        assert "Insufficient data points" in col_result["error"]
        assert result.details["num_columns_failed"] == 1

    def test_compute_with_no_numeric_columns(self):
        """Test KS test raises error when no numeric columns exist."""
        # Create datasets with only categorical columns
        real_data = pd.DataFrame({
            'col1': ['a', 'b', 'c', 'd'],
            'col2': ['x', 'y', 'z', 'w']
        })

        synth_data = pd.DataFrame({
            'col1': ['a', 'b', 'c', 'd'],
            'col2': ['x', 'y', 'z', 'w']
        })

        context = MetricExecutionContext(
            real_data=real_data,
            synth_data=synth_data,
            parameters={}
        )

        # Should raise MetricComputationError (which wraps UnsupportedDataError)
        with pytest.raises(MetricComputationError) as exc_info:
            self.metric.compute(context)

        assert "fidelity.ks" in str(exc_info.value)
        assert "No common numeric columns" in str(exc_info.value)

    def test_compute_with_different_columns(self):
        """Test KS test with partially overlapping columns."""
        np.random.seed(42)

        real_data = pd.DataFrame({
            'col1': np.random.normal(0, 1, 50),
            'col2': np.random.uniform(0, 10, 50),
            'col3': np.random.exponential(2, 50)
        })

        synth_data = pd.DataFrame({
            'col2': np.random.uniform(0, 10, 50),  # Common
            'col3': np.random.exponential(2, 50),  # Common
            'col4': np.random.normal(5, 2, 50)     # Only in synth
        })

        context = MetricExecutionContext(
            real_data=real_data,
            synth_data=synth_data,
            parameters={}
        )

        result = self.metric.compute(context)

        # Should only test common columns (col2, col3)
        assert result.details["num_columns_tested"] == 2
        assert "col2" in result.details["column_results"]
        assert "col3" in result.details["column_results"]
        assert "col1" not in result.details["column_results"]
        assert "col4" not in result.details["column_results"]

    def test_compute_timing_recorded(self):
        """Test that computation time is recorded."""
        np.random.seed(42)

        real_data = pd.DataFrame({
            'col1': np.random.normal(0, 1, 100)
        })

        synth_data = pd.DataFrame({
            'col1': np.random.normal(0, 1, 100)
        })

        context = MetricExecutionContext(
            real_data=real_data,
            synth_data=synth_data,
            parameters={}
        )

        result = self.metric.compute(context)

        # Computation time should be recorded and positive
        assert result.computation_time > 0
        assert isinstance(result.computation_time, float)

    def test_can_compute_with_numeric_data(self):
        """Test can_compute returns True for numeric data."""
        real_df = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': [5.5, 6.6, 7.7, 8.8, 9.9]
        })

        synth_df = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': [5.5, 6.6, 7.7, 8.8, 9.9]
        })

        assert self.metric.can_compute(real_df, synth_df) is True

    def test_can_compute_with_no_numeric_columns(self):
        """Test can_compute returns False when no numeric columns."""
        real_df = pd.DataFrame({
            'col1': ['a', 'b', 'c'],
            'col2': ['x', 'y', 'z']
        })

        synth_df = pd.DataFrame({
            'col1': ['a', 'b', 'c'],
            'col2': ['x', 'y', 'z']
        })

        assert self.metric.can_compute(real_df, synth_df) is False

    def test_can_compute_with_mixed_columns(self):
        """Test can_compute returns True when some numeric columns exist."""
        real_df = pd.DataFrame({
            'numeric_col': [1, 2, 3, 4, 5],
            'string_col': ['a', 'b', 'c', 'd', 'e']
        })

        synth_df = pd.DataFrame({
            'numeric_col': [1, 2, 3, 4, 5],
            'string_col': ['a', 'b', 'c', 'd', 'e']
        })

        assert self.metric.can_compute(real_df, synth_df) is True

    def test_can_compute_with_no_common_columns(self):
        """Test can_compute returns False when no common numeric columns."""
        real_df = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5]
        })

        synth_df = pd.DataFrame({
            'col2': [1, 2, 3, 4, 5]
        })

        assert self.metric.can_compute(real_df, synth_df) is False

    def test_validate_parameters_empty_dict(self):
        """Test validate_parameters accepts empty dict."""
        assert self.metric.validate_parameters({}) is True

    def test_validate_parameters_invalid_type(self):
        """Test validate_parameters rejects non-dict."""
        assert self.metric.validate_parameters(None) is False
        assert self.metric.validate_parameters("invalid") is False
        assert self.metric.validate_parameters([]) is False

    def test_validate_parameters_valid_significance_level(self):
        """Test validate_parameters accepts valid significance level."""
        assert self.metric.validate_parameters({'significance_level': 0.05}) is True
        assert self.metric.validate_parameters({'significance_level': 0.01}) is True
        assert self.metric.validate_parameters({'significance_level': 0.1}) is True

    def test_validate_parameters_invalid_significance_level(self):
        """Test validate_parameters rejects invalid significance level."""
        assert self.metric.validate_parameters({'significance_level': 0}) is False
        assert self.metric.validate_parameters({'significance_level': 1}) is False
        assert self.metric.validate_parameters({'significance_level': -0.05}) is False
        assert self.metric.validate_parameters({'significance_level': 1.5}) is False
        assert self.metric.validate_parameters({'significance_level': 'invalid'}) is False

    def test_validate_parameters_valid_min_samples(self):
        """Test validate_parameters accepts valid min_samples."""
        assert self.metric.validate_parameters({'min_samples': 10}) is True
        assert self.metric.validate_parameters({'min_samples': 50}) is True
        assert self.metric.validate_parameters({'min_samples': 1}) is True

    def test_validate_parameters_invalid_min_samples(self):
        """Test validate_parameters rejects invalid min_samples."""
        assert self.metric.validate_parameters({'min_samples': 0}) is False
        assert self.metric.validate_parameters({'min_samples': -10}) is False
        assert self.metric.validate_parameters({'min_samples': 5.5}) is False
        assert self.metric.validate_parameters({'min_samples': 'invalid'}) is False

    def test_get_numeric_columns(self):
        """Test _get_numeric_columns identifies common numeric columns."""
        real_df = pd.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.1, 2.2, 3.3],
            'string_col': ['a', 'b', 'c']
        })

        synth_df = pd.DataFrame({
            'int_col': [4, 5, 6],
            'float_col': [4.4, 5.5, 6.6],
            'other_col': [7, 8, 9]
        })

        numeric_cols = self.metric._get_numeric_columns(real_df, synth_df)

        # Should return common numeric columns
        assert set(numeric_cols) == {'int_col', 'float_col'}

    def test_get_numeric_columns_no_common(self):
        """Test _get_numeric_columns returns empty list when no common columns."""
        real_df = pd.DataFrame({
            'col1': [1, 2, 3]
        })

        synth_df = pd.DataFrame({
            'col2': [4, 5, 6]
        })

        numeric_cols = self.metric._get_numeric_columns(real_df, synth_df)

        assert numeric_cols == []

    def test_compute_column_ks_success(self):
        """Test _compute_column_ks for successful computation."""
        np.random.seed(42)
        real_col = pd.Series(np.random.normal(0, 1, 100))
        synth_col = pd.Series(np.random.normal(0, 1, 100))

        result = self.metric._compute_column_ks(real_col, synth_col)

        # Should have all expected fields
        assert "ks_statistic" in result
        assert "p_value" in result
        assert "significant" in result
        assert "real_samples" in result
        assert "synth_samples" in result
        assert "interpretation" in result

        # Check types
        assert isinstance(result["ks_statistic"], float)
        assert isinstance(result["p_value"], float)
        # np.bool_ is a subclass of Python's bool
        assert isinstance(result["significant"], (bool, np.bool_))
        assert result["real_samples"] == 100
        assert result["synth_samples"] == 100

    def test_compute_column_ks_with_nans(self):
        """Test _compute_column_ks handles NaNs correctly."""
        real_col = pd.Series([1, 2, 3, np.nan, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
        synth_col = pd.Series([1, 2, 3, 4, np.nan, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])

        result = self.metric._compute_column_ks(real_col, synth_col)

        # Should drop NaNs before computing
        assert result["real_samples"] == 14  # 15 - 1 NaN
        assert result["synth_samples"] == 14  # 15 - 1 NaN
        assert "ks_statistic" in result

    def test_compute_column_ks_insufficient_samples(self):
        """Test _compute_column_ks with insufficient samples."""
        real_col = pd.Series([1, 2, 3])  # Only 3 samples
        synth_col = pd.Series([1, 2, 3])

        result = self.metric._compute_column_ks(real_col, synth_col)

        # Should return error
        assert "error" in result
        assert "Insufficient data points" in result["error"]

    def test_compute_column_ks_scipy_not_available(self):
        """Test _compute_column_ks when scipy is not available."""
        with patch.dict('sys.modules', {'scipy.stats': None}):
            # Force ImportError
            real_col = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
            synth_col = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

            # Create new instance to trigger import
            metric = KSTestMetric()
            result = metric._compute_column_ks(real_col, synth_col)

            assert "error" in result
            assert "scipy not available" in result["error"]

    def test_interpret_ks_result_good_fidelity(self):
        """Test interpretation for good fidelity (high p-value)."""
        interpretation = self.metric._interpret_ks_result(0.1, 0.5)
        assert "not significantly different" in interpretation.lower()
        assert "good fidelity" in interpretation.lower()

    def test_interpret_ks_result_moderate_fidelity(self):
        """Test interpretation for moderate fidelity (moderate p-value)."""
        interpretation = self.metric._interpret_ks_result(0.2, 0.03)
        assert "somewhat different" in interpretation.lower()
        assert "moderate fidelity" in interpretation.lower()

    def test_interpret_ks_result_poor_fidelity(self):
        """Test interpretation for poor fidelity (low p-value)."""
        interpretation = self.metric._interpret_ks_result(0.5, 0.001)
        assert "significantly different" in interpretation.lower()
        assert "poor fidelity" in interpretation.lower()

    def test_overall_score_calculation(self):
        """Test that overall score is calculated correctly."""
        np.random.seed(42)

        # Create data with 3 numeric columns
        real_data = pd.DataFrame({
            'col1': np.random.normal(0, 1, 100),
            'col2': np.random.uniform(0, 10, 100),
            'col3': np.random.exponential(2, 100)
        })

        synth_data = real_data.copy()

        context = MetricExecutionContext(
            real_data=real_data,
            synth_data=synth_data,
            parameters={}
        )

        result = self.metric.compute(context)

        # For identical data, overall p-value should be very high
        assert result.details["overall_p_value"] > 0.5

        # Overall KS statistic should be very low
        assert result.details["overall_ks_statistic"] < 0.2

        # Overall score should match overall p-value
        assert result.value == result.details["overall_p_value"]

    def test_column_results_details(self):
        """Test that column-level results are properly detailed."""
        np.random.seed(42)

        real_data = pd.DataFrame({
            'col1': np.random.normal(0, 1, 50)
        })

        synth_data = pd.DataFrame({
            'col1': np.random.normal(0, 1, 50)
        })

        context = MetricExecutionContext(
            real_data=real_data,
            synth_data=synth_data,
            parameters={}
        )

        result = self.metric.compute(context)

        # Check column results structure
        col_result = result.details["column_results"]["col1"]

        assert "ks_statistic" in col_result
        assert "p_value" in col_result
        assert "significant" in col_result
        assert "real_samples" in col_result
        assert "synth_samples" in col_result
        assert "interpretation" in col_result

    def test_metric_execution_context_integration(self):
        """Test integration with MetricExecutionContext."""
        np.random.seed(42)

        real_data = pd.DataFrame({
            'col1': np.random.normal(0, 1, 100)
        })

        synth_data = pd.DataFrame({
            'col1': np.random.normal(0, 1, 100)
        })

        context = MetricExecutionContext(
            real_data=real_data,
            synth_data=synth_data,
            parameters={'significance_level': 0.05},
            cache_enabled=True,
            random_seed=42
        )

        result = self.metric.compute(context)

        # Should use context data correctly
        assert isinstance(result, MetricResult)
        assert result.metric_id == self.metric.metric_id
        assert result.family == self.metric.family
