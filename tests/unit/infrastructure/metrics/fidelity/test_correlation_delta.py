"""Unit tests for Correlation Delta fidelity metric."""

import pytest
import sys
from pathlib import Path

# Add project root and src to path for imports
project_root = Path(__file__).parent.parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from valtapyV2.infrastructure.metrics.fidelity.correlation_delta import CorrelationDeltaMetric
from valtapyV2.domain.entities import MetricResult
from tests.utils.data_generators import (
    generate_tabular_data,
    create_identical_datasets,
    create_completely_different_datasets,
    generate_correlated_datasets
)
from tests.utils.test_helpers import (
    assert_metric_result_structure,
    assert_value_in_range,
    assert_no_error_in_details,
    assert_has_required_details,
    create_mock_context,
    convert_to_pandas_mock,
    MetricTestCase
)


class MockCorrelationMatrix:
    """Mock correlation matrix for testing."""
    
    def __init__(self, data):
        self.data = data
        self.columns = list(data.keys()) if data else []
    
    def loc(self, rows, cols):
        # Simple mock that returns self for chaining
        return self
    
    @property
    def values(self):
        # Return a simple 2D array representation
        if not self.data:
            return []
        
        size = len(self.columns)
        matrix = []
        for i in range(size):
            row = []
            for j in range(size):
                if i == j:
                    row.append(1.0)  # Perfect correlation with self
                else:
                    # Some correlation value
                    row.append(0.5)
            matrix.append(row)
        return matrix


class TestCorrelationDeltaMetric:
    """Test suite for Correlation Delta fidelity metric."""
    
    def test_metric_initialization(self):
        """Test that the metric can be initialized correctly."""
        metric = CorrelationDeltaMetric()
        
        assert metric.name == "correlation_delta"
        assert metric.family == "fidelity"
        assert isinstance(metric.purpose_tags, set)
        assert "fidelity" in metric.purpose_tags
        assert "correlation_preservation" in metric.purpose_tags
    
    def test_identical_correlation_matrices(self):
        """Test with identical correlation patterns."""
        # Create datasets with strong correlations
        real_data = []
        synth_data = []
        
        for i in range(50):
            real_row = {
                "numeric1": i,
                "numeric2": i * 2,  # Perfect correlation
                "numeric3": i * -1, # Negative correlation
                "target": "test"
            }
            synth_row = {
                "numeric1": i + 0.1,
                "numeric2": (i + 0.1) * 2,  # Same correlation pattern
                "numeric3": (i + 0.1) * -1,
                "target": "test"
            }
            real_data.append(real_row)
            synth_data.append(synth_row)
        
        real_df = convert_to_pandas_mock(real_data)
        synth_df = convert_to_pandas_mock(synth_data)
        
        metric = CorrelationDeltaMetric()
        context = create_mock_context()
        
        metric.fit(real_df, synth_df, context)
        result = metric.compute()
        
        assert_metric_result_structure(result, "fidelity.correlation_delta", "fidelity")
        # Should have high fidelity for similar correlation patterns
        assert_value_in_range(result.value, 0.5, 1.0)
        assert_no_error_in_details(result.details)
    
    def test_different_correlation_patterns(self):
        """Test with very different correlation patterns."""
        real_data = []
        synth_data = []
        
        for i in range(50):
            real_row = {
                "numeric1": i,
                "numeric2": i * 2,      # Correlated in real data
                "numeric3": i * -1,
                "target": "test"
            }
            synth_row = {
                "numeric1": i,
                "numeric2": (50 - i),   # Different correlation pattern
                "numeric3": i * 0.1,
                "target": "test"
            }
            real_data.append(real_row)
            synth_data.append(synth_row)
        
        real_df = convert_to_pandas_mock(real_data)
        synth_df = convert_to_pandas_mock(synth_data)
        
        metric = CorrelationDeltaMetric()
        context = create_mock_context()
        
        metric.fit(real_df, synth_df, context)
        result = metric.compute()
        
        assert_metric_result_structure(result, "fidelity.correlation_delta", "fidelity")
        # Should have lower fidelity for different patterns
        assert_value_in_range(result.value, 0.0, 0.8)
        assert_no_error_in_details(result.details)
    
    def test_no_numeric_columns(self):
        """Test correlation delta with no numeric columns."""
        real_data = [
            {"cat1": "A", "cat2": "X", "target": "high"},
            {"cat1": "B", "cat2": "Y", "target": "low"},
            {"cat1": "A", "cat2": "X", "target": "medium"}
        ]
        synth_data = [
            {"cat1": "A", "cat2": "Y", "target": "high"},
            {"cat1": "B", "cat2": "X", "target": "low"},
            {"cat1": "C", "cat2": "Z", "target": "medium"}
        ]
        
        real_df = convert_to_pandas_mock(real_data)
        synth_df = convert_to_pandas_mock(synth_data)
        
        metric = CorrelationDeltaMetric()
        context = create_mock_context()
        
        metric.fit(real_df, synth_df, context)
        result = metric.compute()
        
        # Should handle gracefully
        assert_metric_result_structure(result, "fidelity.correlation_delta", "fidelity")
        assert result.value == 0.0
        assert "error" in result.details
        assert "No numeric columns" in result.details["error"]
    
    def test_single_numeric_column(self):
        """Test correlation delta with only one numeric column."""
        real_data = [{"numeric1": i, "cat": "A"} for i in range(20)]
        synth_data = [{"numeric1": i + 0.5, "cat": "B"} for i in range(20)]
        
        real_df = convert_to_pandas_mock(real_data)
        synth_df = convert_to_pandas_mock(synth_data)
        
        metric = CorrelationDeltaMetric()
        context = create_mock_context()
        
        metric.fit(real_df, synth_df, context)
        result = metric.compute()
        
        # Should handle gracefully - need at least 2 columns for correlation
        assert_metric_result_structure(result, "fidelity.correlation_delta", "fidelity")
        assert result.value == 0.0
        assert "error" in result.details
        assert "Need at least 2" in result.details["error"]
    
    def test_correlation_details_structure(self):
        """Test detailed structure of correlation delta results."""
        real_data = generate_tabular_data(n_samples=40, n_numeric=3, n_categorical=1, seed=42)
        synth_data = generate_tabular_data(n_samples=40, n_numeric=3, n_categorical=1, seed=43)
        
        real_df = convert_to_pandas_mock(real_data)
        synth_df = convert_to_pandas_mock(synth_data)
        
        metric = CorrelationDeltaMetric()
        context = create_mock_context()
        
        metric.fit(real_df, synth_df, context)
        result = metric.compute()
        
        # Check detailed results structure
        expected_details = [
            "mean_absolute_difference",
            "max_absolute_difference", 
            "rmse",
            "correlation_similarity",
            "n_correlations_compared",
            "n_variables"
        ]
        assert_has_required_details(result.details, expected_details)
        
        # Check value types
        assert isinstance(result.details["mean_absolute_difference"], float)
        assert isinstance(result.details["max_absolute_difference"], float)
        assert isinstance(result.details["rmse"], float)
        assert isinstance(result.details["n_correlations_compared"], int)
        assert isinstance(result.details["n_variables"], int)
        
        # Check reasonable ranges
        assert 0.0 <= result.details["mean_absolute_difference"] <= 2.0
        assert 0.0 <= result.details["max_absolute_difference"] <= 2.0
        assert result.details["n_variables"] >= 2
        assert result.details["n_correlations_compared"] >= 1
    
    def test_identical_datasets_perfect_correlation(self):
        """Test with identical datasets should give perfect correlation preservation."""
        real_data, synth_data = create_identical_datasets(n_samples=30)
        
        real_df = convert_to_pandas_mock(real_data)
        synth_df = convert_to_pandas_mock(synth_data)
        
        metric = CorrelationDeltaMetric()
        context = create_mock_context()
        
        metric.fit(real_df, synth_df, context)
        result = metric.compute()
        
        assert_metric_result_structure(result, "fidelity.correlation_delta", "fidelity")
        # Should have very high fidelity (near 1.0) for identical datasets
        assert_value_in_range(result.value, 0.8, 1.0)
        assert_no_error_in_details(result.details)
        
        # Mean absolute difference should be very small
        assert result.details["mean_absolute_difference"] < 0.2
    
    def test_completely_different_datasets_poor_correlation(self):
        """Test with completely different datasets."""
        real_data, synth_data = create_completely_different_datasets(n_samples=30)
        
        real_df = convert_to_pandas_mock(real_data)
        synth_df = convert_to_pandas_mock(synth_data)
        
        metric = CorrelationDeltaMetric()
        context = create_mock_context()
        
        metric.fit(real_df, synth_df, context)
        result = metric.compute()
        
        assert_metric_result_structure(result, "fidelity.correlation_delta", "fidelity")
        # Should have lower fidelity for different datasets
        assert_value_in_range(result.value, 0.0, 0.7)
        assert_no_error_in_details(result.details)
    
    def test_missing_values_handling(self):
        """Test correlation delta with missing values."""
        real_data = []
        synth_data = []
        
        for i in range(30):
            real_row = {
                "numeric1": i if i % 5 != 0 else None,  # Some missing values
                "numeric2": i * 2,
                "numeric3": i * -1 if i % 7 != 0 else None,
                "target": "test"
            }
            synth_row = {
                "numeric1": i + 0.1 if i % 6 != 0 else None,
                "numeric2": (i + 0.1) * 2,
                "numeric3": (i + 0.1) * -1,
                "target": "test"
            }
            real_data.append(real_row)
            synth_data.append(synth_row)
        
        real_df = convert_to_pandas_mock(real_data)
        synth_df = convert_to_pandas_mock(synth_data)
        
        metric = CorrelationDeltaMetric()
        context = create_mock_context()
        
        metric.fit(real_df, synth_df, context)
        result = metric.compute()
        
        # Should handle missing values gracefully
        assert_metric_result_structure(result, "fidelity.correlation_delta", "fidelity")
        assert_value_in_range(result.value, 0.0, 1.0)
    
    def test_edge_cases_robustness(self):
        """Test various edge cases for robustness."""
        test_cases = [
            MetricTestCase(
                name="constant_values",
                real_data=[{"num1": 5, "num2": 10}] * 20,
                synth_data=[{"num1": 5, "num2": 10}] * 20,
                expected_value_range=(0.5, 1.0),
                description="Constant values should show high correlation preservation"
            ),
            MetricTestCase(
                name="extreme_correlations",
                real_data=[{"num1": i, "num2": i * 1000} for i in range(1, 21)],
                synth_data=[{"num1": i, "num2": i * 999} for i in range(1, 21)],
                expected_value_range=(0.7, 1.0),
                description="Strong correlations with slight differences"
            ),
            MetricTestCase(
                name="zero_correlation_real",
                real_data=[{"num1": i, "num2": 50 - i} for i in range(1, 21)],
                synth_data=[{"num1": i, "num2": 50 - i} for i in range(1, 21)],
                expected_value_range=(0.8, 1.0),
                description="Zero correlation preserved"
            )
        ]
        
        for test_case in test_cases:
            real_df = convert_to_pandas_mock(test_case.real_data)
            synth_df = convert_to_pandas_mock(test_case.synth_data)
            
            metric = CorrelationDeltaMetric()
            context = create_mock_context()
            
            metric.fit(real_df, synth_df, context)
            result = metric.compute()
            
            # Basic structure validation
            assert_metric_result_structure(result, "fidelity.correlation_delta", "fidelity")
            
            if test_case.should_succeed:
                assert_no_error_in_details(result.details)
                min_val, max_val = test_case.expected_value_range
                assert_value_in_range(result.value, min_val, max_val)
    
    def test_caching_integration(self):
        """Test that the metric works with caching (specifically correlation caching)."""
        from valtapyV2.infrastructure.runtime.cache import StatsStore
        
        real_data, synth_data = generate_correlated_datasets(n_samples=25, correlation=0.9)
        real_df = convert_to_pandas_mock(real_data)
        synth_df = convert_to_pandas_mock(synth_data)
        
        # Create context with stats store
        stats_store = StatsStore()
        context = {"stats_store": stats_store}
        
        metric = CorrelationDeltaMetric()
        metric.fit(real_df, synth_df, context)
        result = metric.compute()
        
        # Should complete without errors
        assert_metric_result_structure(result, "fidelity.correlation_delta", "fidelity")
        assert_no_error_in_details(result.details)
        
        # Check that cache was potentially used
        cache_stats = stats_store.get_stats()
        assert isinstance(cache_stats, dict)
    
    @pytest.mark.parametrize("n_numeric", [2, 3, 5])
    def test_different_numeric_column_counts(self, n_numeric):
        """Test metric with different numbers of numeric columns."""
        real_data = generate_tabular_data(50, n_numeric=n_numeric, n_categorical=1, seed=42)
        synth_data = generate_tabular_data(50, n_numeric=n_numeric, n_categorical=1, seed=43)
        
        real_df = convert_to_pandas_mock(real_data)
        synth_df = convert_to_pandas_mock(synth_data)
        
        metric = CorrelationDeltaMetric()
        context = create_mock_context()
        
        metric.fit(real_df, synth_df, context)
        result = metric.compute()
        
        assert_metric_result_structure(result, "fidelity.correlation_delta", "fidelity")
        assert_value_in_range(result.value, 0.0, 1.0)
        assert_no_error_in_details(result.details)
        
        # Check that the right number of variables were analyzed
        assert result.details["n_variables"] == n_numeric
        
        # Number of correlations should be n*(n-1)/2
        expected_correlations = n_numeric * (n_numeric - 1) // 2
        assert result.details["n_correlations_compared"] == expected_correlations