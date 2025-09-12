"""Unit tests for KS Test fidelity metric."""

import pytest
import sys
from pathlib import Path

# Add project root and src to path for imports
project_root = Path(__file__).parent.parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from valtapyV2.infrastructure.metrics.fidelity.ks_test import KSTestMetric
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


class TestKSTestMetric:
    """Test suite for KS Test fidelity metric."""
    
    def test_metric_initialization(self):
        """Test that the metric can be initialized correctly."""
        metric = KSTestMetric()
        
        assert metric.name == "ks_test"
        assert metric.family == "fidelity"
        assert isinstance(metric.purpose_tags, set)
        assert "fidelity" in metric.purpose_tags
    
    def test_identical_datasets(self):
        """Test KS test with identical datasets (should have high fidelity)."""
        real_data, synth_data = create_identical_datasets(n_samples=50)
        
        # Convert to mock DataFrames
        real_df = convert_to_pandas_mock(real_data)
        synth_df = convert_to_pandas_mock(synth_data)
        
        # Run metric
        metric = KSTestMetric()
        context = create_mock_context()
        
        metric.fit(real_df, synth_df, context)
        result = metric.compute()
        
        # Assertions
        assert_metric_result_structure(result, "fidelity.ks", "fidelity")
        assert_value_in_range(result.value, 0.5, 1.0)  # Should be high for identical data
        assert_no_error_in_details(result.details)
        assert_has_required_details(result.details, ["column_results", "overall_p_value"])
    
    def test_completely_different_datasets(self):
        """Test KS test with completely different datasets (should have low fidelity)."""
        real_data, synth_data = create_completely_different_datasets(n_samples=50)
        
        real_df = convert_to_pandas_mock(real_data)
        synth_df = convert_to_pandas_mock(synth_data)
        
        metric = KSTestMetric()
        context = create_mock_context()
        
        metric.fit(real_df, synth_df, context)
        result = metric.compute()
        
        # Assertions
        assert_metric_result_structure(result, "fidelity.ks", "fidelity")
        assert_value_in_range(result.value, 0.0, 0.5)  # Should be low for different data
        assert_no_error_in_details(result.details)
    
    def test_correlated_datasets(self):
        """Test KS test with correlated datasets."""
        real_data, synth_data = generate_correlated_datasets(
            n_samples=100, 
            correlation=0.8, 
            noise_level=0.1
        )
        
        real_df = convert_to_pandas_mock(real_data)
        synth_df = convert_to_pandas_mock(synth_data)
        
        metric = KSTestMetric()
        context = create_mock_context()
        
        metric.fit(real_df, synth_df, context)
        result = metric.compute()
        
        # Should be moderate fidelity
        assert_metric_result_structure(result, "fidelity.ks", "fidelity")
        assert_value_in_range(result.value, 0.0, 1.0)
        assert_no_error_in_details(result.details)
        
        # Check that column results are present
        assert "column_results" in result.details
        assert isinstance(result.details["column_results"], dict)
    
    def test_no_numeric_columns(self):
        """Test KS test with no numeric columns."""
        # Create data with only categorical columns
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
        
        metric = KSTestMetric()
        context = create_mock_context()
        
        metric.fit(real_df, synth_df, context)
        result = metric.compute()
        
        # Should handle gracefully
        assert_metric_result_structure(result, "fidelity.ks", "fidelity")
        assert result.value == 0.0
        assert "error" in result.details
        assert "No numeric columns" in result.details["error"]
    
    def test_insufficient_data(self):
        """Test KS test with insufficient data points."""
        real_data = [{"numeric1": 1, "target": "high"}]  # Only 1 sample
        synth_data = [{"numeric1": 2, "target": "low"}]
        
        real_df = convert_to_pandas_mock(real_data)
        synth_df = convert_to_pandas_mock(synth_data)
        
        metric = KSTestMetric()
        context = create_mock_context()
        
        metric.fit(real_df, synth_df, context)
        result = metric.compute()
        
        # Should handle gracefully
        assert_metric_result_structure(result, "fidelity.ks", "fidelity")
        # Value might be 0 or low due to insufficient data
    
    def test_missing_values_handling(self):
        """Test KS test with missing values."""
        real_data = [
            {"numeric1": 1.0, "numeric2": 2.0, "target": "high"},
            {"numeric1": None, "numeric2": 3.0, "target": "low"},  # Missing value
            {"numeric1": 3.0, "numeric2": None, "target": "medium"}
        ] * 10  # Repeat to have enough samples
        
        synth_data = [
            {"numeric1": 1.1, "numeric2": 2.1, "target": "high"},
            {"numeric1": 2.1, "numeric2": None, "target": "low"},
            {"numeric1": None, "numeric2": 4.1, "target": "medium"}
        ] * 10
        
        real_df = convert_to_pandas_mock(real_data)
        synth_df = convert_to_pandas_mock(synth_data)
        
        metric = KSTestMetric()
        context = create_mock_context()
        
        metric.fit(real_df, synth_df, context)
        result = metric.compute()
        
        # Should handle missing values by dropping them
        assert_metric_result_structure(result, "fidelity.ks", "fidelity")
        assert_value_in_range(result.value, 0.0, 1.0)
    
    def test_single_column_analysis(self):
        """Test detailed analysis of results for single column."""
        # Create data where we know the expected outcome
        real_data = [{"numeric1": i, "target": "test"} for i in range(1, 51)]  # 1 to 50
        synth_data = [{"numeric1": i + 0.1, "target": "test"} for i in range(1, 51)]  # Very similar
        
        real_df = convert_to_pandas_mock(real_data)
        synth_df = convert_to_pandas_mock(synth_data)
        
        metric = KSTestMetric()
        context = create_mock_context()
        
        metric.fit(real_df, synth_df, context)
        result = metric.compute()
        
        # Check detailed results structure
        assert "column_results" in result.details
        assert "numeric1" in result.details["column_results"]
        
        col_result = result.details["column_results"]["numeric1"]
        assert_has_required_details(col_result, ["ks_statistic", "p_value", "significant"])
        
        # Should not be significant (p_value > 0.05) for very similar data
        assert isinstance(col_result["ks_statistic"], float)
        assert isinstance(col_result["p_value"], float)
        assert isinstance(col_result["significant"], bool)
    
    def test_multiple_columns_aggregation(self):
        """Test aggregation across multiple numeric columns."""
        real_data = generate_tabular_data(n_samples=60, n_numeric=4, n_categorical=1, seed=42)
        synth_data = generate_tabular_data(n_samples=60, n_numeric=4, n_categorical=1, seed=43)
        
        real_df = convert_to_pandas_mock(real_data)
        synth_df = convert_to_pandas_mock(synth_data)
        
        metric = KSTestMetric()
        context = create_mock_context()
        
        metric.fit(real_df, synth_df, context)
        result = metric.compute()
        
        # Check aggregation details
        assert_has_required_details(result.details, [
            "column_results", 
            "overall_p_value", 
            "mean_ks_statistic",
            "n_columns_tested"
        ])
        
        # Should have tested multiple columns
        assert result.details["n_columns_tested"] >= 3
        assert len(result.details["column_results"]) >= 3
    
    def test_edge_cases_and_robustness(self):
        """Test various edge cases for robustness."""
        test_cases = [
            MetricTestCase(
                name="extreme_values",
                real_data=[{"num": -1000000}, {"num": 1000000}] * 25,
                synth_data=[{"num": -999999}, {"num": 999999}] * 25,
                expected_value_range=(0.0, 1.0),
                description="Very large numeric values"
            ),
            MetricTestCase(
                name="zero_values",
                real_data=[{"num": 0}] * 50,
                synth_data=[{"num": 0.1}] * 50,
                expected_value_range=(0.0, 1.0),
                description="Values close to zero"
            ),
            MetricTestCase(
                name="small_differences",
                real_data=[{"num": 1.0}] * 30,
                synth_data=[{"num": 1.0000001}] * 30,
                expected_value_range=(0.5, 1.0),
                description="Very small differences should show high fidelity"
            )
        ]
        
        for test_case in test_cases:
            real_df = convert_to_pandas_mock(test_case.real_data)
            synth_df = convert_to_pandas_mock(test_case.synth_data)
            
            metric = KSTestMetric()
            context = create_mock_context()
            
            metric.fit(real_df, synth_df, context)
            result = metric.compute()
            
            # Basic structure validation
            assert_metric_result_structure(result, "fidelity.ks", "fidelity")
            
            if test_case.should_succeed:
                assert_no_error_in_details(result.details)
                min_val, max_val = test_case.expected_value_range
                assert_value_in_range(result.value, min_val, max_val)
    
    def test_caching_integration(self):
        """Test that the metric works with caching."""
        from valtapyV2.infrastructure.runtime.cache import StatsStore
        
        real_data, synth_data = generate_correlated_datasets(n_samples=30)
        real_df = convert_to_pandas_mock(real_data)
        synth_df = convert_to_pandas_mock(synth_data)
        
        # Create context with stats store
        stats_store = StatsStore()
        context = {"stats_store": stats_store}
        
        metric = KSTestMetric()
        metric.fit(real_df, synth_df, context)
        result = metric.compute()
        
        # Should complete without errors
        assert_metric_result_structure(result, "fidelity.ks", "fidelity")
        
        # Check that cache was potentially used
        cache_stats = stats_store.get_stats()
        assert isinstance(cache_stats, dict)
    
    @pytest.mark.parametrize("n_samples,n_numeric", [
        (20, 1),
        (50, 3), 
        (100, 5)
    ])
    def test_different_dataset_sizes(self, n_samples, n_numeric):
        """Test metric with different dataset sizes and column counts."""
        real_data = generate_tabular_data(n_samples, n_numeric=n_numeric, seed=42)
        synth_data = generate_tabular_data(n_samples, n_numeric=n_numeric, seed=43)
        
        real_df = convert_to_pandas_mock(real_data)
        synth_df = convert_to_pandas_mock(synth_data)
        
        metric = KSTestMetric()
        context = create_mock_context()
        
        metric.fit(real_df, synth_df, context)
        result = metric.compute()
        
        assert_metric_result_structure(result, "fidelity.ks", "fidelity")
        assert_value_in_range(result.value, 0.0, 1.0)
        
        if n_numeric > 0:
            assert_no_error_in_details(result.details)
            assert result.details["n_columns_tested"] == n_numeric