"""Unit tests for NNDR privacy metric."""

import pytest
import sys
from pathlib import Path

# Add project root and src to path for imports
project_root = Path(__file__).parent.parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from valtapyV2.infrastructure.metrics.privacy.nndr import NNDRMetric
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


class TestNNDRMetric:
    """Test suite for NNDR privacy metric."""
    
    def test_metric_initialization(self):
        """Test that the metric can be initialized correctly."""
        metric = NNDRMetric()
        
        assert metric.name == "nndr"
        assert metric.family == "privacy"
        assert isinstance(metric.purpose_tags, set)
        assert "privacy" in metric.purpose_tags
        assert "distance_based" in metric.purpose_tags
    
    def test_identical_datasets_high_privacy_risk(self):
        """Test NNDR with identical datasets (should indicate privacy risk)."""
        real_data, synth_data = create_identical_datasets(n_samples=50)
        
        real_df = convert_to_pandas_mock(real_data)
        synth_df = convert_to_pandas_mock(synth_data)
        
        metric = NNDRMetric()
        context = create_mock_context()
        
        metric.fit(real_df, synth_df, context)
        result = metric.compute()
        
        assert_metric_result_structure(result, "privacy.nndr", "privacy")
        
        # Identical datasets should show privacy risk (low privacy scores)
        if "error" not in result.details:
            assert_value_in_range(result.value, 0.0, 0.6)
            assert_no_error_in_details(result.details)
    
    def test_completely_different_datasets_good_privacy(self):
        """Test NNDR with completely different datasets (should have better privacy)."""
        real_data, synth_data = create_completely_different_datasets(n_samples=60)
        
        real_df = convert_to_pandas_mock(real_data)
        synth_df = convert_to_pandas_mock(synth_data)
        
        metric = NNDRMetric()
        context = create_mock_context()
        
        metric.fit(real_df, synth_df, context)
        result = metric.compute()
        
        assert_metric_result_structure(result, "privacy.nndr", "privacy")
        
        # Different datasets should have better privacy (higher scores)
        if "error" not in result.details:
            assert_value_in_range(result.value, 0.3, 1.0)
            assert_no_error_in_details(result.details)
    
    def test_correlated_datasets_moderate_privacy(self):
        """Test NNDR with correlated datasets."""
        real_data, synth_data = generate_correlated_datasets(
            n_samples=70,
            correlation=0.7,
            noise_level=0.2
        )
        
        real_df = convert_to_pandas_mock(real_data)
        synth_df = convert_to_pandas_mock(synth_data)
        
        metric = NNDRMetric()
        context = create_mock_context()
        
        metric.fit(real_df, synth_df, context)
        result = metric.compute()
        
        assert_metric_result_structure(result, "privacy.nndr", "privacy")
        
        # Correlated datasets should have moderate privacy
        if "error" not in result.details:
            assert_value_in_range(result.value, 0.0, 1.0)
            assert_no_error_in_details(result.details)
    
    def test_no_numeric_columns(self):
        """Test NNDR with no numeric columns."""
        real_data = [
            {"cat1": "A", "cat2": "X", "target": "high"},
            {"cat1": "B", "cat2": "Y", "target": "low"},
            {"cat1": "A", "cat2": "X", "target": "medium"}
        ] * 10
        
        synth_data = [
            {"cat1": "A", "cat2": "Y", "target": "high"},
            {"cat1": "B", "cat2": "X", "target": "low"},
            {"cat1": "C", "cat2": "Z", "target": "medium"}
        ] * 10
        
        real_df = convert_to_pandas_mock(real_data)
        synth_df = convert_to_pandas_mock(synth_data)
        
        metric = NNDRMetric()
        context = create_mock_context()
        
        metric.fit(real_df, synth_df, context)
        result = metric.compute()
        
        # Should handle gracefully
        assert_metric_result_structure(result, "privacy.nndr", "privacy")
        assert result.value == 0.0
        assert "error" in result.details
    
    def test_insufficient_neighbors(self):
        """Test NNDR with insufficient data for k=5 neighbors."""
        # Only 3 real data points, but NNDR needs k=5
        real_data = [
            {"numeric1": 1, "numeric2": 2, "target": "A"},
            {"numeric1": 2, "numeric2": 3, "target": "B"},
            {"numeric1": 3, "numeric2": 4, "target": "C"}
        ]
        
        synth_data = [
            {"numeric1": 1.1, "numeric2": 2.1, "target": "A"},
            {"numeric1": 2.1, "numeric2": 3.1, "target": "B"}
        ]
        
        real_df = convert_to_pandas_mock(real_data)
        synth_df = convert_to_pandas_mock(synth_data)
        
        metric = NNDRMetric()
        context = create_mock_context()
        
        metric.fit(real_df, synth_df, context)
        result = metric.compute()
        
        # Should handle insufficient neighbors gracefully
        assert_metric_result_structure(result, "privacy.nndr", "privacy")
        # May have error or use available neighbors
    
    def test_nndr_details_structure(self):
        """Test the detailed structure of NNDR results."""
        real_data = generate_tabular_data(n_samples=80, n_numeric=4, n_categorical=1, seed=42)
        synth_data = generate_tabular_data(n_samples=80, n_numeric=4, n_categorical=1, seed=43)
        
        real_df = convert_to_pandas_mock(real_data)
        synth_df = convert_to_pandas_mock(synth_data)
        
        metric = NNDRMetric()
        context = create_mock_context()
        
        metric.fit(real_df, synth_df, context)
        result = metric.compute()
        
        if "error" not in result.details:
            # Check for expected details structure
            expected_keys = [
                "mean_nndr",
                "median_nndr",
                "min_nndr",
                "max_nndr",
                "std_nndr",
                "n_synthetic_points",
                "n_valid_ratios",
                "n_low_ratios",
                "fraction_low_ratios",
                "low_ratio_threshold"
            ]
            
            assert_has_required_details(result.details, expected_keys)
            
            # Check value types and ranges
            assert isinstance(result.details["mean_nndr"], float)
            assert isinstance(result.details["median_nndr"], float)
            assert isinstance(result.details["min_nndr"], float)
            assert isinstance(result.details["max_nndr"], float)
            assert isinstance(result.details["n_synthetic_points"], int)
            assert isinstance(result.details["n_valid_ratios"], int)
            assert isinstance(result.details["fraction_low_ratios"], float)
            
            # Check reasonable ranges
            assert result.details["min_nndr"] >= 0.0
            assert result.details["max_nndr"] >= result.details["min_nndr"]
            assert 0.0 <= result.details["fraction_low_ratios"] <= 1.0
            assert result.details["n_synthetic_points"] > 0
    
    def test_nndr_percentiles_information(self):
        """Test that NNDR includes percentile information."""
        real_data = generate_tabular_data(n_samples=60, n_numeric=3, seed=42)
        synth_data = generate_tabular_data(n_samples=60, n_numeric=3, seed=43)
        
        real_df = convert_to_pandas_mock(real_data)
        synth_df = convert_to_pandas_mock(synth_data)
        
        metric = NNDRMetric()
        context = create_mock_context()
        
        metric.fit(real_df, synth_df, context)
        result = metric.compute()
        
        if "error" not in result.details:
            # Check for percentile keys
            percentile_keys = ["nndr_p10", "nndr_p25", "nndr_p75", "nndr_p90"]
            
            for key in percentile_keys:
                if key in result.details:
                    assert isinstance(result.details[key], float)
                    assert result.details[key] >= 0.0
    
    def test_low_ratio_detection(self):
        """Test NNDR's ability to detect potentially problematic low ratios."""
        # Create synthetic data very close to real data (privacy risk)
        real_data = []
        synth_data = []
        
        for i in range(40):
            real_row = {
                "feature1": i,
                "feature2": i * 2,
                "feature3": i * 0.5,
                "target": "test"
            }
            # Synthetic data very close to real (small noise)
            synth_row = {
                "feature1": i + 0.001,  # Very small differences
                "feature2": i * 2 + 0.002,
                "feature3": i * 0.5 + 0.001,
                "target": "test"
            }
            real_data.append(real_row)
            synth_data.append(synth_row)
        
        real_df = convert_to_pandas_mock(real_data)
        synth_df = convert_to_pandas_mock(synth_data)
        
        metric = NNDRMetric()
        context = create_mock_context()
        
        metric.fit(real_df, synth_df, context)
        result = metric.compute()
        
        assert_metric_result_structure(result, "privacy.nndr", "privacy")
        
        if "error" not in result.details:
            # Should detect low ratios due to close synthetic points
            assert "n_low_ratios" in result.details
            assert "fraction_low_ratios" in result.details
            assert isinstance(result.details["n_low_ratios"], int)
            assert isinstance(result.details["fraction_low_ratios"], float)
            
            # Privacy score should be lower due to close points
            assert result.value < 0.8
    
    def test_zero_distance_handling(self):
        """Test NNDR handling of zero distances (identical points)."""
        # Create some identical points (zero distance)
        real_data = [
            {"num1": 1, "num2": 2},
            {"num1": 3, "num2": 4},
            {"num1": 5, "num2": 6},
            {"num1": 7, "num2": 8},
            {"num1": 9, "num2": 10}
        ] * 6  # Repeat to have enough points
        
        # Some synthetic points are identical to real points
        synth_data = [
            {"num1": 1, "num2": 2},  # Identical to real
            {"num1": 3.1, "num2": 4.1},  # Close but not identical
            {"num1": 100, "num2": 200},  # Very different
        ] * 10
        
        real_df = convert_to_pandas_mock(real_data)
        synth_df = convert_to_pandas_mock(synth_data)
        
        metric = NNDRMetric()
        context = create_mock_context()
        
        metric.fit(real_df, synth_df, context)
        result = metric.compute()
        
        # Should handle zero distances gracefully
        assert_metric_result_structure(result, "privacy.nndr", "privacy")
        assert_value_in_range(result.value, 0.0, 1.0)
    
    def test_single_feature_nndr(self):
        """Test NNDR with only one numeric feature."""
        real_data = [{"feature": i, "cat": "A"} for i in range(1, 21)]
        synth_data = [{"feature": i + 0.5, "cat": "B"} for i in range(1, 21)]
        
        real_df = convert_to_pandas_mock(real_data)
        synth_df = convert_to_pandas_mock(synth_data)
        
        metric = NNDRMetric()
        context = create_mock_context()
        
        metric.fit(real_df, synth_df, context)
        result = metric.compute()
        
        assert_metric_result_structure(result, "privacy.nndr", "privacy")
        # Should work with single feature
        if "error" not in result.details:
            assert_value_in_range(result.value, 0.0, 1.0)
            assert_no_error_in_details(result.details)
    
    def test_missing_values_handling(self):
        """Test NNDR with missing values."""
        real_data = []
        synth_data = []
        
        for i in range(30):
            real_data.append({
                "feature1": i if i % 5 != 0 else None,  # Some missing values
                "feature2": i * 2,
                "feature3": i * -1 if i % 7 != 0 else None,
                "target": "test"
            })
            synth_data.append({
                "feature1": i + 0.1 if i % 6 != 0 else None,
                "feature2": (i + 0.1) * 2,
                "feature3": (i + 0.1) * -1,
                "target": "test"
            })
        
        real_df = convert_to_pandas_mock(real_data)
        synth_df = convert_to_pandas_mock(synth_data)
        
        metric = NNDRMetric()
        context = create_mock_context()
        
        metric.fit(real_df, synth_df, context)
        result = metric.compute()
        
        # Should handle missing values gracefully (via base class fillna)
        assert_metric_result_structure(result, "privacy.nndr", "privacy")
        assert_value_in_range(result.value, 0.0, 1.0)
    
    def test_edge_cases_robustness(self):
        """Test various edge cases for robustness."""
        test_cases = [
            MetricTestCase(
                name="constant_values",
                real_data=[{"num": 5.0}] * 20,
                synth_data=[{"num": 5.0}] * 20,
                expected_value_range=(0.0, 0.5),  # Should show privacy risk
                description="All constant values"
            ),
            MetricTestCase(
                name="extreme_outliers",
                real_data=[{"num": -1000000}, {"num": 1000000}] * 15,
                synth_data=[{"num": -999999}, {"num": 999999}] * 15,
                expected_value_range=(0.0, 1.0),
                description="Extreme outlier values"
            ),
            MetricTestCase(
                name="mixed_scales",
                real_data=[
                    {"small": 0.001, "large": 1000000}
                    for _ in range(25)
                ],
                synth_data=[
                    {"small": 0.002, "large": 999999}
                    for _ in range(25)
                ],
                expected_value_range=(0.0, 1.0),
                description="Features with very different scales"
            )
        ]
        
        for test_case in test_cases:
            real_df = convert_to_pandas_mock(test_case.real_data)
            synth_df = convert_to_pandas_mock(test_case.synth_data)
            
            metric = NNDRMetric()
            context = create_mock_context()
            
            metric.fit(real_df, synth_df, context)
            result = metric.compute()
            
            # Basic structure validation
            assert_metric_result_structure(result, "privacy.nndr", "privacy")
            
            if test_case.should_succeed:
                min_val, max_val = test_case.expected_value_range
                assert_value_in_range(result.value, min_val, max_val)
    
    def test_caching_integration(self):
        """Test that NNDR works with caching for KNN computations."""
        from valtapyV2.infrastructure.runtime.cache import StatsStore
        
        real_data, synth_data = generate_correlated_datasets(n_samples=40, correlation=0.8)
        real_df = convert_to_pandas_mock(real_data)
        synth_df = convert_to_pandas_mock(synth_data)
        
        # Create context with stats store
        stats_store = StatsStore()
        context = {"stats_store": stats_store}
        
        metric = NNDRMetric()
        metric.fit(real_df, synth_df, context)
        result = metric.compute()
        
        # Should complete without errors
        assert_metric_result_structure(result, "privacy.nndr", "privacy")
        
        # Check that cache was potentially used
        cache_stats = stats_store.get_stats()
        assert isinstance(cache_stats, dict)
    
    @pytest.mark.parametrize("n_samples,expected_complexity", [
        (20, "simple"),
        (100, "moderate"),
        (200, "complex")
    ])
    def test_different_dataset_sizes(self, n_samples, expected_complexity):
        """Test NNDR with different dataset sizes."""
        real_data = generate_tabular_data(n_samples, n_numeric=3, seed=42)
        synth_data = generate_tabular_data(n_samples, n_numeric=3, seed=43)
        
        real_df = convert_to_pandas_mock(real_data)
        synth_df = convert_to_pandas_mock(synth_data)
        
        metric = NNDRMetric()
        context = create_mock_context()
        
        metric.fit(real_df, synth_df, context)
        result = metric.compute()
        
        assert_metric_result_structure(result, "privacy.nndr", "privacy")
        assert_value_in_range(result.value, 0.0, 1.0)
        
        if "error" not in result.details:
            assert_no_error_in_details(result.details)
            assert result.details["n_synthetic_points"] == n_samples
    
    def test_privacy_score_interpretation(self):
        """Test that NNDR privacy scores have correct interpretation."""
        # Test case 1: Very risky scenario (identical data)
        real_data_risky = [{"num": i} for i in range(30)]
        synth_data_risky = [{"num": i} for i in range(30)]  # Identical
        
        real_df_risky = convert_to_pandas_mock(real_data_risky)
        synth_df_risky = convert_to_pandas_mock(synth_data_risky)
        
        metric_risky = NNDRMetric()
        context = create_mock_context()
        
        metric_risky.fit(real_df_risky, synth_df_risky, context)
        result_risky = metric_risky.compute()
        
        # Test case 2: Safer scenario (very different data)
        real_data_safe = [{"num": i} for i in range(30)]
        synth_data_safe = [{"num": i + 1000} for i in range(30)]  # Very different
        
        real_df_safe = convert_to_pandas_mock(real_data_safe)
        synth_df_safe = convert_to_pandas_mock(synth_data_safe)
        
        metric_safe = NNDRMetric()
        metric_safe.fit(real_df_safe, synth_df_safe, context)
        result_safe = metric_safe.compute()
        
        # Risky scenario should have lower privacy score than safe scenario
        if "error" not in result_risky.details and "error" not in result_safe.details:
            assert result_risky.value < result_safe.value, \
                f"Risky scenario ({result_risky.value}) should have lower privacy than safe scenario ({result_safe.value})"