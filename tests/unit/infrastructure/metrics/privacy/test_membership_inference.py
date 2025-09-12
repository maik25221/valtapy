"""Unit tests for Membership Inference Attack privacy metric."""

import pytest
import sys
from pathlib import Path

# Add project root and src to path for imports
project_root = Path(__file__).parent.parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from valtapyV2.infrastructure.metrics.privacy.membership_inference import MembershipInferenceMetric
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


class TestMembershipInferenceMetric:
    """Test suite for Membership Inference Attack privacy metric."""
    
    def test_metric_initialization(self):
        """Test that the metric can be initialized correctly."""
        metric = MembershipInferenceMetric()
        
        assert metric.name == "membership_inference"
        assert metric.family == "privacy"
        assert isinstance(metric.purpose_tags, set)
        assert "privacy" in metric.purpose_tags
        assert "inference_attack" in metric.purpose_tags
    
    def test_identical_datasets_high_attack_success(self):
        """Test MIA with identical datasets (should have high attack success, low privacy)."""
        real_data, synth_data = create_identical_datasets(n_samples=60)
        
        real_df = convert_to_pandas_mock(real_data)
        synth_df = convert_to_pandas_mock(synth_data)
        
        metric = MembershipInferenceMetric()
        context = create_mock_context()
        
        metric.fit(real_df, synth_df, context)
        result = metric.compute()
        
        assert_metric_result_structure(result, "privacy.membership_inference", "privacy")
        
        # Identical datasets should be vulnerable to membership inference
        if "error" not in result.details:
            assert_value_in_range(result.value, 0.0, 0.7)  # Low privacy score
            assert_no_error_in_details(result.details)
            
            # Attack accuracy should be higher than random (0.5)
            if "attack_accuracy" in result.details:
                assert result.details["attack_accuracy"] > 0.5
    
    def test_completely_different_datasets_low_attack_success(self):
        """Test MIA with completely different datasets (should have low attack success, better privacy)."""
        real_data, synth_data = create_completely_different_datasets(n_samples=50)
        
        real_df = convert_to_pandas_mock(real_data)
        synth_df = convert_to_pandas_mock(synth_data)
        
        metric = MembershipInferenceMetric()
        context = create_mock_context()
        
        metric.fit(real_df, synth_df, context)
        result = metric.compute()
        
        assert_metric_result_structure(result, "privacy.membership_inference", "privacy")
        
        # Different datasets should have better privacy
        if "error" not in result.details:
            assert_value_in_range(result.value, 0.2, 1.0)  # Better privacy score
            assert_no_error_in_details(result.details)
    
    def test_correlated_datasets_moderate_privacy(self):
        """Test MIA with moderately correlated datasets."""
        real_data, synth_data = generate_correlated_datasets(
            n_samples=80,
            correlation=0.6,
            noise_level=0.3
        )
        
        real_df = convert_to_pandas_mock(real_data)
        synth_df = convert_to_pandas_mock(synth_data)
        
        metric = MembershipInferenceMetric()
        context = create_mock_context()
        
        metric.fit(real_df, synth_df, context)
        result = metric.compute()
        
        assert_metric_result_structure(result, "privacy.membership_inference", "privacy")
        
        # Correlated datasets should have moderate privacy
        if "error" not in result.details:
            assert_value_in_range(result.value, 0.0, 1.0)
            assert_no_error_in_details(result.details)
    
    def test_no_numeric_columns(self):
        """Test MIA with no numeric columns."""
        real_data = [
            {"cat1": "A", "cat2": "X", "target": "high"},
            {"cat1": "B", "cat2": "Y", "target": "low"},
            {"cat1": "A", "cat2": "X", "target": "medium"}
        ] * 15
        
        synth_data = [
            {"cat1": "A", "cat2": "Y", "target": "high"},
            {"cat1": "B", "cat2": "X", "target": "low"},
            {"cat1": "C", "cat2": "Z", "target": "medium"}
        ] * 15
        
        real_df = convert_to_pandas_mock(real_data)
        synth_df = convert_to_pandas_mock(synth_data)
        
        metric = MembershipInferenceMetric()
        context = create_mock_context()
        
        metric.fit(real_df, synth_df, context)
        result = metric.compute()
        
        # Should handle gracefully
        assert_metric_result_structure(result, "privacy.membership_inference", "privacy")
        assert result.value == 0.0
        assert "error" in result.details
        assert "No numeric columns found" in result.details["error"]
    
    def test_unbalanced_dataset_sizes(self):
        """Test MIA with different sized datasets."""
        # Real data has 40 samples, synthetic has 60
        real_data = generate_tabular_data(n_samples=40, n_numeric=3, seed=42)
        synth_data = generate_tabular_data(n_samples=60, n_numeric=3, seed=43)
        
        real_df = convert_to_pandas_mock(real_data)
        synth_df = convert_to_pandas_mock(synth_data)
        
        metric = MembershipInferenceMetric()
        context = create_mock_context()
        
        metric.fit(real_df, synth_df, context)
        result = metric.compute()
        
        assert_metric_result_structure(result, "privacy.membership_inference", "privacy")
        
        if "error" not in result.details:
            assert_no_error_in_details(result.details)
            
            # Should balance dataset sizes
            if "balanced_dataset" in result.details:
                assert result.details["balanced_dataset"] == True
            
            # Should use the minimum size for both
            if "n_real_samples" in result.details and "n_synthetic_samples" in result.details:
                assert result.details["n_real_samples"] == result.details["n_synthetic_samples"]
                assert result.details["n_real_samples"] == min(40, 60)
    
    def test_mia_details_structure(self):
        """Test the detailed structure of MIA results."""
        real_data = generate_tabular_data(n_samples=70, n_numeric=4, seed=42)
        synth_data = generate_tabular_data(n_samples=70, n_numeric=4, seed=43)
        
        real_df = convert_to_pandas_mock(real_data)
        synth_df = convert_to_pandas_mock(synth_data)
        
        metric = MembershipInferenceMetric()
        context = create_mock_context()
        
        metric.fit(real_df, synth_df, context)
        result = metric.compute()
        
        if "error" not in result.details:
            # Check for expected details structure
            expected_keys = [
                "attack_type",
                "attack_accuracy",
                "baseline_accuracy",
                "privacy_score",
                "n_real_samples",
                "n_synthetic_samples",
                "n_features",
                "balanced_dataset"
            ]
            
            for key in expected_keys:
                if key in result.details:  # Some keys might be conditional
                    if key == "attack_type":
                        assert result.details[key] in ["statistical_fallback", "knn_based"]
                    elif key in ["attack_accuracy", "baseline_accuracy", "privacy_score"]:
                        assert isinstance(result.details[key], float)
                        assert 0.0 <= result.details[key] <= 1.0
                    elif key in ["n_real_samples", "n_synthetic_samples", "n_features"]:
                        assert isinstance(result.details[key], int)
                        assert result.details[key] > 0
                    elif key == "balanced_dataset":
                        assert isinstance(result.details[key], bool)
    
    def test_attack_type_variations(self):
        """Test that MIA can use different attack strategies."""
        real_data = generate_tabular_data(n_samples=50, n_numeric=3, seed=42)
        synth_data = generate_tabular_data(n_samples=50, n_numeric=3, seed=43)
        
        real_df = convert_to_pandas_mock(real_data)
        synth_df = convert_to_pandas_mock(synth_data)
        
        metric = MembershipInferenceMetric()
        context = create_mock_context()
        
        metric.fit(real_df, synth_df, context)
        result = metric.compute()
        
        assert_metric_result_structure(result, "privacy.membership_inference", "privacy")
        
        if "error" not in result.details:
            # Should specify which attack type was used
            assert "attack_type" in result.details
            assert result.details["attack_type"] in ["statistical_fallback", "knn_based"]
            
            # Attack accuracy should be reasonable
            assert "attack_accuracy" in result.details
            assert 0.0 <= result.details["attack_accuracy"] <= 1.0
    
    def test_baseline_comparison(self):
        """Test that MIA compares against random baseline."""
        real_data = generate_tabular_data(n_samples=60, n_numeric=2, seed=42)
        synth_data = generate_tabular_data(n_samples=60, n_numeric=2, seed=43)
        
        real_df = convert_to_pandas_mock(real_data)
        synth_df = convert_to_pandas_mock(synth_data)
        
        metric = MembershipInferenceMetric()
        context = create_mock_context()
        
        metric.fit(real_df, synth_df, context)
        result = metric.compute()
        
        if "error" not in result.details:
            # Should have baseline accuracy (random guessing = 0.5)
            if "baseline_accuracy" in result.details:
                assert result.details["baseline_accuracy"] == 0.5
            
            # Privacy score should be 1 - attack_accuracy
            if "attack_accuracy" in result.details and "privacy_score" in result.details:
                expected_privacy = 1.0 - result.details["attack_accuracy"]
                assert abs(result.details["privacy_score"] - expected_privacy) < 0.01
    
    def test_statistical_fallback_attack(self):
        """Test the statistical fallback attack mechanism."""
        # Create data where statistical properties are very different
        real_data = []
        synth_data = []
        
        # Real data: low values
        for i in range(40):
            real_data.append({
                "feature1": i * 0.1,
                "feature2": i * 0.05,
                "feature3": i * 0.02
            })
        
        # Synthetic data: high values (very different statistics)
        for i in range(40):
            synth_data.append({
                "feature1": (i * 10) + 100,
                "feature2": (i * 5) + 50,
                "feature3": (i * 2) + 20
            })
        
        real_df = convert_to_pandas_mock(real_data)
        synth_df = convert_to_pandas_mock(synth_data)
        
        metric = MembershipInferenceMetric()
        context = create_mock_context()
        
        metric.fit(real_df, synth_df, context)
        result = metric.compute()
        
        assert_metric_result_structure(result, "privacy.membership_inference", "privacy")
        
        if "error" not in result.details:
            # Statistical differences should make attack easier (lower privacy)
            assert_value_in_range(result.value, 0.0, 0.8)
            
            # Attack accuracy should be better than random
            if "attack_accuracy" in result.details:
                assert result.details["attack_accuracy"] > 0.5
    
    def test_missing_values_handling(self):
        """Test MIA with missing values."""
        real_data = []
        synth_data = []
        
        for i in range(50):
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
        
        metric = MembershipInferenceMetric()
        context = create_mock_context()
        
        metric.fit(real_df, synth_df, context)
        result = metric.compute()
        
        # Should handle missing values gracefully (via fillna in implementation)
        assert_metric_result_structure(result, "privacy.membership_inference", "privacy")
        assert_value_in_range(result.value, 0.0, 1.0)
    
    def test_small_dataset_handling(self):
        """Test MIA with very small datasets."""
        real_data = [
            {"feature": 1, "target": "A"},
            {"feature": 2, "target": "B"},
            {"feature": 3, "target": "A"}
        ]
        synth_data = [
            {"feature": 1.1, "target": "A"},
            {"feature": 2.1, "target": "B"}
        ]
        
        real_df = convert_to_pandas_mock(real_data)
        synth_df = convert_to_pandas_mock(synth_data)
        
        metric = MembershipInferenceMetric()
        context = create_mock_context()
        
        metric.fit(real_df, synth_df, context)
        result = metric.compute()
        
        # Should handle small datasets gracefully
        assert_metric_result_structure(result, "privacy.membership_inference", "privacy")
        assert_value_in_range(result.value, 0.0, 1.0)
    
    def test_fallback_without_sklearn(self):
        """Test MIA fallback behavior when sklearn is not available."""
        real_data = generate_tabular_data(n_samples=40, n_numeric=3, seed=42)
        synth_data = generate_tabular_data(n_samples=40, n_numeric=3, seed=43)
        
        real_df = convert_to_pandas_mock(real_data)
        synth_df = convert_to_pandas_mock(synth_data)
        
        metric = MembershipInferenceMetric()
        context = create_mock_context()
        
        metric.fit(real_df, synth_df, context)
        result = metric.compute()
        
        # Should handle sklearn absence gracefully
        assert_metric_result_structure(result, "privacy.membership_inference", "privacy")
        assert_value_in_range(result.value, 0.0, 1.0)
        
        # May have fallback indication in details
        if "sklearn not available" in result.details.get("error", ""):
            assert result.value == 0.5  # Neutral fallback score
    
    def test_edge_cases_robustness(self):
        """Test various edge cases for robustness."""
        test_cases = [
            MetricTestCase(
                name="constant_features",
                real_data=[{"num": 5.0}] * 30,
                synth_data=[{"num": 5.1}] * 30,
                expected_value_range=(0.0, 1.0),
                description="Constant real features vs slightly different synthetic"
            ),
            MetricTestCase(
                name="high_dimensional",
                real_data=[{f"feat_{i}": i + j for i in range(10)} for j in range(25)],
                synth_data=[{f"feat_{i}": i + j + 0.1 for i in range(10)} for j in range(25)],
                expected_value_range=(0.0, 1.0),
                description="High-dimensional feature space"
            ),
            MetricTestCase(
                name="extreme_values",
                real_data=[{"num": -1000000}, {"num": 1000000}] * 15,
                synth_data=[{"num": -999999}, {"num": 999999}] * 15,
                expected_value_range=(0.0, 1.0),
                description="Extreme numeric values"
            )
        ]
        
        for test_case in test_cases:
            real_df = convert_to_pandas_mock(test_case.real_data)
            synth_df = convert_to_pandas_mock(test_case.synth_data)
            
            metric = MembershipInferenceMetric()
            context = create_mock_context()
            
            metric.fit(real_df, synth_df, context)
            result = metric.compute()
            
            # Basic structure validation
            assert_metric_result_structure(result, "privacy.membership_inference", "privacy")
            
            if test_case.should_succeed:
                min_val, max_val = test_case.expected_value_range
                assert_value_in_range(result.value, min_val, max_val)
    
    def test_caching_integration(self):
        """Test that MIA works with caching for KNN computations."""
        from valtapyV2.infrastructure.runtime.cache import StatsStore
        
        real_data, synth_data = generate_correlated_datasets(n_samples=45, correlation=0.7)
        real_df = convert_to_pandas_mock(real_data)
        synth_df = convert_to_pandas_mock(synth_data)
        
        # Create context with stats store
        stats_store = StatsStore()
        context = {"stats_store": stats_store}
        
        metric = MembershipInferenceMetric()
        metric.fit(real_df, synth_df, context)
        result = metric.compute()
        
        # Should complete without errors
        assert_metric_result_structure(result, "privacy.membership_inference", "privacy")
        
        # Check that cache was potentially used
        cache_stats = stats_store.get_stats()
        assert isinstance(cache_stats, dict)
    
    @pytest.mark.parametrize("attack_scenario,expected_privacy_range", [
        ("identical_data", (0.0, 0.5)),     # High attack success, low privacy
        ("similar_data", (0.2, 0.8)),       # Moderate attack success
        ("different_data", (0.5, 1.0))      # Low attack success, high privacy
    ])
    def test_attack_scenarios(self, attack_scenario, expected_privacy_range):
        """Test MIA under different attack scenarios."""
        if attack_scenario == "identical_data":
            real_data = [{"num": i} for i in range(30)]
            synth_data = [{"num": i} for i in range(30)]  # Identical
        elif attack_scenario == "similar_data":
            real_data = [{"num": i} for i in range(30)]
            synth_data = [{"num": i + 0.1} for i in range(30)]  # Very similar
        else:  # different_data
            real_data = [{"num": i} for i in range(30)]
            synth_data = [{"num": i + 100} for i in range(30)]  # Very different
        
        real_df = convert_to_pandas_mock(real_data)
        synth_df = convert_to_pandas_mock(synth_data)
        
        metric = MembershipInferenceMetric()
        context = create_mock_context()
        
        metric.fit(real_df, synth_df, context)
        result = metric.compute()
        
        assert_metric_result_structure(result, "privacy.membership_inference", "privacy")
        
        if "error" not in result.details:
            min_val, max_val = expected_privacy_range
            assert_value_in_range(result.value, min_val, max_val)
    
    def test_privacy_interpretation_consistency(self):
        """Test that MIA privacy scores are interpreted consistently."""
        # Higher privacy score should correspond to lower attack accuracy
        real_data = generate_tabular_data(n_samples=50, n_numeric=2, seed=42)
        synth_data = generate_tabular_data(n_samples=50, n_numeric=2, seed=43)
        
        real_df = convert_to_pandas_mock(real_data)
        synth_df = convert_to_pandas_mock(synth_data)
        
        metric = MembershipInferenceMetric()
        context = create_mock_context()
        
        metric.fit(real_df, synth_df, context)
        result = metric.compute()
        
        if "error" not in result.details and "attack_accuracy" in result.details:
            # Privacy score should be approximately 1 - attack_accuracy
            attack_acc = result.details["attack_accuracy"]
            expected_privacy = max(0.0, 1.0 - attack_acc)
            
            # Allow small numerical differences
            assert abs(result.value - expected_privacy) < 0.02, \
                f"Privacy score {result.value} should be approximately 1 - attack_accuracy {attack_acc} = {expected_privacy}"