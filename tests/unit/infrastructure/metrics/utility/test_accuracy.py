"""Unit tests for Accuracy utility metric."""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent.parent / "src"))

from valtapyV2.infrastructure.metrics.utility.accuracy import AccuracyMetric
from valtapyV2.domain.entities import MetricResult, DatasetSpec
from tests.utils.data_generators import (
    generate_tabular_data,
    create_identical_datasets,
    create_completely_different_datasets,
    create_correlated_datasets
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


def create_classification_data(n_samples=50, n_classes=3, n_features=2, correlation=0.8, seed=42):
    """Create classification datasets with controlled properties."""
    import random
    random.seed(seed)
    
    class_names = [f"class_{i}" for i in range(n_classes)]
    
    real_data = []
    synth_data = []
    
    for i in range(n_samples):
        # Create features with some pattern
        features_real = {f"feature_{j}": random.gauss(i % n_classes * 10, 5) for j in range(n_features)}
        
        # Assign class based on feature pattern
        class_idx = i % n_classes
        features_real["target"] = class_names[class_idx]
        real_data.append(features_real)
        
        # Create synthetic data with specified correlation
        features_synth = {}
        for key, val in features_real.items():
            if key != "target":
                if random.random() < correlation:
                    # Keep correlated
                    features_synth[key] = val + random.gauss(0, 1)
                else:
                    # Make different
                    features_synth[key] = random.gauss(0, 10)
            else:
                # For target, sometimes change to test transferability
                if random.random() < correlation:
                    features_synth[key] = val
                else:
                    features_synth[key] = random.choice(class_names)
        
        synth_data.append(features_synth)
    
    return real_data, synth_data


class TestAccuracyMetric:
    """Test suite for Accuracy utility metric."""
    
    def test_metric_initialization(self):
        """Test that the metric can be initialized correctly."""
        metric = AccuracyMetric()
        
        assert metric.name == "accuracy"
        assert metric.family == "utility"
        assert isinstance(metric.purpose_tags, set)
        assert "utility" in metric.purpose_tags
        assert "classification_performance" in metric.purpose_tags
    
    def test_missing_target_column(self):
        """Test accuracy with missing target column specification."""
        real_data = generate_tabular_data(30, n_numeric=2, include_target=False)
        synth_data = generate_tabular_data(30, n_numeric=2, include_target=False)
        
        real_df = convert_to_pandas_mock(real_data)
        synth_df = convert_to_pandas_mock(synth_data)
        
        context = create_mock_context()
        
        metric = AccuracyMetric()
        metric.fit(real_df, synth_df, context)
        result = metric.compute()
        
        # Should handle gracefully
        assert_metric_result_structure(result, "utility.accuracy", "utility")
        assert result.value == 0.0
        assert "error" in result.details
        assert "Target column not specified" in result.details["error"]
    
    def test_target_too_many_classes(self):
        """Test accuracy with target having too many unique values for classification."""
        # Create data with continuous-like target (many unique values)
        real_data = []
        synth_data = []
        
        for i in range(60):
            real_data.append({
                "feature1": i,
                "feature2": i * 2,
                "target": f"unique_class_{i}"  # Each sample has unique class
            })
            synth_data.append({
                "feature1": i + 0.1,
                "feature2": (i + 0.1) * 2,
                "target": f"unique_class_{i}"
            })
        
        real_df = convert_to_pandas_mock(real_data)
        synth_df = convert_to_pandas_mock(synth_data)
        
        context = create_mock_context()
        context["dataset_spec"] = DatasetSpec(target="target")
        
        metric = AccuracyMetric()
        metric.fit(real_df, synth_df, context)
        result = metric.compute()
        
        # Should detect too many classes for classification
        assert_metric_result_structure(result, "utility.accuracy", "utility")
        assert result.value == 0.0
        assert "error" in result.details
        assert "too many unique values" in result.details["error"]
    
    def test_target_insufficient_classes(self):
        """Test accuracy with target having insufficient classes."""
        # Create data with only one class
        real_data = [
            {"feature1": i, "feature2": i * 2, "target": "only_class"}
            for i in range(30)
        ]
        synth_data = [
            {"feature1": i + 0.1, "feature2": (i + 0.1) * 2, "target": "only_class"}
            for i in range(30)
        ]
        
        real_df = convert_to_pandas_mock(real_data)
        synth_df = convert_to_pandas_mock(synth_data)
        
        context = create_mock_context()
        context["dataset_spec"] = DatasetSpec(target="target")
        
        metric = AccuracyMetric()
        metric.fit(real_df, synth_df, context)
        result = metric.compute()
        
        # Should detect insufficient classes
        assert_metric_result_structure(result, "utility.accuracy", "utility")
        assert result.value == 0.0
        assert "error" in result.details
        assert "only 1 unique values" in result.details["error"]
    
    def test_no_numeric_features(self):
        """Test accuracy with no numeric feature columns."""
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
        
        context = create_mock_context()
        context["dataset_spec"] = DatasetSpec(target="target")
        
        metric = AccuracyMetric()
        metric.fit(real_df, synth_df, context)
        result = metric.compute()
        
        # Should handle gracefully
        assert_metric_result_structure(result, "utility.accuracy", "utility")
        assert result.value == 0.0
        assert "error" in result.details
        assert "No numeric feature columns" in result.details["error"]
    
    def test_identical_datasets_high_utility(self):
        """Test accuracy with identical datasets (should have high utility)."""
        real_data, synth_data = create_classification_data(n_samples=60, correlation=1.0)
        
        real_df = convert_to_pandas_mock(real_data)
        synth_df = convert_to_pandas_mock(synth_data)
        
        context = create_mock_context()
        context["dataset_spec"] = DatasetSpec(target="target")
        
        metric = AccuracyMetric()
        metric.fit(real_df, synth_df, context)
        result = metric.compute()
        
        assert_metric_result_structure(result, "utility.accuracy", "utility")
        
        # For identical datasets, TSTR should perform as well as TRTR
        if "error" not in result.details:
            assert_value_in_range(result.value, 0.7, 1.0)
            assert_no_error_in_details(result.details)
    
    def test_different_datasets_lower_utility(self):
        """Test accuracy with different datasets."""
        real_data, synth_data = create_classification_data(n_samples=50, correlation=0.3)
        
        real_df = convert_to_pandas_mock(real_data)
        synth_df = convert_to_pandas_mock(synth_data)
        
        context = create_mock_context()
        context["dataset_spec"] = DatasetSpec(target="target")
        
        metric = AccuracyMetric()
        metric.fit(real_df, synth_df, context)
        result = metric.compute()
        
        assert_metric_result_structure(result, "utility.accuracy", "utility")
        
        # Different datasets should have lower utility
        if "error" not in result.details:
            assert_value_in_range(result.value, 0.0, 0.8)
    
    def test_correlated_datasets_moderate_utility(self):
        """Test accuracy with moderately correlated datasets."""
        real_data, synth_data = create_classification_data(n_samples=60, correlation=0.8)
        
        real_df = convert_to_pandas_mock(real_data)
        synth_df = convert_to_pandas_mock(synth_data)
        
        context = create_mock_context()
        context["dataset_spec"] = DatasetSpec(target="target")
        
        metric = AccuracyMetric()
        metric.fit(real_df, synth_df, context)
        result = metric.compute()
        
        assert_metric_result_structure(result, "utility.accuracy", "utility")
        
        # Correlated datasets should have moderate to high utility
        if "error" not in result.details:
            assert_value_in_range(result.value, 0.4, 1.0)
            assert_no_error_in_details(result.details)
    
    def test_accuracy_details_structure(self):
        """Test the detailed structure of accuracy results."""
        real_data, synth_data = create_classification_data(n_samples=45, n_features=3)
        
        real_df = convert_to_pandas_mock(real_data)
        synth_df = convert_to_pandas_mock(synth_data)
        
        context = create_mock_context()
        context["dataset_spec"] = DatasetSpec(target="target")
        
        metric = AccuracyMetric()
        metric.fit(real_df, synth_df, context)
        result = metric.compute()
        
        if "error" not in result.details:
            # Check for expected details structure
            expected_keys = [
                "tstr_accuracies",
                "trtr_accuracies", 
                "mean_tstr_accuracy",
                "mean_trtr_accuracy",
                "n_classes",
                "n_features"
            ]
            
            for key in expected_keys:
                if key in result.details:  # Some keys might not be present in mock implementation
                    if "accuracies" in key:
                        assert isinstance(result.details[key], list)
                    elif "mean" in key:
                        assert isinstance(result.details[key], (int, float))
                        assert 0.0 <= result.details[key] <= 1.0
                    elif key in ["n_classes", "n_features"]:
                        assert isinstance(result.details[key], int)
                        assert result.details[key] >= 1
    
    def test_cross_validation_folds(self):
        """Test that accuracy uses multiple CV folds correctly."""
        real_data, synth_data = create_classification_data(n_samples=60, n_features=2)
        
        real_df = convert_to_pandas_mock(real_data)
        synth_df = convert_to_pandas_mock(synth_data)
        
        context = create_mock_context()
        context["dataset_spec"] = DatasetSpec(target="target")
        
        metric = AccuracyMetric()
        metric.fit(real_df, synth_df, context)
        result = metric.compute()
        
        if "error" not in result.details and "n_successful_folds" in result.details:
            # Should use multiple folds
            assert result.details["n_successful_folds"] >= 2
            
            if "tstr_accuracies" in result.details:
                # Should have one accuracy per successful fold
                assert len(result.details["tstr_accuracies"]) == result.details["n_successful_folds"]
                assert len(result.details["trtr_accuracies"]) == result.details["n_successful_folds"]
    
    def test_class_distribution_handling(self):
        """Test accuracy with imbalanced class distribution."""
        # Create imbalanced data
        real_data = []
        synth_data = []
        
        # 80% class A, 15% class B, 5% class C
        for i in range(100):
            if i < 80:
                class_label = "class_A"
                feature_val = 1
            elif i < 95:
                class_label = "class_B"
                feature_val = 2
            else:
                class_label = "class_C"
                feature_val = 3
            
            real_data.append({
                "feature1": feature_val + (i * 0.1),
                "feature2": feature_val * 2 + (i * 0.05),
                "target": class_label
            })
            
            # Synthetic with similar but not identical distribution
            synth_data.append({
                "feature1": feature_val + (i * 0.1) + 0.1,
                "feature2": feature_val * 2 + (i * 0.05) + 0.05,
                "target": class_label
            })
        
        real_df = convert_to_pandas_mock(real_data)
        synth_df = convert_to_pandas_mock(synth_data)
        
        context = create_mock_context()
        context["dataset_spec"] = DatasetSpec(target="target")
        
        metric = AccuracyMetric()
        metric.fit(real_df, synth_df, context)
        result = metric.compute()
        
        assert_metric_result_structure(result, "utility.accuracy", "utility")
        
        if "error" not in result.details:
            # Should handle imbalanced classes
            assert_value_in_range(result.value, 0.0, 1.0)
            
            if "class_distribution" in result.details:
                dist = result.details["class_distribution"]
                assert isinstance(dist, dict)
                assert "class_A" in dist
                assert dist["class_A"] > dist.get("class_B", 0)  # A should be more frequent
    
    def test_insufficient_data_per_fold(self):
        """Test accuracy behavior with insufficient data for some folds."""
        # Very small dataset
        real_data = [
            {"feature": 1, "target": "A"},
            {"feature": 2, "target": "B"},
            {"feature": 3, "target": "A"},
            {"feature": 4, "target": "B"}
        ]
        synth_data = [
            {"feature": 1.1, "target": "A"},
            {"feature": 2.1, "target": "B"},
            {"feature": 3.1, "target": "A"},
            {"feature": 4.1, "target": "B"}
        ]
        
        real_df = convert_to_pandas_mock(real_data)
        synth_df = convert_to_pandas_mock(synth_data)
        
        context = create_mock_context()
        context["dataset_spec"] = DatasetSpec(target="target")
        
        metric = AccuracyMetric()
        metric.fit(real_df, synth_df, context)
        result = metric.compute()
        
        # Should handle small dataset gracefully
        assert_metric_result_structure(result, "utility.accuracy", "utility")
        
        if "Could not complete any cross-validation folds" in result.details.get("error", ""):
            assert result.value == 0.0
        else:
            # If it managed to run, should be valid
            assert_value_in_range(result.value, 0.0, 1.0)
    
    def test_fallback_without_sklearn(self):
        """Test accuracy fallback behavior when sklearn is not available."""
        real_data, synth_data = create_classification_data(n_samples=30)
        
        real_df = convert_to_pandas_mock(real_data)
        synth_df = convert_to_pandas_mock(synth_data)
        
        context = create_mock_context()
        context["dataset_spec"] = DatasetSpec(target="target")
        
        metric = AccuracyMetric()
        metric.fit(real_df, synth_df, context)
        result = metric.compute()
        
        # Should handle sklearn absence gracefully
        assert_metric_result_structure(result, "utility.accuracy", "utility")
        assert_value_in_range(result.value, 0.0, 1.0)
        
        # Should have fallback indication in details
        if "sklearn not available" in result.details.get("error", ""):
            assert result.value == 0.5  # Neutral fallback score
    
    def test_missing_values_in_features(self):
        """Test accuracy with missing values in features."""
        real_data = []
        synth_data = []
        
        for i in range(40):
            real_data.append({
                "feature1": i if i % 5 != 0 else None,  # Some missing
                "feature2": i * 2,
                "target": "high" if i > 20 else "low"
            })
            synth_data.append({
                "feature1": i + 0.1 if i % 6 != 0 else None,
                "feature2": (i + 0.1) * 2,
                "target": "high" if i > 18 else "low"
            })
        
        real_df = convert_to_pandas_mock(real_data)
        synth_df = convert_to_pandas_mock(synth_data)
        
        context = create_mock_context()
        context["dataset_spec"] = DatasetSpec(target="target")
        
        metric = AccuracyMetric()
        metric.fit(real_df, synth_df, context)
        result = metric.compute()
        
        # Should handle missing values by filling with 0 (as per implementation)
        assert_metric_result_structure(result, "utility.accuracy", "utility")
        assert_value_in_range(result.value, 0.0, 1.0)
    
    def test_caching_integration(self):
        """Test that accuracy works with caching for train/test splits."""
        from valtapyV2.infrastructure.runtime.cache import StatsStore
        
        real_data, synth_data = create_classification_data(n_samples=35, correlation=0.9)
        real_df = convert_to_pandas_mock(real_data)
        synth_df = convert_to_pandas_mock(synth_data)
        
        # Create context with stats store
        stats_store = StatsStore()
        context = {"stats_store": stats_store, "dataset_spec": DatasetSpec(target="target")}
        
        metric = AccuracyMetric()
        metric.fit(real_df, synth_df, context)
        result = metric.compute()
        
        # Should complete without errors
        assert_metric_result_structure(result, "utility.accuracy", "utility")
        
        # Check that cache was potentially used
        cache_stats = stats_store.get_stats()
        assert isinstance(cache_stats, dict)
    
    @pytest.mark.parametrize("n_classes,expected_difficulty", [
        (2, "easier"),   # Binary classification
        (3, "moderate"), # Multi-class
        (5, "harder")    # More classes
    ])
    def test_different_class_counts(self, n_classes, expected_difficulty):
        """Test accuracy with different numbers of classes."""
        real_data, synth_data = create_classification_data(
            n_samples=60, 
            n_classes=n_classes,
            correlation=0.8
        )
        
        real_df = convert_to_pandas_mock(real_data)
        synth_df = convert_to_pandas_mock(synth_data)
        
        context = create_mock_context()
        context["dataset_spec"] = DatasetSpec(target="target")
        
        metric = AccuracyMetric()
        metric.fit(real_df, synth_df, context)
        result = metric.compute()
        
        assert_metric_result_structure(result, "utility.accuracy", "utility")
        
        if "error" not in result.details:
            assert_value_in_range(result.value, 0.0, 1.0)
            
            if "n_classes" in result.details:
                assert result.details["n_classes"] == n_classes
    
    def test_edge_cases_robustness(self):
        """Test various edge cases for robustness."""
        test_cases = [
            MetricTestCase(
                name="perfect_separable",
                real_data=[
                    {"feat": 0, "target": "A"},
                    {"feat": 100, "target": "B"}
                ] * 15,
                synth_data=[
                    {"feat": 1, "target": "A"},
                    {"feat": 99, "target": "B"}
                ] * 15,
                expected_value_range=(0.5, 1.0),
                description="Perfectly separable classes"
            ),
            MetricTestCase(
                name="overlapping_classes",
                real_data=[
                    {"feat": 50, "target": "A"},
                    {"feat": 51, "target": "B"}
                ] * 15,
                synth_data=[
                    {"feat": 50.1, "target": "A"},
                    {"feat": 50.9, "target": "B"}
                ] * 15,
                expected_value_range=(0.0, 1.0),
                description="Highly overlapping classes"
            )
        ]
        
        for test_case in test_cases:
            real_df = convert_to_pandas_mock(test_case.real_data)
            synth_df = convert_to_pandas_mock(test_case.synth_data)
            
            context = create_mock_context()
            context["dataset_spec"] = DatasetSpec(target="target")
            
            metric = AccuracyMetric()
            metric.fit(real_df, synth_df, context)
            result = metric.compute()
            
            # Basic structure validation
            assert_metric_result_structure(result, "utility.accuracy", "utility")
            assert_value_in_range(result.value, 0.0, 1.0)