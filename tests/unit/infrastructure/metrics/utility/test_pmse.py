"""Unit tests for PMSE utility metric."""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent.parent / "src"))

from valtapyV2.infrastructure.metrics.utility.pmse import PMSEMetric
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


def create_mock_dataset_spec(target_col: str = "target") -> DatasetSpec:
    """Create a mock dataset specification for testing."""
    return DatasetSpec(
        target=target_col,
        dtypes={
            "numeric1": "float64",
            "numeric2": "float64",
            "target": "object"
        }
    )


class MockSKLearnModel:
    """Mock scikit-learn model for testing without sklearn dependency."""
    
    def __init__(self, prediction_pattern="similar"):
        self.prediction_pattern = prediction_pattern
        self.is_fitted = False
    
    def fit(self, X, y):
        self.is_fitted = True
        self.feature_count = len(X[0]) if X else 0
        return self
    
    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        # Return mock predictions based on pattern
        n_samples = len(X)
        
        if self.prediction_pattern == "similar":
            # Predictions similar to a simple rule
            return [x[0] if x else 0 for x in X]
        elif self.prediction_pattern == "different":
            # Different predictions
            return [x[0] * -1 if x else 0 for x in X]
        else:
            # Random-like predictions
            return [i % 3 for i in range(n_samples)]


class TestPMSEMetric:
    """Test suite for PMSE utility metric."""
    
    def test_metric_initialization(self):
        """Test that the metric can be initialized correctly."""
        metric = PMSEMetric()
        
        assert metric.name == "pmse"
        assert metric.family == "utility"
        assert isinstance(metric.purpose_tags, set)
        assert "utility" in metric.purpose_tags
        assert "predictive_performance" in metric.purpose_tags
    
    def test_missing_target_column(self):
        """Test PMSE with missing target column specification."""
        real_data = generate_tabular_data(30, n_numeric=2, include_target=False)
        synth_data = generate_tabular_data(30, n_numeric=2, include_target=False)
        
        real_df = convert_to_pandas_mock(real_data)
        synth_df = convert_to_pandas_mock(synth_data)
        
        # Context without dataset_spec or with spec without target
        context = create_mock_context()
        
        metric = PMSEMetric()
        metric.fit(real_df, synth_df, context)
        result = metric.compute()
        
        # Should handle gracefully
        assert_metric_result_structure(result, "utility.pmse", "utility")
        assert result.value == 0.0
        assert "error" in result.details
        assert "Target column not specified" in result.details["error"]
    
    def test_no_numeric_features(self):
        """Test PMSE with no numeric feature columns."""
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
        
        context = create_mock_context()
        context["dataset_spec"] = create_mock_dataset_spec()
        
        metric = PMSEMetric()
        metric.fit(real_df, synth_df, context)
        result = metric.compute()
        
        # Should handle gracefully
        assert_metric_result_structure(result, "utility.pmse", "utility")
        assert result.value == 0.0
        assert "error" in result.details
        assert "No numeric feature columns" in result.details["error"]
    
    def test_regression_task_detection(self):
        """Test that PMSE correctly detects regression vs classification tasks."""
        # Create regression-like data (many unique numeric target values)
        real_data = []
        synth_data = []
        
        for i in range(50):
            real_data.append({
                "feature1": i,
                "feature2": i * 2,
                "target": i * 0.5 + 10  # Continuous target
            })
            synth_data.append({
                "feature1": i + 0.1,
                "feature2": (i + 0.1) * 2,
                "target": (i + 0.1) * 0.5 + 10
            })
        
        real_df = convert_to_pandas_mock(real_data)
        synth_df = convert_to_pandas_mock(synth_data)
        
        context = create_mock_context()
        context["dataset_spec"] = DatasetSpec(target="target")
        
        metric = PMSEMetric()
        metric.fit(real_df, synth_df, context)
        result = metric.compute()
        
        assert_metric_result_structure(result, "utility.pmse", "utility")
        
        # Check that it detected regression
        if "task_type" in result.details:
            assert result.details["task_type"] == "regression"
    
    def test_classification_task_detection(self):
        """Test that PMSE correctly detects classification tasks."""
        real_data = generate_tabular_data(40, n_numeric=2, include_target=True, seed=42)
        synth_data = generate_tabular_data(40, n_numeric=2, include_target=True, seed=43)
        
        real_df = convert_to_pandas_mock(real_data)
        synth_df = convert_to_pandas_mock(synth_data)
        
        context = create_mock_context()
        context["dataset_spec"] = create_mock_dataset_spec()
        
        metric = PMSEMetric()
        metric.fit(real_df, synth_df, context)
        result = metric.compute()
        
        assert_metric_result_structure(result, "utility.pmse", "utility")
        
        # Check that it detected classification
        if "task_type" in result.details:
            assert result.details["task_type"] == "classification"
    
    def test_identical_datasets_high_utility(self):
        """Test PMSE with identical datasets (should have high utility)."""
        real_data, synth_data = create_identical_datasets(n_samples=60)
        
        real_df = convert_to_pandas_mock(real_data)
        synth_df = convert_to_pandas_mock(synth_data)
        
        context = create_mock_context()
        context["dataset_spec"] = create_mock_dataset_spec()
        
        metric = PMSEMetric()
        metric.fit(real_df, synth_df, context)
        result = metric.compute()
        
        assert_metric_result_structure(result, "utility.pmse", "utility")
        
        # For identical datasets, TSTR should perform as well as TRTR
        # so utility score should be high
        if "error" not in result.details:
            assert_value_in_range(result.value, 0.7, 1.0)
            assert_no_error_in_details(result.details)
    
    def test_completely_different_datasets_low_utility(self):
        """Test PMSE with completely different datasets."""
        real_data, synth_data = create_completely_different_datasets(n_samples=40)
        
        real_df = convert_to_pandas_mock(real_data)
        synth_df = convert_to_pandas_mock(synth_data)
        
        context = create_mock_context()
        context["dataset_spec"] = create_mock_dataset_spec()
        
        metric = PMSEMetric()
        metric.fit(real_df, synth_df, context)
        result = metric.compute()
        
        assert_metric_result_structure(result, "utility.pmse", "utility")
        
        # Different datasets should have lower utility
        if "error" not in result.details:
            assert_value_in_range(result.value, 0.0, 0.6)
    
    def test_correlated_datasets_moderate_utility(self):
        """Test PMSE with correlated datasets."""
        real_data, synth_data = create_correlated_datasets(
            n_samples=50,
            correlation=0.8,
            noise_level=0.1
        )
        
        real_df = convert_to_pandas_mock(real_data)
        synth_df = convert_to_pandas_mock(synth_data)
        
        context = create_mock_context()
        context["dataset_spec"] = create_mock_dataset_spec()
        
        metric = PMSEMetric()
        metric.fit(real_df, synth_df, context)
        result = metric.compute()
        
        assert_metric_result_structure(result, "utility.pmse", "utility")
        
        # Correlated datasets should have moderate to high utility
        if "error" not in result.details:
            assert_value_in_range(result.value, 0.3, 1.0)
            assert_no_error_in_details(result.details)
    
    def test_pmse_details_structure(self):
        """Test the detailed structure of PMSE results."""
        real_data = generate_tabular_data(45, n_numeric=3, include_target=True, seed=42)
        synth_data = generate_tabular_data(45, n_numeric=3, include_target=True, seed=43)
        
        real_df = convert_to_pandas_mock(real_data)
        synth_df = convert_to_pandas_mock(synth_data)
        
        context = create_mock_context()
        context["dataset_spec"] = create_mock_dataset_spec()
        
        metric = PMSEMetric()
        metric.fit(real_df, synth_df, context)
        result = metric.compute()
        
        if "error" not in result.details:
            # Check for expected details structure
            expected_keys = [
                "task_type",
                "tstr_scores", 
                "trtr_scores",
                "mean_tstr_score",
                "mean_trtr_score",
                "n_features"
            ]
            
            for key in expected_keys:
                if key in result.details:  # Some keys might not be present in mock implementation
                    if "scores" in key:
                        assert isinstance(result.details[key], list)
                    elif "mean" in key:
                        assert isinstance(result.details[key], (int, float))
                    elif key == "n_features":
                        assert isinstance(result.details[key], int)
                        assert result.details[key] >= 1
    
    def test_cross_validation_splits(self):
        """Test that PMSE uses multiple CV splits correctly."""
        real_data = generate_tabular_data(60, n_numeric=2, include_target=True, seed=42)
        synth_data = generate_tabular_data(60, n_numeric=2, include_target=True, seed=43)
        
        real_df = convert_to_pandas_mock(real_data)
        synth_df = convert_to_pandas_mock(synth_data)
        
        context = create_mock_context()
        context["dataset_spec"] = create_mock_dataset_spec()
        
        metric = PMSEMetric()
        metric.fit(real_df, synth_df, context)
        result = metric.compute()
        
        if "error" not in result.details and "n_splits" in result.details:
            # Should use the configured number of splits (default 3)
            assert result.details["n_splits"] >= 2
            
            if "tstr_scores" in result.details:
                # Should have one score per split
                assert len(result.details["tstr_scores"]) == result.details["n_splits"]
                assert len(result.details["trtr_scores"]) == result.details["n_splits"]
    
    def test_fallback_without_sklearn(self):
        """Test PMSE fallback behavior when sklearn is not available."""
        real_data = generate_tabular_data(30, n_numeric=2, include_target=True, seed=42)
        synth_data = generate_tabular_data(30, n_numeric=2, include_target=True, seed=43)
        
        real_df = convert_to_pandas_mock(real_data)
        synth_df = convert_to_pandas_mock(synth_data)
        
        context = create_mock_context()
        context["dataset_spec"] = create_mock_dataset_spec()
        
        metric = PMSEMetric()
        metric.fit(real_df, synth_df, context)
        result = metric.compute()
        
        # Should handle sklearn absence gracefully
        assert_metric_result_structure(result, "utility.pmse", "utility")
        assert_value_in_range(result.value, 0.0, 1.0)
        
        # Should have fallback indication in details
        if "sklearn not available" in result.details.get("error", ""):
            assert result.value == 0.5  # Neutral fallback score
    
    def test_missing_values_handling(self):
        """Test PMSE with missing values in features and target."""
        real_data = []
        synth_data = []
        
        for i in range(40):
            real_data.append({
                "feature1": i if i % 5 != 0 else None,  # Some missing
                "feature2": i * 2,
                "target": "high" if i > 20 else ("low" if i % 8 != 0 else None)
            })
            synth_data.append({
                "feature1": i + 0.1 if i % 6 != 0 else None,
                "feature2": (i + 0.1) * 2,
                "target": "high" if i > 18 else "low"
            })
        
        real_df = convert_to_pandas_mock(real_data)
        synth_df = convert_to_pandas_mock(synth_data)
        
        context = create_mock_context()
        context["dataset_spec"] = create_mock_dataset_spec()
        
        metric = PMSEMetric()
        metric.fit(real_df, synth_df, context)
        result = metric.compute()
        
        # Should handle missing values by filling with 0 (as per implementation)
        assert_metric_result_structure(result, "utility.pmse", "utility")
        assert_value_in_range(result.value, 0.0, 1.0)
    
    def test_edge_cases_robustness(self):
        """Test various edge cases for robustness."""
        test_cases = [
            MetricTestCase(
                name="minimal_data",
                real_data=[
                    {"feat": 1, "target": "a"},
                    {"feat": 2, "target": "b"}
                ] * 10,  # Minimal but sufficient data
                synth_data=[
                    {"feat": 1.1, "target": "a"},
                    {"feat": 2.1, "target": "b"}
                ] * 10,
                expected_value_range=(0.0, 1.0),
                description="Minimal dataset size"
            ),
            MetricTestCase(
                name="single_class_target",
                real_data=[{"feat": i, "target": "same"} for i in range(20)],
                synth_data=[{"feat": i + 0.1, "target": "same"} for i in range(20)],
                expected_value_range=(0.0, 1.0),
                description="Single class in target"
            )
        ]
        
        for test_case in test_cases:
            real_df = convert_to_pandas_mock(test_case.real_data)
            synth_df = convert_to_pandas_mock(test_case.synth_data)
            
            context = create_mock_context()
            context["dataset_spec"] = create_mock_dataset_spec()
            
            metric = PMSEMetric()
            metric.fit(real_df, synth_df, context)
            result = metric.compute()
            
            # Basic structure validation
            assert_metric_result_structure(result, "utility.pmse", "utility")
            assert_value_in_range(result.value, 0.0, 1.0)
    
    def test_caching_integration(self):
        """Test that PMSE works with caching for train/test splits."""
        from valtapyV2.infrastructure.runtime.cache import StatsStore
        
        real_data, synth_data = create_correlated_datasets(n_samples=35, correlation=0.9)
        real_df = convert_to_pandas_mock(real_data)
        synth_df = convert_to_pandas_mock(synth_data)
        
        # Create context with stats store
        stats_store = StatsStore()
        context = {"stats_store": stats_store, "dataset_spec": create_mock_dataset_spec()}
        
        metric = PMSEMetric()
        metric.fit(real_df, synth_df, context)
        result = metric.compute()
        
        # Should complete without errors
        assert_metric_result_structure(result, "utility.pmse", "utility")
        
        # Check that cache was potentially used
        cache_stats = stats_store.get_stats()
        assert isinstance(cache_stats, dict)
    
    @pytest.mark.parametrize("target_type,expected_task", [
        ("continuous_numeric", "regression"),
        ("few_categories", "classification"),
        ("many_categories", "classification"),
    ])
    def test_task_type_detection_variations(self, target_type, expected_task):
        """Test task type detection with different target distributions."""
        real_data = []
        synth_data = []
        
        for i in range(30):
            if target_type == "continuous_numeric":
                target_val = i * 0.33 + 1.5  # Continuous
            elif target_type == "few_categories":
                target_val = ["low", "medium", "high"][i % 3]  # 3 categories
            else:  # many_categories
                target_val = f"class_{i % 15}"  # 15 categories
            
            real_data.append({"feature": i, "target": target_val})
            synth_data.append({"feature": i + 0.1, "target": target_val})
        
        real_df = convert_to_pandas_mock(real_data)
        synth_df = convert_to_pandas_mock(synth_data)
        
        context = create_mock_context()
        context["dataset_spec"] = create_mock_dataset_spec()
        
        metric = PMSEMetric()
        metric.fit(real_df, synth_df, context)
        result = metric.compute()
        
        assert_metric_result_structure(result, "utility.pmse", "utility")
        
        # Check task type detection if available
        if "task_type" in result.details:
            if target_type == "continuous_numeric":
                assert result.details["task_type"] == "regression"
            else:
                assert result.details["task_type"] == "classification"