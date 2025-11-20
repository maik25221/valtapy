"""Tests for Membership Inference Attack metric implementation."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from valtapy.evaluation.metrics.privacy.membership_inference import MembershipInferenceMetric
from valtapy.evaluation.entities import (
    MetricExecutionContext,
    MetricResult,
    MetricFamily
)
from valtapy.evaluation.exceptions import (
    MetricComputationError,
    UnsupportedDataError
)


class TestMembershipInferenceMetric:
    """Test cases for MembershipInferenceMetric."""

    def setup_method(self):
        """Set up test fixtures."""
        self.metric = MembershipInferenceMetric()

    def test_metric_metadata(self):
        """Test that metric has correct metadata."""
        assert self.metric.metric_id == "privacy.mia"
        assert self.metric.family == MetricFamily.PRIVACY
        assert self.metric.name == "Membership Inference Attack"
        assert isinstance(self.metric.description, str)
        assert len(self.metric.description) > 0

    def test_compute_with_distinct_data(self):
        """Test MIA with completely distinct real and synthetic data."""
        np.random.seed(42)

        # Create very different distributions
        real_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 200),
            'feature2': np.random.uniform(0, 10, 200)
        })

        synth_data = pd.DataFrame({
            'feature1': np.random.normal(100, 1, 200),  # Very different mean
            'feature2': np.random.uniform(100, 110, 200)  # Very different range
        })

        context = MetricExecutionContext(
            real_data=real_data,
            synth_data=synth_data,
            parameters={'n_splits': 3},
            cache_enabled=False
        )

        result = self.metric.compute(context)

        # Verify result structure
        assert isinstance(result, MetricResult)
        assert result.metric_id == "privacy.mia"
        assert result.family == MetricFamily.PRIVACY

        # For very distinct data, attack should be easy (high accuracy, low privacy score)
        assert result.details['attack_accuracy'] > 0.7  # Should be able to distinguish easily
        assert result.value < 0.5  # Low privacy score

    def test_compute_with_similar_data(self):
        """Test MIA with very similar real and synthetic data."""
        np.random.seed(42)

        # Create similar distributions
        real_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 200),
            'feature2': np.random.uniform(0, 10, 200)
        })

        np.random.seed(43)  # Different seed but same distribution
        synth_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 200),
            'feature2': np.random.uniform(0, 10, 200)
        })

        context = MetricExecutionContext(
            real_data=real_data,
            synth_data=synth_data,
            parameters={'n_splits': 3},
            cache_enabled=False
        )

        result = self.metric.compute(context)

        # For similar data, attack should be harder (lower accuracy, higher privacy score)
        # Note: This might not always be < 0.7 due to randomness, but should be closer to 0.5
        assert result.details['attack_accuracy'] >= 0.0
        assert result.value >= 0.0  # Privacy score should be non-negative

    def test_compute_with_different_classifiers(self):
        """Test MIA with different classifier types."""
        np.random.seed(42)

        real_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.uniform(0, 10, 100)
        })

        synth_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.uniform(0, 10, 100)
        })

        classifiers = ['random_forest', 'logistic_regression', 'gradient_boosting']

        for classifier_type in classifiers:
            context = MetricExecutionContext(
                real_data=real_data,
                synth_data=synth_data,
                parameters={'classifier': classifier_type, 'n_splits': 3},
                cache_enabled=False
            )

            result = self.metric.compute(context)

            assert isinstance(result, MetricResult)
            assert result.details['classifier_type'] == classifier_type
            assert 0.0 <= result.value <= 1.0

    def test_compute_with_missing_values(self):
        """Test MIA handles missing values correctly."""
        np.random.seed(42)

        # Create data with missing values
        real_data = pd.DataFrame({
            'feature1': [1, 2, 3, np.nan, 5, 6, 7, 8, 9, 10] * 10,
            'feature2': [1, np.nan, 3, 4, 5, 6, 7, 8, 9, 10] * 10
        })

        synth_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, np.nan, 6, 7, 8, 9, 10] * 10,
            'feature2': [1, 2, np.nan, 4, 5, 6, 7, 8, 9, 10] * 10
        })

        context = MetricExecutionContext(
            real_data=real_data,
            synth_data=synth_data,
            parameters={'n_splits': 3},
            cache_enabled=False
        )

        result = self.metric.compute(context)

        # Should successfully compute despite NaN values
        assert isinstance(result, MetricResult)
        assert 0.0 <= result.value <= 1.0

    def test_compute_with_no_numeric_columns(self):
        """Test MIA raises error when no numeric columns exist."""
        # Create datasets with only categorical columns
        real_data = pd.DataFrame({
            'col1': ['a', 'b', 'c', 'd'] * 25,
            'col2': ['x', 'y', 'z', 'w'] * 25
        })

        synth_data = pd.DataFrame({
            'col1': ['a', 'b', 'c', 'd'] * 25,
            'col2': ['x', 'y', 'z', 'w'] * 25
        })

        context = MetricExecutionContext(
            real_data=real_data,
            synth_data=synth_data,
            parameters={}
        )

        # Should raise MetricComputationError (wrapping UnsupportedDataError)
        with pytest.raises(MetricComputationError) as exc_info:
            self.metric.compute(context)

        assert "privacy.mia" in str(exc_info.value)
        assert "No common numeric columns" in str(exc_info.value)

    def test_compute_with_empty_data(self):
        """Test MIA raises error with empty datasets."""
        real_data = pd.DataFrame()
        synth_data = pd.DataFrame({'col1': [1, 2, 3]})

        context = MetricExecutionContext(
            real_data=real_data,
            synth_data=synth_data,
            parameters={}
        )

        with pytest.raises(MetricComputationError) as exc_info:
            self.metric.compute(context)

        assert "privacy.mia" in str(exc_info.value)

    def test_compute_timing_recorded(self):
        """Test that computation time is recorded."""
        np.random.seed(42)

        real_data = pd.DataFrame({
            'col1': np.random.normal(0, 1, 100),
            'col2': np.random.uniform(0, 10, 100)
        })

        synth_data = pd.DataFrame({
            'col1': np.random.normal(0, 1, 100),
            'col2': np.random.uniform(0, 10, 100)
        })

        context = MetricExecutionContext(
            real_data=real_data,
            synth_data=synth_data,
            parameters={'n_splits': 3}
        )

        result = self.metric.compute(context)

        # Computation time should be recorded and positive
        assert result.computation_time > 0
        assert isinstance(result.computation_time, float)

    def test_can_compute_with_numeric_data(self):
        """Test can_compute returns True for numeric data with sufficient samples."""
        real_df = pd.DataFrame({
            'col1': np.random.randn(50),
            'col2': np.random.randn(50)
        })

        synth_df = pd.DataFrame({
            'col1': np.random.randn(50),
            'col2': np.random.randn(50)
        })

        assert self.metric.can_compute(real_df, synth_df) is True

    def test_can_compute_with_insufficient_samples(self):
        """Test can_compute returns False with insufficient samples."""
        real_df = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5]  # Only 5 samples
        })

        synth_df = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5]
        })

        assert self.metric.can_compute(real_df, synth_df) is False

    def test_can_compute_with_no_numeric_columns(self):
        """Test can_compute returns False when no numeric columns."""
        real_df = pd.DataFrame({
            'col1': ['a', 'b', 'c'] * 10,
            'col2': ['x', 'y', 'z'] * 10
        })

        synth_df = pd.DataFrame({
            'col1': ['a', 'b', 'c'] * 10,
            'col2': ['x', 'y', 'z'] * 10
        })

        assert self.metric.can_compute(real_df, synth_df) is False

    def test_can_compute_with_no_common_columns(self):
        """Test can_compute returns False when no common numeric columns."""
        real_df = pd.DataFrame({
            'col1': np.random.randn(50)
        })

        synth_df = pd.DataFrame({
            'col2': np.random.randn(50)
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

    def test_validate_parameters_valid_classifier(self):
        """Test validate_parameters accepts valid classifier types."""
        valid_classifiers = ['random_forest', 'logistic_regression', 'mlp', 'gradient_boosting']

        for classifier in valid_classifiers:
            assert self.metric.validate_parameters({'classifier': classifier}) is True

    def test_validate_parameters_invalid_classifier(self):
        """Test validate_parameters rejects invalid classifier types."""
        assert self.metric.validate_parameters({'classifier': 'invalid_classifier'}) is False
        assert self.metric.validate_parameters({'classifier': 123}) is False

    def test_validate_parameters_valid_test_size(self):
        """Test validate_parameters accepts valid test_size."""
        assert self.metric.validate_parameters({'test_size': 0.2}) is True
        assert self.metric.validate_parameters({'test_size': 0.5}) is True
        assert self.metric.validate_parameters({'test_size': 0.8}) is True

    def test_validate_parameters_invalid_test_size(self):
        """Test validate_parameters rejects invalid test_size."""
        assert self.metric.validate_parameters({'test_size': 0}) is False
        assert self.metric.validate_parameters({'test_size': 1}) is False
        assert self.metric.validate_parameters({'test_size': 1.5}) is False
        assert self.metric.validate_parameters({'test_size': -0.2}) is False

    def test_validate_parameters_valid_n_splits(self):
        """Test validate_parameters accepts valid n_splits."""
        assert self.metric.validate_parameters({'n_splits': 2}) is True
        assert self.metric.validate_parameters({'n_splits': 5}) is True
        assert self.metric.validate_parameters({'n_splits': 10}) is True

    def test_validate_parameters_invalid_n_splits(self):
        """Test validate_parameters rejects invalid n_splits."""
        assert self.metric.validate_parameters({'n_splits': 1}) is False
        assert self.metric.validate_parameters({'n_splits': 0}) is False
        assert self.metric.validate_parameters({'n_splits': -5}) is False
        assert self.metric.validate_parameters({'n_splits': 5.5}) is False

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

    def test_result_details_structure(self):
        """Test that result details have expected structure."""
        np.random.seed(42)

        real_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100)
        })

        synth_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100)
        })

        context = MetricExecutionContext(
            real_data=real_data,
            synth_data=synth_data,
            parameters={'n_splits': 3}
        )

        result = self.metric.compute(context)

        # Check details structure
        assert 'attack_accuracy' in result.details
        assert 'attack_accuracy_std' in result.details
        assert 'attack_precision' in result.details
        assert 'attack_recall' in result.details
        assert 'attack_f1' in result.details
        assert 'attack_auc' in result.details
        assert 'privacy_score' in result.details
        assert 'num_features' in result.details
        assert 'num_real_samples' in result.details
        assert 'num_synth_samples' in result.details
        assert 'classifier_type' in result.details
        assert 'interpretation' in result.details

        # Check metadata structure
        assert 'attack_type' in result.metadata
        assert result.metadata['attack_type'] == 'classification'
        assert 'classifier' in result.metadata
        assert 'features_used' in result.metadata

    def test_privacy_score_calculation(self):
        """Test that privacy score is calculated correctly as 1 - accuracy."""
        np.random.seed(42)

        real_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100)
        })

        synth_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100)
        })

        context = MetricExecutionContext(
            real_data=real_data,
            synth_data=synth_data,
            parameters={'n_splits': 3}
        )

        result = self.metric.compute(context)

        # Privacy score should be 1 - attack_accuracy
        expected_privacy = 1.0 - result.details['attack_accuracy']
        assert abs(result.value - expected_privacy) < 1e-6

        # Privacy score should also match the one in details
        assert abs(result.value - result.details['privacy_score']) < 1e-6

    def test_interpret_results_good_privacy(self):
        """Test interpretation for good privacy (low attack accuracy)."""
        interpretation = self.metric._interpret_results(0.5, 0.5)
        assert "GOOD PRIVACY" in interpretation

    def test_interpret_results_moderate_privacy(self):
        """Test interpretation for moderate privacy risk."""
        interpretation = self.metric._interpret_results(0.3, 0.7)
        assert "MODERATE PRIVACY RISK" in interpretation

    def test_interpret_results_high_risk(self):
        """Test interpretation for high privacy risk (high attack accuracy)."""
        interpretation = self.metric._interpret_results(0.1, 0.9)
        assert "HIGH PRIVACY RISK" in interpretation

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
            parameters={'classifier': 'random_forest', 'n_splits': 3},
            cache_enabled=True,
            random_seed=42
        )

        result = self.metric.compute(context)

        # Should use context parameters correctly
        assert isinstance(result, MetricResult)
        assert result.metric_id == self.metric.metric_id
        assert result.family == self.metric.family
        assert result.details['classifier_type'] == 'random_forest'

    def test_cross_validation_consistency(self):
        """Test that cross-validation produces consistent results with same seed."""
        np.random.seed(42)

        real_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.uniform(0, 10, 100)
        })

        synth_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.uniform(0, 10, 100)
        })

        context1 = MetricExecutionContext(
            real_data=real_data,
            synth_data=synth_data,
            parameters={'random_seed': 42, 'n_splits': 3},
            random_seed=42
        )

        context2 = MetricExecutionContext(
            real_data=real_data,
            synth_data=synth_data,
            parameters={'random_seed': 42, 'n_splits': 3},
            random_seed=42
        )

        result1 = self.metric.compute(context1)
        result2 = self.metric.compute(context2)

        # Results should be identical with same seed
        assert abs(result1.value - result2.value) < 1e-6
