"""Unit tests for ML efficiency metrics (TTR, TTS, TRTS, TSTR, TTRS)."""

import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression

from src.valtapy.evaluation.metrics.utility import (
    TTRMetric,
    TTSMetric,
    TRTSMetric,
    TSTRMetric,
    TTRSMetric,
    MachineLearningEfficiency
)
from src.valtapy.evaluation.entities import MetricExecutionContext, MetricFamily
from src.valtapy.evaluation.exceptions import MetricComputationError


@pytest.fixture
def classification_data():
    """Generate synthetic classification datasets for testing."""
    # Real data
    X_real, y_real = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_classes=3,
        random_state=42
    )
    real_df = pd.DataFrame(X_real, columns=[f'feature_{i}' for i in range(10)])
    real_df['target'] = y_real

    # Synthetic data (similar but with some noise)
    X_synth, y_synth = make_classification(
        n_samples=150,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_classes=3,
        random_state=123
    )
    synth_df = pd.DataFrame(X_synth, columns=[f'feature_{i}' for i in range(10)])
    synth_df['target'] = y_synth

    return real_df, synth_df


@pytest.fixture
def regression_data():
    """Generate synthetic regression datasets for testing."""
    # Real data
    X_real, y_real = make_regression(
        n_samples=200,
        n_features=10,
        n_informative=7,
        noise=10,
        random_state=42
    )
    real_df = pd.DataFrame(X_real, columns=[f'feature_{i}' for i in range(10)])
    real_df['target'] = y_real

    # Synthetic data
    X_synth, y_synth = make_regression(
        n_samples=150,
        n_features=10,
        n_informative=7,
        noise=10,
        random_state=123
    )
    synth_df = pd.DataFrame(X_synth, columns=[f'feature_{i}' for i in range(10)])
    synth_df['target'] = y_synth

    return real_df, synth_df


class TestTTRMetric:
    """Tests for TTR (Train on Training, Test on Real) metric."""

    def test_initialization(self):
        """Test TTR metric initialization."""
        metric = TTRMetric()
        assert metric.metric_id == "ml_efficiency_ttr"
        assert metric.family == MetricFamily.UTILITY
        assert metric.test_size == 0.2
        assert metric.random_seed == 42

    def test_compute_classification(self, classification_data):
        """Test TTR computation with classification data."""
        real_df, synth_df = classification_data
        metric = TTRMetric()
        context = MetricExecutionContext(
            real_data=real_df,
            synth_data=synth_df,
            random_seed=42
        )

        result = metric.compute(context)

        assert result.metric_id == "ml_efficiency_ttr"
        assert result.family == MetricFamily.UTILITY
        assert 0 <= result.value <= 1  # Accuracy should be between 0 and 1
        assert 'accuracy' in result.details
        assert 'f1_score' in result.details
        assert 'precision' in result.details
        assert 'recall' in result.details
        assert result.details['task_type'] == 'classification'
        assert result.details['baseline'] is True
        assert result.details['data_source']['train'] == 'real'
        assert result.details['data_source']['test'] == 'real'

    def test_compute_regression(self, regression_data):
        """Test TTR computation with regression data."""
        real_df, synth_df = regression_data
        metric = TTRMetric()
        context = MetricExecutionContext(
            real_data=real_df,
            synth_data=synth_df,
            random_seed=42
        )

        result = metric.compute(context)

        assert result.metric_id == "ml_efficiency_ttr"
        assert result.family == MetricFamily.UTILITY
        assert 'rmse' in result.details
        assert 'mae' in result.details
        assert 'r2_score' in result.details
        assert result.details['task_type'] == 'regression'

    def test_can_compute(self, classification_data):
        """Test can_compute validation."""
        real_df, synth_df = classification_data
        metric = TTRMetric()

        assert metric.can_compute(real_df, synth_df) is True

        # Test with invalid data
        empty_df = pd.DataFrame()
        assert metric.can_compute(empty_df, synth_df) is False
        assert metric.can_compute(real_df, empty_df) is False


class TestTTSMetric:
    """Tests for TTS (Train on Training, Test on Synthetic) metric."""

    def test_initialization(self):
        """Test TTS metric initialization."""
        metric = TTSMetric()
        assert metric.metric_id == "ml_efficiency_tts"
        assert metric.family == MetricFamily.UTILITY

    def test_compute_classification(self, classification_data):
        """Test TTS computation with classification data."""
        real_df, synth_df = classification_data
        metric = TTSMetric()
        context = MetricExecutionContext(
            real_data=real_df,
            synth_data=synth_df,
            random_seed=42
        )

        result = metric.compute(context)

        assert result.metric_id == "ml_efficiency_tts"
        assert 0 <= result.value <= 1
        assert result.details['baseline'] is False
        assert result.details['data_source']['train'] == 'real'
        assert result.details['data_source']['test'] == 'synthetic'
        assert 'evaluation_aspect' in result.details


class TestTRTSMetric:
    """Tests for TRTS (Train on Real, Test on Synthetic) metric."""

    def test_initialization(self):
        """Test TRTS metric initialization."""
        metric = TRTSMetric()
        assert metric.metric_id == "ml_efficiency_trts"
        assert metric.family == MetricFamily.UTILITY

    def test_compute_classification(self, classification_data):
        """Test TRTS computation with classification data."""
        real_df, synth_df = classification_data
        metric = TRTSMetric()
        context = MetricExecutionContext(
            real_data=real_df,
            synth_data=synth_df,
            random_seed=42
        )

        result = metric.compute(context)

        assert result.metric_id == "ml_efficiency_trts"
        assert result.details['data_source']['train'] == 'real_full'
        assert result.details['data_source']['test'] == 'synthetic'
        assert result.details['evaluation_aspect'] == 'distribution_match'


class TestTSTRMetric:
    """Tests for TSTR (Train on Synthetic, Test on Real) metric."""

    def test_initialization(self):
        """Test TSTR metric initialization."""
        metric = TSTRMetric()
        assert metric.metric_id == "ml_efficiency_tstr"
        assert metric.family == MetricFamily.UTILITY

    def test_compute_classification(self, classification_data):
        """Test TSTR computation with classification data."""
        real_df, synth_df = classification_data
        metric = TSTRMetric()
        context = MetricExecutionContext(
            real_data=real_df,
            synth_data=synth_df,
            random_seed=42
        )

        result = metric.compute(context)

        assert result.metric_id == "ml_efficiency_tstr"
        assert result.details['data_source']['train'] == 'synthetic'
        assert result.details['data_source']['test'] == 'real'
        assert result.details['evaluation_aspect'] == 'generalization_to_real'
        assert 'utility_indicator' in result.details


class TestTTRSMetric:
    """Tests for TTRS (Train on Training, Test on Real and Synthetic) metric."""

    def test_initialization(self):
        """Test TTRS metric initialization."""
        metric = TTRSMetric()
        assert metric.metric_id == "ml_efficiency_ttrs"
        assert metric.family == MetricFamily.UTILITY

    def test_compute_classification(self, classification_data):
        """Test TTRS computation with classification data."""
        real_df, synth_df = classification_data
        metric = TTRSMetric()
        context = MetricExecutionContext(
            real_data=real_df,
            synth_data=synth_df,
            random_seed=42
        )

        result = metric.compute(context)

        assert result.metric_id == "ml_efficiency_ttrs"
        assert result.details['data_source']['train'] == 'real'
        assert result.details['data_source']['test'] == 'real_and_synthetic_combined'
        assert result.details['evaluation_aspect'] == 'cross_data_consistency'
        # Test samples should be larger than just real test samples
        assert result.details['test_samples'] > 40  # 20% of 200 is 40


class TestMLEfficiencyValidation:
    """Tests for validation and error handling."""

    def test_validate_parameters(self):
        """Test parameter validation."""
        metric = TTRMetric()

        # Valid parameters
        assert metric.validate_parameters({'test_size': 0.3, 'random_seed': 123}) is True

        # Invalid test_size
        assert metric.validate_parameters({'test_size': 1.5}) is False
        assert metric.validate_parameters({'test_size': -0.1}) is False
        assert metric.validate_parameters({'test_size': 0}) is False

        # Invalid random_seed
        assert metric.validate_parameters({'random_seed': -1}) is False
        assert metric.validate_parameters({'random_seed': 'invalid'}) is False

    def test_empty_data_raises_error(self):
        """Test that empty data raises appropriate error."""
        metric = TTRMetric()
        empty_df = pd.DataFrame()
        real_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})

        context = MetricExecutionContext(
            real_data=empty_df,
            synth_data=real_df,
            random_seed=42
        )

        with pytest.raises(MetricComputationError):
            metric.compute(context)

    def test_mismatched_columns_raises_error(self):
        """Test that mismatched columns raise error."""
        metric = TTRMetric()
        real_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        synth_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})

        context = MetricExecutionContext(
            real_data=real_df,
            synth_data=synth_df,
            random_seed=42
        )

        with pytest.raises(MetricComputationError):
            metric.compute(context)

    def test_insufficient_samples_raises_error(self):
        """Test that insufficient samples raise error."""
        metric = TTRMetric()
        real_df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        synth_df = pd.DataFrame({'a': [5, 6], 'b': [7, 8]})

        context = MetricExecutionContext(
            real_data=real_df,
            synth_data=synth_df,
            random_seed=42
        )

        with pytest.raises(MetricComputationError):
            metric.compute(context)


class TestMLEfficiencyComparison:
    """Tests comparing different ML efficiency techniques."""

    def test_all_metrics_produce_consistent_results(self, classification_data):
        """Test that all metrics run successfully and produce valid results."""
        real_df, synth_df = classification_data
        metrics = [
            TTRMetric(),
            TTSMetric(),
            TRTSMetric(),
            TSTRMetric(),
            TTRSMetric()
        ]

        results = []
        for metric in metrics:
            context = MetricExecutionContext(
                real_data=real_df,
                synth_data=synth_df,
                random_seed=42
            )
            result = metric.compute(context)
            results.append(result)

        # Check all results are valid
        assert len(results) == 5
        for result in results:
            assert result.family == MetricFamily.UTILITY
            assert 0 <= result.value <= 1  # Assuming classification
            assert result.computation_time >= 0
            assert 'task_type' in result.details
            assert 'data_source' in result.details

    def test_ttr_baseline_vs_tstr_comparison(self, classification_data):
        """Test that TTR and TSTR can be compared (common use case)."""
        real_df, synth_df = classification_data

        ttr_metric = TTRMetric()
        tstr_metric = TSTRMetric()

        context = MetricExecutionContext(
            real_data=real_df,
            synth_data=synth_df,
            random_seed=42
        )

        ttr_result = ttr_metric.compute(context)
        tstr_result = tstr_metric.compute(context)

        # Both should have the same primary metric
        assert ttr_result.details['primary_metric'] == tstr_result.details['primary_metric']

        # TSTR performance relative to TTR indicates synthetic data quality
        performance_ratio = tstr_result.value / ttr_result.value if ttr_result.value > 0 else 0
        assert 0 <= performance_ratio <= 2  # Reasonable range


class TestMachineLearningEfficiency:
    """Tests for the unified MachineLearningEfficiency metric."""

    def test_initialization(self):
        """Test MachineLearningEfficiency initialization."""
        metric = MachineLearningEfficiency()
        assert metric.metric_id == "ml_efficiency"
        assert metric.family == MetricFamily.UTILITY
        assert metric.test_size == 0.2
        assert metric.random_seed == 42
        assert len(metric.techniques) == 5  # Default: all 5 techniques

    def test_initialization_with_custom_techniques(self):
        """Test initialization with specific techniques."""
        metric = MachineLearningEfficiency(techniques=['ttr', 'tstr'])
        assert len(metric.techniques) == 2
        assert 'ttr' in metric.techniques
        assert 'tstr' in metric.techniques
        assert len(metric._technique_instances) == 2

    def test_compute_classification(self, classification_data):
        """Test MachineLearningEfficiency with classification data."""
        real_df, synth_df = classification_data
        metric = MachineLearningEfficiency()
        context = MetricExecutionContext(
            real_data=real_df,
            synth_data=synth_df,
            random_seed=42
        )

        result = metric.compute(context)

        # Check basic properties
        assert result.metric_id == "ml_efficiency"
        assert result.family == MetricFamily.UTILITY
        assert 0 <= result.value <= 1.5  # Can exceed 1.0 if synthetic is better
        assert result.computation_time > 0

        # Check details structure
        assert 'techniques_evaluated' in result.details
        assert 'baseline_technique' in result.details
        assert 'technique_scores' in result.details
        assert 'relative_efficiencies' in result.details
        assert 'task_type' in result.details
        assert 'primary_metric' in result.details

        # Check that all techniques were evaluated
        assert len(result.details['techniques_evaluated']) == 5
        assert all(t in result.details['technique_scores'] for t in metric.techniques)
        assert all(t in result.details['relative_efficiencies'] for t in metric.techniques)

        # Check TTR is the baseline (efficiency = 1.0)
        assert result.details['relative_efficiencies']['ttr'] == 1.0

        # Check individual technique details are present
        for technique in metric.techniques:
            details_key = f'{technique}_details'
            assert details_key in result.details
            assert 'value' in result.details[details_key]
            assert 'computation_time' in result.details[details_key]
            assert 'samples' in result.details[details_key]

    def test_compute_regression(self, regression_data):
        """Test MachineLearningEfficiency with regression data."""
        real_df, synth_df = regression_data
        metric = MachineLearningEfficiency()
        context = MetricExecutionContext(
            real_data=real_df,
            synth_data=synth_df,
            random_seed=42
        )

        result = metric.compute(context)

        assert result.metric_id == "ml_efficiency"
        assert result.details['task_type'] == 'regression'
        assert result.details['primary_metric'] == 'r2_score'
        assert 'technique_scores' in result.details
        assert 'relative_efficiencies' in result.details

    def test_efficiency_calculation(self, classification_data):
        """Test that efficiency calculations are correct."""
        real_df, synth_df = classification_data
        metric = MachineLearningEfficiency(techniques=['ttr', 'tstr'])
        context = MetricExecutionContext(
            real_data=real_df,
            synth_data=synth_df,
            random_seed=42
        )

        result = metric.compute(context)

        ttr_score = result.details['technique_scores']['ttr']
        tstr_score = result.details['technique_scores']['tstr']
        tstr_efficiency = result.details['relative_efficiencies']['tstr']

        # Manual calculation check
        expected_efficiency = tstr_score / ttr_score if ttr_score > 0 else 0
        assert abs(tstr_efficiency - expected_efficiency) < 1e-6

        # TTR should always be 1.0 (baseline)
        assert result.details['relative_efficiencies']['ttr'] == 1.0

    def test_aggregation_score(self, classification_data):
        """Test that aggregation produces a reasonable score."""
        real_df, synth_df = classification_data
        metric = MachineLearningEfficiency()
        context = MetricExecutionContext(
            real_data=real_df,
            synth_data=synth_df,
            random_seed=42
        )

        result = metric.compute(context)

        # Aggregated score should be in reasonable range
        assert 0 <= result.value <= 1.5

        # Should be influenced most by TSTR (50% weight)
        tstr_efficiency = result.details['relative_efficiencies']['tstr']
        # The aggregated score should be reasonably close to TSTR efficiency
        # (not exactly equal due to other techniques)
        assert abs(result.value - tstr_efficiency) < 0.5

    def test_selective_techniques(self, classification_data):
        """Test running only specific techniques."""
        real_df, synth_df = classification_data

        # Test with only TTR and TSTR
        metric = MachineLearningEfficiency(techniques=['ttr', 'tstr'])
        context = MetricExecutionContext(
            real_data=real_df,
            synth_data=synth_df,
            random_seed=42
        )

        result = metric.compute(context)

        assert len(result.details['techniques_evaluated']) == 2
        assert 'ttr' in result.details['technique_scores']
        assert 'tstr' in result.details['technique_scores']
        assert 'tts' not in result.details['technique_scores']
        assert 'trts' not in result.details['technique_scores']
        assert 'ttrs' not in result.details['technique_scores']

    def test_can_compute(self, classification_data):
        """Test can_compute validation."""
        real_df, synth_df = classification_data
        metric = MachineLearningEfficiency()

        assert metric.can_compute(real_df, synth_df) is True

        # Test with invalid data
        empty_df = pd.DataFrame()
        assert metric.can_compute(empty_df, synth_df) is False
        assert metric.can_compute(real_df, empty_df) is False

    def test_validate_parameters(self):
        """Test parameter validation."""
        metric = MachineLearningEfficiency()

        # Valid parameters
        assert metric.validate_parameters({'test_size': 0.3, 'random_seed': 123}) is True
        assert metric.validate_parameters({'techniques': ['ttr', 'tstr']}) is True

        # Invalid test_size
        assert metric.validate_parameters({'test_size': 1.5}) is False
        assert metric.validate_parameters({'test_size': -0.1}) is False

        # Invalid random_seed
        assert metric.validate_parameters({'random_seed': -1}) is False

        # Invalid techniques
        assert metric.validate_parameters({'techniques': 'not_a_list'}) is False
        assert metric.validate_parameters({'techniques': ['invalid_technique']}) is False
        assert metric.validate_parameters({'techniques': ['ttr', 'invalid']}) is False

    def test_custom_model_factory(self, classification_data):
        """Test using a custom model factory."""
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        class CustomModelFactory:
            def create_classifier(self, random_state=42, **kwargs):
                return RandomForestClassifier(n_estimators=10, random_state=random_state)

            def create_regressor(self, random_state=42, **kwargs):
                return RandomForestRegressor(n_estimators=10, random_state=random_state)

            def supports_classification(self):
                return True

            def supports_regression(self):
                return True

        real_df, synth_df = classification_data
        metric = MachineLearningEfficiency(
            model_factory=CustomModelFactory(),
            techniques=['ttr', 'tstr']  # Limit techniques for faster execution
        )
        context = MetricExecutionContext(
            real_data=real_df,
            synth_data=synth_df,
            random_seed=42
        )

        result = metric.compute(context)

        assert result.metric_id == "ml_efficiency"
        assert result.details['model_type'] == 'CustomModelFactory'
        assert 'technique_scores' in result.details
        assert 'relative_efficiencies' in result.details

    def test_computation_time_tracking(self, classification_data):
        """Test that computation times are tracked correctly."""
        real_df, synth_df = classification_data
        metric = MachineLearningEfficiency(techniques=['ttr', 'tstr'])
        context = MetricExecutionContext(
            real_data=real_df,
            synth_data=synth_df,
            random_seed=42
        )

        result = metric.compute(context)

        # Overall computation time should be positive
        assert result.computation_time > 0

        # Each technique should have computation time
        assert result.details['ttr_details']['computation_time'] > 0
        assert result.details['tstr_details']['computation_time'] > 0

        # Overall time should be roughly the sum of technique times
        technique_times = (
            result.details['ttr_details']['computation_time'] +
            result.details['tstr_details']['computation_time']
        )
        # Allow some overhead (10%)
        assert result.computation_time <= technique_times * 1.1

    def test_metadata(self, classification_data):
        """Test that metadata is properly set."""
        real_df, synth_df = classification_data
        metric = MachineLearningEfficiency(test_size=0.3, random_seed=99)
        context = MetricExecutionContext(
            real_data=real_df,
            synth_data=synth_df,
            random_seed=42
        )

        result = metric.compute(context)

        assert 'test_size' in result.metadata
        assert 'random_seed' in result.metadata
        assert 'n_techniques' in result.metadata
        assert result.metadata['n_techniques'] == 5
