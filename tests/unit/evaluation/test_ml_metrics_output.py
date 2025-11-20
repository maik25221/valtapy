"""Test to verify all ML metrics are properly returned in results."""

import pytest
import pandas as pd
from sklearn.datasets import make_classification

from src.valtapy.evaluation.metrics.utility import TSTRMetric, TTRMetric
from src.valtapy.evaluation.entities import MetricExecutionContext


@pytest.fixture
def sample_data():
    """Generate sample classification data."""
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_classes=3,
        random_state=42
    )
    real_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
    real_df['target'] = y

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


def test_all_classification_metrics_are_returned(sample_data):
    """Test that all classification metrics are returned in details."""
    real_df, synth_df = sample_data
    metric = TSTRMetric()

    context = MetricExecutionContext(
        real_data=real_df,
        synth_data=synth_df,
        random_seed=42
    )

    result = metric.compute(context)

    # Check that all metrics are in details
    assert 'accuracy' in result.details
    assert 'f1_score' in result.details
    assert 'precision' in result.details
    assert 'recall' in result.details
    assert 'auc_roc' in result.details

    # Check that all metrics are valid numbers
    assert 0 <= result.details['accuracy'] <= 1
    assert 0 <= result.details['f1_score'] <= 1
    assert 0 <= result.details['precision'] <= 1
    assert 0 <= result.details['recall'] <= 1
    if result.details['auc_roc'] is not None:
        assert 0 <= result.details['auc_roc'] <= 1

    # Check primary metric
    assert result.details['primary_metric'] == 'accuracy'
    assert result.value == result.details['accuracy']

    print("\n=== Classification Metrics Output ===")
    print(f"Primary (value): {result.value:.4f}")
    print(f"Accuracy: {result.details['accuracy']:.4f}")
    print(f"F1-Score: {result.details['f1_score']:.4f}")
    print(f"Precision: {result.details['precision']:.4f}")
    print(f"Recall: {result.details['recall']:.4f}")
    auc_str = f"{result.details['auc_roc']:.4f}" if result.details['auc_roc'] is not None else "N/A"
    print(f"AUC-ROC: {auc_str}")


def test_metrics_comparison_between_techniques(sample_data):
    """Test comparing metrics between TTR (baseline) and TSTR."""
    real_df, synth_df = sample_data

    ttr = TTRMetric()
    tstr = TSTRMetric()

    context = MetricExecutionContext(
        real_data=real_df,
        synth_data=synth_df,
        random_seed=42
    )

    ttr_result = ttr.compute(context)
    tstr_result = tstr.compute(context)

    print("\n=== TTR (Baseline) vs TSTR Comparison ===")
    print(f"\nTTR (Train on Real, Test on Real):")
    print(f"  Accuracy:  {ttr_result.details['accuracy']:.4f}")
    print(f"  F1-Score:  {ttr_result.details['f1_score']:.4f}")
    print(f"  Precision: {ttr_result.details['precision']:.4f}")
    print(f"  Recall:    {ttr_result.details['recall']:.4f}")

    print(f"\nTSTR (Train on Synthetic, Test on Real):")
    print(f"  Accuracy:  {tstr_result.details['accuracy']:.4f}")
    print(f"  F1-Score:  {tstr_result.details['f1_score']:.4f}")
    print(f"  Precision: {tstr_result.details['precision']:.4f}")
    print(f"  Recall:    {tstr_result.details['recall']:.4f}")

    # Calculate performance ratios
    print(f"\nPerformance Ratio (TSTR/TTR):")
    print(f"  Accuracy:  {tstr_result.details['accuracy']/ttr_result.details['accuracy']:.2%}")
    print(f"  F1-Score:  {tstr_result.details['f1_score']/ttr_result.details['f1_score']:.2%}")
    print(f"  Precision: {tstr_result.details['precision']/ttr_result.details['precision']:.2%}")
    print(f"  Recall:    {tstr_result.details['recall']/ttr_result.details['recall']:.2%}")

    # Quality assessment
    avg_ratio = (
        tstr_result.details['accuracy']/ttr_result.details['accuracy'] +
        tstr_result.details['f1_score']/ttr_result.details['f1_score'] +
        tstr_result.details['precision']/ttr_result.details['precision'] +
        tstr_result.details['recall']/ttr_result.details['recall']
    ) / 4

    print(f"\nAverage Performance Ratio: {avg_ratio:.2%}")

    if avg_ratio >= 0.9:
        quality = "Excellent - Synthetic data is highly useful"
    elif avg_ratio >= 0.75:
        quality = "Good - Synthetic data is useful"
    elif avg_ratio >= 0.6:
        quality = "Moderate - Synthetic data has some utility"
    else:
        quality = "Poor - Synthetic data has limited utility"

    print(f"Synthetic Data Quality: {quality}")

    # All ratios should be positive
    assert avg_ratio > 0


def test_detailed_output_structure(sample_data):
    """Test the complete structure of the output."""
    real_df, synth_df = sample_data
    metric = TSTRMetric()

    context = MetricExecutionContext(
        real_data=real_df,
        synth_data=synth_df,
        random_seed=42
    )

    result = metric.compute(context)

    print("\n=== Complete Result Structure ===")
    print(f"\nTop-level attributes:")
    print(f"  metric_id: {result.metric_id}")
    print(f"  family: {result.family.value}")
    print(f"  value (primary): {result.value:.4f}")
    print(f"  computation_time: {result.computation_time:.4f}s")

    print(f"\nMetadata:")
    for key, value in result.metadata.items():
        print(f"  {key}: {value}")

    print(f"\nDetails (all metrics and info):")
    for key, value in sorted(result.details.items()):
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        elif isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")

    # Verify essential fields
    assert result.metric_id
    assert result.family
    assert result.value is not None
    assert result.computation_time > 0
    assert len(result.details) > 0
    assert len(result.metadata) > 0
