"""Example demonstrating the MachineLearningEfficiency metric.

This example shows how to use the unified ML efficiency metric to evaluate
synthetic data utility using different ML model strategies.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier

# Import ValtaPy evaluation components
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.valtapy.evaluation.entities import MetricExecutionContext
from src.valtapy.evaluation.metrics.utility import (
    MachineLearningEfficiency,
    DecisionTreeModelFactory
)


def generate_sample_data(task_type='classification', n_samples=1000, n_features=10):
    """Generate sample synthetic and real datasets for testing.

    Args:
        task_type: 'classification' or 'regression'
        n_samples: Number of samples to generate
        n_features: Number of features

    Returns:
        Tuple of (real_df, synthetic_df)
    """
    if task_type == 'classification':
        # Generate real data
        X_real, y_real = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=7,
            n_redundant=2,
            n_classes=3,
            random_state=42
        )

        # Generate synthetic data (with some noise to simulate imperfect generation)
        X_synth, y_synth = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=7,
            n_redundant=2,
            n_classes=3,
            random_state=43  # Different seed
        )
        # Add some noise to make it slightly different
        X_synth = X_synth + np.random.normal(0, 0.1, X_synth.shape)

    else:  # regression
        X_real, y_real = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=8,
            noise=10,
            random_state=42
        )

        X_synth, y_synth = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=8,
            noise=12,
            random_state=43
        )
        X_synth = X_synth + np.random.normal(0, 0.1, X_synth.shape)

    # Create DataFrames
    feature_names = [f'feature_{i}' for i in range(n_features)]

    real_df = pd.DataFrame(X_real, columns=feature_names)
    real_df['target'] = y_real

    synth_df = pd.DataFrame(X_synth, columns=feature_names)
    synth_df['target'] = y_synth

    return real_df, synth_df


def example_basic_usage():
    """Example 1: Basic usage with default DecisionTree strategy."""
    print("=" * 80)
    print("Example 1: Basic Usage with Default DecisionTree")
    print("=" * 80)

    # Generate sample data
    real_df, synth_df = generate_sample_data(task_type='classification')

    # Create the metric with default settings (DecisionTree)
    metric = MachineLearningEfficiency()

    # Create execution context
    context = MetricExecutionContext(
        real_data=real_df,
        synth_data=synth_df,
        parameters={}
    )

    # Compute the metric
    result = metric.compute(context)

    # Display results
    print(f"\nMetric: {result.metric_id}")
    print(f"Overall ML Efficiency Score: {result.value:.4f}")
    print(f"Computation Time: {result.computation_time:.2f}s")
    print(f"\nTask Type: {result.details['task_type']}")
    print(f"Primary Metric: {result.details['primary_metric']}")
    print(f"Model Type: {result.details['model_type']}")

    print("\n--- Individual Technique Scores ---")
    for technique in result.details['techniques_evaluated']:
        score = result.details['technique_scores'][technique]
        print(f"{technique.upper():5s}: {score:.4f}")

    print("\n--- Relative Efficiencies (vs TTR baseline) ---")
    for technique, efficiency in result.details['relative_efficiencies'].items():
        print(f"{technique.upper():5s}: {efficiency:.4f} ({efficiency*100:.1f}% of baseline)")

    print("\n" + "=" * 80 + "\n")


def example_custom_model_strategy():
    """Example 2: Using a custom model strategy (e.g., RandomForest)."""
    print("=" * 80)
    print("Example 2: Custom Model Strategy (RandomForest)")
    print("=" * 80)

    # Generate sample data
    real_df, synth_df = generate_sample_data(task_type='classification')

    # Create a custom model factory for RandomForest
    class RandomForestModelFactory:
        """Custom factory for RandomForest models."""

        def create_classifier(self, random_state=42, **kwargs):
            return RandomForestClassifier(
                n_estimators=50,
                max_depth=10,
                random_state=random_state,
                **kwargs
            )

        def create_regressor(self, random_state=42, **kwargs):
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(
                n_estimators=50,
                max_depth=10,
                random_state=random_state,
                **kwargs
            )

        def supports_classification(self):
            return True

        def supports_regression(self):
            return True

    # Create metric with custom factory
    metric = MachineLearningEfficiency(
        model_factory=RandomForestModelFactory()
    )

    # Create execution context
    context = MetricExecutionContext(
        real_data=real_df,
        synth_data=synth_df,
        parameters={}
    )

    # Compute the metric
    result = metric.compute(context)

    # Display results
    print(f"\nMetric: {result.metric_id}")
    print(f"Overall ML Efficiency Score: {result.value:.4f}")
    print(f"Model Type: {result.details['model_type']}")

    print("\n--- Relative Efficiencies (vs TTR baseline) ---")
    for technique, efficiency in result.details['relative_efficiencies'].items():
        print(f"{technique.upper():5s}: {efficiency:.4f}")

    print("\n" + "=" * 80 + "\n")


def example_selective_techniques():
    """Example 3: Running only specific techniques."""
    print("=" * 80)
    print("Example 3: Selective Techniques (TTR, TSTR only)")
    print("=" * 80)

    # Generate sample data
    real_df, synth_df = generate_sample_data(task_type='classification')

    # Create metric with only TTR and TSTR
    metric = MachineLearningEfficiency(
        techniques=['ttr', 'tstr']  # Only run baseline and key metric
    )

    # Create execution context
    context = MetricExecutionContext(
        real_data=real_df,
        synth_data=synth_df,
        parameters={}
    )

    # Compute the metric
    result = metric.compute(context)

    # Display results
    print(f"\nMetric: {result.metric_id}")
    print(f"Overall ML Efficiency Score: {result.value:.4f}")
    print(f"Techniques Evaluated: {result.details['techniques_evaluated']}")

    print("\n--- Technique Scores ---")
    for technique in result.details['techniques_evaluated']:
        score = result.details['technique_scores'][technique]
        efficiency = result.details['relative_efficiencies'][technique]
        print(f"{technique.upper():5s}: Score={score:.4f}, Efficiency={efficiency:.4f}")

    print("\n" + "=" * 80 + "\n")


def example_regression_task():
    """Example 4: Using ML efficiency with regression task."""
    print("=" * 80)
    print("Example 4: Regression Task")
    print("=" * 80)

    # Generate sample regression data
    real_df, synth_df = generate_sample_data(task_type='regression')

    # Create metric
    metric = MachineLearningEfficiency()

    # Create execution context
    context = MetricExecutionContext(
        real_data=real_df,
        synth_data=synth_df,
        parameters={}
    )

    # Compute the metric
    result = metric.compute(context)

    # Display results
    print(f"\nMetric: {result.metric_id}")
    print(f"Overall ML Efficiency Score: {result.value:.4f}")
    print(f"Task Type: {result.details['task_type']}")
    print(f"Primary Metric: {result.details['primary_metric']}")

    print("\n--- Individual Technique Scores (R² Score) ---")
    for technique in result.details['techniques_evaluated']:
        score = result.details['technique_scores'][technique]
        print(f"{technique.upper():5s}: {score:.4f}")

    print("\n--- Relative Efficiencies ---")
    for technique, efficiency in result.details['relative_efficiencies'].items():
        print(f"{technique.upper():5s}: {efficiency:.4f}")

    print("\n" + "=" * 80 + "\n")


def example_detailed_inspection():
    """Example 5: Detailed inspection of results."""
    print("=" * 80)
    print("Example 5: Detailed Result Inspection")
    print("=" * 80)

    # Generate sample data
    real_df, synth_df = generate_sample_data(task_type='classification', n_samples=500)

    # Create metric
    metric = MachineLearningEfficiency()

    # Create execution context
    context = MetricExecutionContext(
        real_data=real_df,
        synth_data=synth_df,
        parameters={'test_size': 0.3}  # Use 30% for testing
    )

    # Compute the metric
    result = metric.compute(context)

    # Detailed inspection
    print("\n--- Comprehensive Results ---")
    print(f"Overall Efficiency: {result.value:.4f}")
    print(f"Computation Time: {result.computation_time:.2f}s")

    print("\n--- Per-Technique Details ---")
    for technique in result.details['techniques_evaluated']:
        details_key = f'{technique}_details'
        technique_details = result.details[details_key]

        print(f"\n{technique.upper()}:")
        print(f"  Score: {technique_details['value']:.4f}")
        print(f"  Train Samples: {technique_details['samples']['train']}")
        print(f"  Test Samples: {technique_details['samples']['test']}")
        print(f"  Computation Time: {technique_details['computation_time']:.3f}s")

    print("\n--- Efficiency Analysis ---")
    ttr_score = result.details['technique_scores']['ttr']
    tstr_score = result.details['technique_scores']['tstr']
    tstr_efficiency = result.details['relative_efficiencies']['tstr']

    print(f"Baseline (TTR) Score: {ttr_score:.4f}")
    print(f"TSTR Score: {tstr_score:.4f}")
    print(f"TSTR Efficiency: {tstr_efficiency:.4f}")

    if tstr_efficiency >= 0.9:
        print("[+] Excellent: Synthetic data is highly effective for training!")
    elif tstr_efficiency >= 0.75:
        print("[+] Good: Synthetic data captures most important patterns")
    elif tstr_efficiency >= 0.6:
        print("[~] Moderate: Synthetic data has some utility but room for improvement")
    else:
        print("[-] Poor: Synthetic data may not be suitable for training")

    print("\n" + "=" * 80 + "\n")


if __name__ == '__main__':
    # Run all examples
    example_basic_usage()
    example_custom_model_strategy()
    example_selective_techniques()
    example_regression_task()
    example_detailed_inspection()

    print("\n" + "=" * 80)
    print("All examples completed successfully!")
    print("=" * 80)
