"""TSTR (Train on Synthetic, Test on Real) metric implementation."""

from typing import Tuple
import pandas as pd

from .base_ml_efficiency import BaseMLEfficiencyMetric


class TSTRMetric(BaseMLEfficiencyMetric):
    """Train on Synthetic data, Test on Real data.

    This is one of the most important metrics for synthetic data utility.
    It trains a model entirely on synthetic data and tests it on real data.

    High performance indicates that the synthetic data contains useful
    patterns and relationships that generalize to real data - meaning the
    synthetic data could potentially be used as a privacy-preserving
    replacement for training ML models.

    Performance close to TTR (baseline) suggests high-quality synthetic data.
    """

    def __init__(self, test_size: float = 0.2, random_seed: int = 42, **kwargs):
        """Initialize TSTR metric.

        Args:
            test_size: Proportion of real data to use for testing (default: 0.2)
            random_seed: Random seed for reproducibility (default: 42)
            **kwargs: Additional parameters passed to BaseMLEfficiencyMetric
        """
        super().__init__(
            metric_id="ml_efficiency_tstr",
            name="TSTR - Train on Synthetic, Test on Real",
            description=(
                "Trains on synthetic data and tests on real data. "
                "Key metric for evaluating synthetic data utility: high performance "
                "indicates synthetic data captures useful patterns for real-world tasks."
            ),
            test_size=test_size,
            random_seed=random_seed,
            **kwargs
        )

    def _prepare_train_test_data(
        self,
        X_real: pd.DataFrame,
        y_real: pd.Series,
        X_synth: pd.DataFrame,
        y_synth: pd.Series,
        test_size: float,
        random_seed: int
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Prepare data for TSTR: train on synthetic, test on real test split.

        Args:
            X_real: Real features
            y_real: Real target
            X_synth: Synthetic features
            y_synth: Synthetic target
            test_size: Proportion of real data for testing
            random_seed: Random seed

        Returns:
            Tuple of (X_synth, y_synth, test_X_real, test_y_real)
        """
        # Split real data to get test portion
        _, X_test, _, y_test = self.data_splitter.split(
            X_real, y_real, test_size, random_seed
        )

        # Use all synthetic data for training
        return X_synth, y_synth, X_test, y_test

    def _add_technique_specific_details(
        self,
        details: dict,
        performance_metrics: dict
    ) -> None:
        """Add TSTR-specific details.

        Args:
            details: Dictionary to add details to (modified in-place)
            performance_metrics: Computed performance metrics
        """
        details['baseline'] = False
        details['data_source'] = {
            'train': 'synthetic',
            'test': 'real'
        }
        details['evaluation_aspect'] = 'generalization_to_real'
        details['utility_indicator'] = 'high' if performance_metrics.get('accuracy', 0) > 0.7 else 'moderate' if performance_metrics.get('accuracy', 0) > 0.5 else 'low'
