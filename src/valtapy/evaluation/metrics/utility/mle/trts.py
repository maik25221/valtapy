"""TRTS (Train on Real, Test on Synthetic) metric implementation."""

from typing import Tuple
import pandas as pd

from .base_ml_efficiency import BaseMLEfficiencyMetric


class TRTSMetric(BaseMLEfficiencyMetric):
    """Train on Real data, Test on Synthetic data.

    This metric trains a model on all available real data and tests it
    on synthetic data. It's similar to TTS but uses the full real dataset
    for training, which may give a better assessment of how well the
    synthetic data matches the real data distribution.

    This is useful to check if synthetic data captures the same relationships
    that a model learned from the complete real dataset.
    """

    def __init__(self, test_size: float = 0.2, random_seed: int = 42, **kwargs):
        """Initialize TRTS metric.

        Args:
            test_size: Not used in TRTS (all real data used for training)
            random_seed: Random seed for reproducibility (default: 42)
            **kwargs: Additional parameters passed to BaseMLEfficiencyMetric
        """
        super().__init__(
            metric_id="ml_efficiency_trts",
            name="TRTS - Train on Real, Test on Synthetic",
            description=(
                "Trains on all real data and tests on synthetic data. "
                "Evaluates how well synthetic data matches the complete real data distribution."
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
        """Prepare data for TRTS: train on all real data, test on synthetic.

        Args:
            X_real: Real features
            y_real: Real target
            X_synth: Synthetic features
            y_synth: Synthetic target
            test_size: Not used (kept for interface compatibility)
            random_seed: Random seed (not used but kept for interface)

        Returns:
            Tuple of (X_real, y_real, X_synth, y_synth)
        """
        # Use all real data for training, all synthetic for testing
        return X_real, y_real, X_synth, y_synth

    def _add_technique_specific_details(
        self,
        details: dict,
        performance_metrics: dict
    ) -> None:
        """Add TRTS-specific details.

        Args:
            details: Dictionary to add details to (modified in-place)
            performance_metrics: Computed performance metrics
        """
        details['baseline'] = False
        details['data_source'] = {
            'train': 'real_full',
            'test': 'synthetic'
        }
        details['evaluation_aspect'] = 'distribution_match'
