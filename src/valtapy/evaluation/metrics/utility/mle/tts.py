"""TTS (Train on Training, Test on Synthetic) metric implementation."""

from typing import Tuple
import pandas as pd

from .base_ml_efficiency import BaseMLEfficiencyMetric


class TTSMetric(BaseMLEfficiencyMetric):
    """Train on Training data (real), Test on Synthetic data.

    This metric trains a model on real training data and tests it on
    synthetic data. It helps evaluate if the synthetic data has similar
    patterns to the real data by seeing how well a model trained on
    real data can predict synthetic samples.

    Lower performance compared to TTR might indicate that synthetic data
    has different patterns or distributions.
    """

    def __init__(self, test_size: float = 0.2, random_seed: int = 42, **kwargs):
        """Initialize TTS metric.

        Args:
            test_size: Proportion of real data to use for training split (default: 0.2)
            random_seed: Random seed for reproducibility (default: 42)
            **kwargs: Additional parameters passed to BaseMLEfficiencyMetric
        """
        super().__init__(
            metric_id="ml_efficiency_tts",
            name="TTS - Train on Training, Test on Synthetic",
            description=(
                "Trains on real training data and tests on synthetic data. "
                "Evaluates if synthetic data has similar patterns to real data."
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
        """Prepare data for TTS: train on real training split, test on synthetic.

        Args:
            X_real: Real features
            y_real: Real target
            X_synth: Synthetic features
            y_synth: Synthetic target
            test_size: Proportion for real data training split
            random_seed: Random seed

        Returns:
            Tuple of (train_X_real, train_y_real, test_X_synth, test_y_synth)
        """
        # Split real data to get training portion
        X_train, _, y_train, _ = self.data_splitter.split(
            X_real, y_real, test_size, random_seed
        )

        # Use all synthetic data for testing
        return X_train, y_train, X_synth, y_synth

    def _add_technique_specific_details(
        self,
        details: dict,
        performance_metrics: dict
    ) -> None:
        """Add TTS-specific details.

        Args:
            details: Dictionary to add details to (modified in-place)
            performance_metrics: Computed performance metrics
        """
        details['baseline'] = False
        details['data_source'] = {
            'train': 'real',
            'test': 'synthetic'
        }
        details['evaluation_aspect'] = 'pattern_similarity'
