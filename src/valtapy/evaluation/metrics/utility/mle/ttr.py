"""TTR (Train on Training, Test on Real) metric implementation."""

from typing import Tuple
import pandas as pd

from .base_ml_efficiency import BaseMLEfficiencyMetric


class TTRMetric(BaseMLEfficiencyMetric):
    """Train on Training data, Test on Real data.

    This is typically the baseline metric for ML efficiency evaluation.
    It trains a model on a portion of the real data (training set) and
    tests it on another portion of real data (test set).

    This establishes the baseline performance that other techniques
    (TTS, TSTR, TRTS, TTRS) will be compared against.
    """

    def __init__(self, test_size: float = 0.2, random_seed: int = 42, **kwargs):
        """Initialize TTR metric.

        Args:
            test_size: Proportion of real data to use for testing (default: 0.2)
            random_seed: Random seed for reproducibility (default: 42)
            **kwargs: Additional parameters passed to BaseMLEfficiencyMetric
        """
        super().__init__(
            metric_id="ml_efficiency_ttr",
            name="TTR - Train on Training, Test on Real",
            description=(
                "Baseline metric: trains on real training data and tests on real test data. "
                "Establishes the baseline performance for comparing synthetic data quality."
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
        """Prepare data for TTR: split real data into train and test.

        Args:
            X_real: Real features
            y_real: Real target
            X_synth: Synthetic features (not used in TTR)
            y_synth: Synthetic target (not used in TTR)
            test_size: Proportion for test split
            random_seed: Random seed

        Returns:
            Tuple of (train_X, train_y, test_X, test_y) all from real data
        """
        # Split real data into train and test
        X_train, X_test, y_train, y_test = self.data_splitter.split(
            X_real, y_real, test_size, random_seed
        )

        return X_train, y_train, X_test, y_test

    def _add_technique_specific_details(
        self,
        details: dict,
        performance_metrics: dict
    ) -> None:
        """Add TTR-specific details.

        Args:
            details: Dictionary to add details to (modified in-place)
            performance_metrics: Computed performance metrics
        """
        details['baseline'] = True
        details['data_source'] = {
            'train': 'real',
            'test': 'real'
        }
