"""TTRS (Train on Training, Test on Real and Synthetic) metric implementation."""

from typing import Tuple
import pandas as pd
import numpy as np

from .base_ml_efficiency import BaseMLEfficiencyMetric


class TTRSMetric(BaseMLEfficiencyMetric):
    """Train on Training data (real), Test on both Real and Synthetic data.

    This metric trains a model on real training data and tests it on
    a combined test set containing both real and synthetic data.

    It provides insights into whether the model performs consistently
    across both data types, which can indicate whether synthetic data
    has similar characteristics to real data.
    """

    def __init__(self, test_size: float = 0.2, random_seed: int = 42, **kwargs):
        """Initialize TTRS metric.

        Args:
            test_size: Proportion of real data to use for testing (default: 0.2)
            random_seed: Random seed for reproducibility (default: 42)
            **kwargs: Additional parameters passed to BaseMLEfficiencyMetric
        """
        super().__init__(
            metric_id="ml_efficiency_ttrs",
            name="TTRS - Train on Training, Test on Real and Synthetic",
            description=(
                "Trains on real training data and tests on combined real and synthetic test data. "
                "Evaluates consistency of model performance across both data types."
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
        """Prepare data for TTRS: train on real training split, test on combined data.

        Args:
            X_real: Real features
            y_real: Real target
            X_synth: Synthetic features
            y_synth: Synthetic target
            test_size: Proportion of real data for testing
            random_seed: Random seed

        Returns:
            Tuple of (train_X_real, train_y_real, combined_X_test, combined_y_test)
        """
        # Split real data into train and test
        X_train, X_test_real, y_train, y_test_real = self.data_splitter.split(
            X_real, y_real, test_size, random_seed
        )

        # Combine real test data with all synthetic data
        X_test_combined = pd.concat([X_test_real, X_synth], axis=0, ignore_index=True)
        y_test_combined = pd.concat([y_test_real, y_synth], axis=0, ignore_index=True)

        return X_train, y_train, X_test_combined, y_test_combined

    def _add_technique_specific_details(
        self,
        details: dict,
        performance_metrics: dict
    ) -> None:
        """Add TTRS-specific details.

        Args:
            details: Dictionary to add details to (modified in-place)
            performance_metrics: Computed performance metrics
        """
        details['baseline'] = False
        details['data_source'] = {
            'train': 'real',
            'test': 'real_and_synthetic_combined'
        }
        details['evaluation_aspect'] = 'cross_data_consistency'
