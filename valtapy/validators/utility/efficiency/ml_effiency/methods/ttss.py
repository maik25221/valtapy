from typing import Any, Dict, Tuple

import pandas as pd

from .base_ml_validator import (
    BaseMLValidator,
    MLModelConfig,
    ValidationMetrics,
    ValidationTechnique,
)


class TTSSValidator(BaseMLValidator):
    """Train and Test on Synthetic - Internal consistency validation"""

    def __init__(self, random_seed: int = 42, test_size: float = 0.2):
        config = MLModelConfig(random_seed=random_seed, test_size=test_size)
        super().__init__(config)

    def _prepare_data(
        self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame
    ) -> Tuple[Tuple[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series]]:
        """Prepare data for TTSS: both training and testing use synthetic data."""
        X_synthetic, y_synthetic = self._prepare_dataset(synthetic_data)

        # Split synthetic data for training and testing
        X_train, X_test, y_train, y_test = self.data_splitter.split_data(
            X_synthetic, y_synthetic, self.config.test_size, self.config.random_seed
        )

        return (X_train, y_train), (X_test, y_test)

    def get_technique_name(self) -> ValidationTechnique:
        """Return the technique name."""
        return ValidationTechnique.TTSS

    def get_description(self) -> str:
        """Return the technique description."""
        return "Train and Test on Synthetic (Internal Consistency)"

    def _add_technique_specific_metrics(
        self, result: Dict[str, Any], metrics: ValidationMetrics
    ) -> None:
        """Add TTSS-specific metrics."""
        result["internal_consistency"] = self._classify_consistency_level(
            metrics.accuracy
        )

    @staticmethod
    def _classify_consistency_level(accuracy: float) -> str:
        """Classify internal consistency level based on accuracy."""
        if accuracy > 0.8:
            return "high"
        elif accuracy > 0.6:
            return "medium"
        else:
            return "low"
