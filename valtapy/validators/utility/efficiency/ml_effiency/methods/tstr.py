from typing import Any, Dict, Tuple

import pandas as pd

from .base_ml_validator import (
    BaseMLValidator,
    MLModelConfig,
    ValidationMetrics,
    ValidationTechnique,
)


class TSTRValidator(BaseMLValidator):
    """Train on Synthetic, Test on Real - Generalization capability validation"""

    def __init__(self, random_seed: int = 42, test_size: float = 0.2):
        config = MLModelConfig(random_seed=random_seed, test_size=test_size)
        super().__init__(config)

    def _prepare_data(
        self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame
    ) -> Tuple[Tuple[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series]]:
        """Prepare data for TSTR: train on synthetic, test on real."""
        # Training data: use all synthetic data
        X_train, y_train = self._prepare_dataset(synthetic_data)

        # Testing data: split real data and use test portion
        X_real, y_real = self._prepare_dataset(real_data)
        _, X_test, _, y_test = self.data_splitter.split_data(
            X_real, y_real, self.config.test_size, self.config.random_seed
        )

        return (X_train, y_train), (X_test, y_test)

    def get_technique_name(self) -> ValidationTechnique:
        """Return the technique name."""
        return ValidationTechnique.TSTR

    def get_description(self) -> str:
        """Return the technique description."""
        return "Train on Synthetic, Test on Real (Generalization)"

    def _add_technique_specific_metrics(
        self, result: Dict[str, Any], metrics: ValidationMetrics
    ) -> None:
        """Add TSTR-specific metrics."""
        result["generalization_capability"] = self._classify_generalization_level(
            metrics.accuracy
        )

    @staticmethod
    def _classify_generalization_level(accuracy: float) -> str:
        """Classify generalization capability based on accuracy."""
        if accuracy > 0.7:
            return "high"
        elif accuracy > 0.5:
            return "medium"
        else:
            return "low"
