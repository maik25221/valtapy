from typing import Any, Dict, Tuple

import pandas as pd

from .base_ml_validator import (
    BaseMLValidator,
    MLModelConfig,
    ValidationMetrics,
    ValidationTechnique,
)


class TRTSValidator(BaseMLValidator):
    """Train on Real, Test on Synthetic - Fidelity validation"""

    def __init__(self, random_seed: int = 42, test_size: float = 0.2):
        config = MLModelConfig(random_seed=random_seed, test_size=test_size)
        super().__init__(config)

    def _prepare_data(
        self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame
    ) -> Tuple[Tuple[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series]]:
        """Prepare data for TRTS: train on real, test on synthetic."""
        # Training data: split real data and use train portion
        X_real, y_real = self._prepare_dataset(real_data)
        X_train, _, y_train, _ = self.data_splitter.split_data(
            X_real, y_real, self.config.test_size, self.config.random_seed
        )

        # Testing data: use all synthetic data
        X_test, y_test = self._prepare_dataset(synthetic_data)

        return (X_train, y_train), (X_test, y_test)

    def get_technique_name(self) -> ValidationTechnique:
        """Return the technique name."""
        return ValidationTechnique.TRTS

    def get_description(self) -> str:
        """Return the technique description."""
        return "Train on Real, Test on Synthetic (Fidelity)"

    def _add_technique_specific_metrics(
        self, result: Dict[str, Any], metrics: ValidationMetrics
    ) -> None:
        """Add TRTS-specific metrics."""
        result["fidelity_level"] = self._classify_fidelity_level(metrics.accuracy)

    @staticmethod
    def _classify_fidelity_level(accuracy: float) -> str:
        """Classify fidelity level based on accuracy."""
        if accuracy > 0.7:
            return "high"
        elif accuracy > 0.5:
            return "medium"
        else:
            return "low"
