from typing import Tuple

import pandas as pd

from .base_ml_validator import BaseMLValidator, MLModelConfig, ValidationTechnique


class TTRRValidator(BaseMLValidator):
    """Train and Test on Real - Baseline validation using only real data"""

    def __init__(self, random_seed: int = 42, test_size: float = 0.2):
        config = MLModelConfig(random_seed=random_seed, test_size=test_size)
        super().__init__(config)

    def _prepare_data(
        self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame
    ) -> Tuple[Tuple[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series]]:
        """Prepare data for TTRR: both training and testing use real data."""
        X_real, y_real = self._prepare_dataset(real_data)

        # Split real data for training and testing
        X_train, X_test, y_train, y_test = self.data_splitter.split_data(
            X_real, y_real, self.config.test_size, self.config.random_seed
        )

        return (X_train, y_train), (X_test, y_test)

    def get_technique_name(self) -> ValidationTechnique:
        """Return the technique name."""
        return ValidationTechnique.TTRR

    def get_description(self) -> str:
        """Return the technique description."""
        return "Train and Test on Real (Baseline)"
