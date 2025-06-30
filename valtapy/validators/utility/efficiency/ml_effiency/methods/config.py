"""
Advanced configuration classes for ML validation.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class MLAlgorithm(Enum):
    """Supported ML algorithms."""

    CATBOOST = "catboost"
    # Future: XGBOOST = "xgboost"
    # Future: RANDOM_FOREST = "random_forest"


@dataclass
class AdvancedMLConfig:
    """Advanced configuration for ML validation with more options."""

    random_seed: int = 42
    test_size: float = 0.2
    algorithm: MLAlgorithm = MLAlgorithm.CATBOOST
    verbose: bool = False
    eval_metric: str = "F1"

    # CatBoost specific parameters
    catboost_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "iterations": 100,
            "depth": 6,
            "learning_rate": 0.1,
            "loss_function": "Logloss",
            "bootstrap_type": "Bayesian",
            "bagging_temperature": 1,
            "od_type": "Iter",
            "od_wait": 20,
        }
    )

    # Cross-validation settings
    use_cross_validation: bool = False
    cv_folds: int = 5

    # Early stopping
    early_stopping_rounds: Optional[int] = 20

    def get_model_params(self) -> Dict[str, Any]:
        """Get parameters for the specified algorithm."""
        base_params = {
            "random_seed": self.random_seed,
            "verbose": self.verbose,
            "eval_metric": self.eval_metric,
        }

        if self.algorithm == MLAlgorithm.CATBOOST:
            base_params.update(self.catboost_params)
            if self.early_stopping_rounds:
                base_params["early_stopping_rounds"] = self.early_stopping_rounds

        return base_params
