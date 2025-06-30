"""
Base classes and utilities for ML validation techniques.
Implements common functionality to avoid code duplication.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from valtapy.validators.utility.efficiency import EfficiencyValidator

from ......exceptions.exceptions import (
    DataPreparationError,
    InvalidDataError,
    MetricsCalculationError,
    ModelTrainingError,
)


class ValidationTechnique(Enum):
    """Enumeration of ML validation techniques."""

    TTRR = "TTRR"
    TTSS = "TTSS"
    TSTR = "TSTR"
    TRTS = "TRTS"


@dataclass
class MLModelConfig:
    """Configuration for ML models."""

    random_seed: int = 42
    verbose: bool = False
    eval_metric: str = "F1"
    test_size: float = 0.2


@dataclass
class ValidationMetrics:
    """Container for validation metrics."""

    accuracy: float
    f1_score: float
    precision: float
    auc_roc: Optional[float]
    train_size: int
    test_size: int


class MetricsCalculator:
    """Utility class for calculating ML metrics."""

    @staticmethod
    def calculate_metrics(
        y_true: pd.Series,
        y_pred: pd.Series,
        y_pred_proba: pd.DataFrame,
        train_size: int,
        test_size: int,
    ) -> ValidationMetrics:
        """Calculate all relevant metrics for validation."""
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted")
        precision = precision_score(y_true, y_pred, average="weighted")

        # Calculate AUC-ROC
        auc_roc = MetricsCalculator._calculate_auc_roc(y_true, y_pred_proba)

        return ValidationMetrics(
            accuracy=accuracy,
            f1_score=f1,
            precision=precision,
            auc_roc=auc_roc,
            train_size=train_size,
            test_size=test_size,
        )

    @staticmethod
    def _calculate_auc_roc(
        y_true: pd.Series, y_pred_proba: pd.DataFrame
    ) -> Optional[float]:
        """Calculate AUC-ROC score safely."""
        try:
            unique_classes = len(y_true.unique())
            if unique_classes == 2:
                return roc_auc_score(y_true, y_pred_proba[:, 1])
            else:
                return roc_auc_score(
                    y_true, y_pred_proba, multi_class="ovr", average="weighted"
                )
        except (ValueError, KeyError):
            return None


class MLModelFactory:
    """Factory for creating ML models."""

    @staticmethod
    def create_catboost_classifier(config: MLModelConfig) -> CatBoostClassifier:
        """Create a configured CatBoost classifier."""
        return CatBoostClassifier(
            random_seed=config.random_seed,
            verbose=config.verbose,
            eval_metric=config.eval_metric,
        )


class DataSplitter:
    """Utility class for data splitting operations."""

    @staticmethod
    def split_data(
        x: pd.DataFrame, y: pd.Series, test_size: float, random_state: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data with stratification when possible."""
        stratify = y if len(y.unique()) > 1 else None
        return train_test_split(
            x, y, test_size=test_size, random_state=random_state, stratify=stratify
        )


class BaseMLValidator(EfficiencyValidator, ABC):
    """Abstract base class for ML validation techniques."""

    def __init__(self, config: Optional[MLModelConfig] = None):
        self.config = config or MLModelConfig()
        self.metrics_calculator = MetricsCalculator()
        self.model_factory = MLModelFactory()
        self.data_splitter = DataSplitter()

    def validate_ml_technique(
        self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Template method for ML validation."""
        try:
            # Validate input data
            self._validate_input_data(real_data, synthetic_data)

            # Prepare training and testing data (strategy pattern)
            train_data, test_data = self._prepare_data(real_data, synthetic_data)

            # Train model
            model = self._train_model(train_data)

            # Make predictions
            predictions = self._make_predictions(model, test_data)

            # Calculate metrics
            metrics = self._calculate_metrics(test_data, predictions, train_data)

            # Build result dictionary
            return self._build_result_dict(metrics, success=True)

        except (
            DataPreparationError,
            ModelTrainingError,
            MetricsCalculationError,
            InvalidDataError,
        ) as e:
            return self._build_error_result(str(e))
        except (RuntimeError, ValueError) as e:
            return self._build_error_result(f"Unexpected error: {str(e)}")

    def _validate_input_data(
        self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame
    ) -> None:
        """Validate input data integrity."""
        if real_data.empty:
            raise InvalidDataError("Real data cannot be empty")
        if synthetic_data.empty:
            raise InvalidDataError("Synthetic data cannot be empty")
        if real_data.shape[1] != synthetic_data.shape[1]:
            raise InvalidDataError(
                "Real and synthetic data must have the same number of columns"
            )
        if real_data.shape[1] < 2:
            raise InvalidDataError(
                "Data must have at least 2 columns (features + target)"
            )

    @abstractmethod
    def _prepare_data(
        self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame
    ) -> Tuple[Tuple[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series]]:
        """Prepare training and testing data according to the validation technique."""
        raise NotImplementedError

    @abstractmethod
    def get_technique_name(self) -> ValidationTechnique:
        """Get the name of the validation technique."""
        raise NotImplementedError

    @abstractmethod
    def get_description(self) -> str:
        """Get the description of the validation technique."""
        raise NotImplementedError

    def _train_model(
        self, train_data: Tuple[pd.DataFrame, pd.Series]
    ) -> CatBoostClassifier:
        """Train the ML model."""
        try:
            X_train, y_train = train_data
            if X_train.empty or y_train.empty:
                raise DataPreparationError("Training data cannot be empty")

            model = self.model_factory.create_catboost_classifier(self.config)
            model.fit(X_train, y_train)
            return model
        except (ValueError, TypeError) as e:
            raise ModelTrainingError(f"Failed to train model: {str(e)}") from e

    def _make_predictions(
        self, model: CatBoostClassifier, test_data: Tuple[pd.DataFrame, pd.Series]
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """Make predictions on test data."""
        try:
            X_test, _ = test_data
            if X_test.empty:
                raise DataPreparationError("Test data cannot be empty")

            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            return y_pred, y_pred_proba
        except Exception as e:
            raise MetricsCalculationError(
                f"Failed to make predictions: {str(e)}"
            ) from e

    def _calculate_metrics(
        self,
        test_data: Tuple[pd.DataFrame, pd.Series],
        predictions: Tuple[pd.Series, pd.DataFrame],
        train_data: Tuple[pd.DataFrame, pd.Series],
    ) -> ValidationMetrics:
        """Calculate validation metrics."""
        try:
            _, y_test = test_data
            y_pred, y_pred_proba = predictions
            X_train, _ = train_data

            return self.metrics_calculator.calculate_metrics(
                y_test, y_pred, y_pred_proba, len(X_train), len(y_test)
            )
        except Exception as e:
            raise MetricsCalculationError(
                f"Failed to calculate metrics: {str(e)}"
            ) from e

    def _build_result_dict(
        self, metrics: ValidationMetrics, success: bool = True
    ) -> Dict[str, Any]:
        """Build the result dictionary."""
        result = {
            "technique": self.get_technique_name().value,
            "description": self.get_description(),
            "accuracy": metrics.accuracy,
            "f1_score": metrics.f1_score,
            "precision": metrics.precision,
            "auc_roc": metrics.auc_roc,
            "train_size": metrics.train_size,
            "test_size": metrics.test_size,
            "status": "success" if success else "failed",
        }

        # Add technique-specific metrics
        self._add_technique_specific_metrics(result, metrics)

        return result

    def _build_error_result(self, error_message: str) -> Dict[str, Any]:
        """Build error result dictionary."""
        return {
            "technique": self.get_technique_name().value,
            "description": self.get_description(),
            "accuracy": 0.0,
            "f1_score": 0.0,
            "precision": 0.0,
            "auc_roc": None,
            "error": error_message,
            "status": "failed",
        }

    def _add_technique_specific_metrics(
        self, result: Dict[str, Any], metrics: ValidationMetrics
    ) -> None:
        """Add technique-specific metrics to result. Override in subclasses."""
        # Default implementation does nothing - intentionally unused parameters
        _ = result, metrics

    def _get_ml_techniques(self) -> List[EfficiencyValidator]:
        """Return self as the only technique."""
        return [self]

    @staticmethod
    def _prepare_dataset(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare dataset by separating features and target."""
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        return X, y

    def calculate_efficiency_from_baseline(
        self, technique_accuracy: float, baseline_accuracy: float
    ) -> Dict[str, Any]:
        """Calculate efficiency metrics based on difference from baseline (TTRR)."""
        if baseline_accuracy <= 0:
            return {
                "efficiency_score": 0.0,
                "efficiency_percentage": 0.0,
                "performance_delta": 0.0,
                "efficiency_category": "unknown",
            }

        # Calculate efficiency as difference from baseline
        efficiency_score = technique_accuracy - baseline_accuracy
        efficiency_percentage = (efficiency_score / baseline_accuracy) * 100

        # Categorize efficiency level
        if efficiency_score >= 0.05:  # 5% improvement or more
            efficiency_category = "high_efficiency"
        elif efficiency_score >= 0.01:  # 1% improvement or more
            efficiency_category = "moderate_efficiency"
        elif efficiency_score >= -0.01:  # Within 1% of baseline
            efficiency_category = "baseline_equivalent"
        elif efficiency_score >= -0.05:  # Up to 5% degradation
            efficiency_category = "low_efficiency"
        else:
            efficiency_category = "poor_efficiency"

        return {
            "efficiency_score": efficiency_score,
            "efficiency_percentage": efficiency_percentage,
            "performance_delta": efficiency_score,
            "efficiency_category": efficiency_category,
            "baseline_accuracy": baseline_accuracy,
            "technique_accuracy": technique_accuracy,
        }
