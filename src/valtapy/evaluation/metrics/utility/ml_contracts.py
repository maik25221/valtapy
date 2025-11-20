"""Contracts for ML efficiency metrics with flexible model injection."""

from typing import Protocol, Any, Tuple
from abc import abstractmethod
import pandas as pd
import numpy as np


class MLModel(Protocol):
    """Protocol for machine learning models used in efficiency evaluation.

    This allows injection of any sklearn-compatible model (DecisionTree,
    RandomForest, LogisticRegression, etc.)
    """

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'MLModel':
        """Train the model on the given data.

        Args:
            X: Feature matrix
            y: Target vector

        Returns:
            Self for method chaining
        """
        ...

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on the given data.

        Args:
            X: Feature matrix

        Returns:
            Predicted values
        """
        ...

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities (for classifiers).

        Args:
            X: Feature matrix

        Returns:
            Probability estimates for each class
        """
        ...


class MLModelFactory(Protocol):
    """Protocol for creating ML models.

    This allows different factories to provide different model types
    (DecisionTree, RandomForest, etc.) while keeping the metric code decoupled.
    """

    def create_classifier(self, random_state: int = 42, **kwargs) -> Any:
        """Create a classifier model.

        Args:
            random_state: Random seed for reproducibility
            **kwargs: Additional model-specific parameters

        Returns:
            A classifier implementing the MLModel protocol
        """
        ...

    def create_regressor(self, random_state: int = 42, **kwargs) -> Any:
        """Create a regressor model.

        Args:
            random_state: Random seed for reproducibility
            **kwargs: Additional model-specific parameters

        Returns:
            A regressor implementing the MLModel protocol
        """
        ...

    def supports_classification(self) -> bool:
        """Check if this factory supports classification tasks."""
        ...

    def supports_regression(self) -> bool:
        """Check if this factory supports regression tasks."""
        ...


class MetricsCalculator(Protocol):
    """Protocol for calculating performance metrics from predictions.

    This allows different metric calculation strategies for different task types.
    """

    def calculate_metrics(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray = None
    ) -> dict[str, float]:
        """Calculate performance metrics.

        Args:
            y_true: True target values
            y_pred: Predicted values
            y_pred_proba: Predicted probabilities (optional, for classification)

        Returns:
            Dictionary of metric names to values
        """
        ...

    def get_primary_metric(self) -> str:
        """Get the name of the primary metric for comparison."""
        ...


class DataSplitter(Protocol):
    """Protocol for splitting data into train/test sets."""

    def split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float,
        random_state: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into train and test sets.

        Args:
            X: Feature matrix
            y: Target vector
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        ...
