"""Concrete implementations of ML contracts for efficiency metrics."""

from typing import Tuple, Any
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


class DecisionTreeModelFactory:
    """Factory for creating Decision Tree models.

    This is the default implementation using sklearn's DecisionTree.
    Can be replaced with other factories for different model types.
    """

    def __init__(self, **default_params):
        """Initialize factory with default parameters.

        Args:
            **default_params: Default parameters to apply to all created models
        """
        # Set reasonable defaults to prevent overfitting
        self.default_params = {
            'max_depth': 10,           # Limit tree depth
            'min_samples_split': 20,   # Require at least 20 samples to split
            'min_samples_leaf': 10,    # Require at least 10 samples per leaf
            **default_params           # Allow override
        }

    def create_classifier(self, random_state: int = 42, **kwargs) -> DecisionTreeClassifier:
        """Create a Decision Tree classifier.

        Args:
            random_state: Random seed for reproducibility
            **kwargs: Additional model-specific parameters

        Returns:
            Configured DecisionTreeClassifier
        """
        params = {**self.default_params, **kwargs, 'random_state': random_state}
        return DecisionTreeClassifier(**params)

    def create_regressor(self, random_state: int = 42, **kwargs) -> DecisionTreeRegressor:
        """Create a Decision Tree regressor.

        Args:
            random_state: Random seed for reproducibility
            **kwargs: Additional model-specific parameters

        Returns:
            Configured DecisionTreeRegressor
        """
        params = {**self.default_params, **kwargs, 'random_state': random_state}
        return DecisionTreeRegressor(**params)

    def supports_classification(self) -> bool:
        """Check if this factory supports classification tasks."""
        return True

    def supports_regression(self) -> bool:
        """Check if this factory supports regression tasks."""
        return True


class ClassificationMetricsCalculator:
    """Calculator for classification metrics."""

    def calculate_metrics(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray = None
    ) -> dict[str, float]:
        """Calculate classification metrics.

        Args:
            y_true: True target values
            y_pred: Predicted values
            y_pred_proba: Predicted probabilities (optional)

        Returns:
            Dictionary with accuracy, f1_score, precision, recall, and optionally auc_roc
        """
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'f1_score': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
            'precision': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
        }

        # Add AUC-ROC if probabilities are provided
        if y_pred_proba is not None:
            try:
                n_classes = len(np.unique(y_true))
                if n_classes == 2:
                    # Binary classification
                    metrics['auc_roc'] = float(roc_auc_score(y_true, y_pred_proba[:, 1]))
                else:
                    # Multiclass classification
                    metrics['auc_roc'] = float(
                        roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
                    )
            except (ValueError, IndexError):
                # AUC-ROC cannot be computed (e.g., only one class in y_true)
                metrics['auc_roc'] = None

        return metrics

    def get_primary_metric(self) -> str:
        """Get the name of the primary metric for comparison."""
        return 'accuracy'


class RegressionMetricsCalculator:
    """Calculator for regression metrics."""

    def calculate_metrics(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray = None
    ) -> dict[str, float]:
        """Calculate regression metrics.

        Args:
            y_true: True target values
            y_pred: Predicted values
            y_pred_proba: Not used for regression

        Returns:
            Dictionary with rmse, mae, and r2_score
        """
        return {
            'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'mae': float(mean_absolute_error(y_true, y_pred)),
            'r2_score': float(r2_score(y_true, y_pred))
        }

    def get_primary_metric(self) -> str:
        """Get the name of the primary metric for comparison."""
        return 'r2_score'


class SklearnDataSplitter:
    """Data splitter using sklearn's train_test_split."""

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
        # Use stratification for classification if there are enough samples per class
        stratify = None
        if self._is_classification(y):
            min_class_count = y.value_counts().min()
            if min_class_count >= 2:
                stratify = y

        return train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify
        )

    @staticmethod
    def _is_classification(y: pd.Series) -> bool:
        """Check if target is categorical (classification task)."""
        return y.dtype == 'object' or y.dtype.name == 'category' or len(y.unique()) < 20


def detect_task_type(y: pd.Series) -> str:
    """Detect if the target represents a classification or regression task.

    Args:
        y: Target vector

    Returns:
        'classification' or 'regression'
    """
    if y.dtype == 'object' or y.dtype.name == 'category':
        return 'classification'

    n_unique = len(y.unique())
    n_samples = len(y)

    # Heuristic: if unique values < 20 or < 5% of samples, likely classification
    if n_unique < 20 or (n_unique / n_samples) < 0.05:
        return 'classification'

    return 'regression'
