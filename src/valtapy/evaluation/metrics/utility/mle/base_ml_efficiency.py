"""Base class for ML efficiency metrics (TTR, TTS, TRTS, TSTR, TTRS)."""

from abc import abstractmethod
from typing import Tuple, Optional, Any
import time
import pandas as pd
import numpy as np

from ....contracts import Metric
from ....entities import MetricResult, MetricExecutionContext, MetricFamily
from ....exceptions import MetricComputationError
from ..ml_contracts import MLModelFactory, MetricsCalculator, DataSplitter
from ..ml_implementations import (
    DecisionTreeModelFactory,
    ClassificationMetricsCalculator,
    RegressionMetricsCalculator,
    SklearnDataSplitter,
    detect_task_type
)


class BaseMLEfficiencyMetric:
    """Base class for ML efficiency evaluation metrics.

    This class implements the Template Method pattern to standardize
    the evaluation process while allowing each technique (TTR, TTS, etc.)
    to customize the data preparation strategy.

    The metric uses dependency injection to allow flexible model selection.
    """

    def __init__(
        self,
        metric_id: str,
        name: str,
        description: str,
        model_factory: Optional[MLModelFactory] = None,
        classification_metrics: Optional[MetricsCalculator] = None,
        regression_metrics: Optional[MetricsCalculator] = None,
        data_splitter: Optional[DataSplitter] = None,
        test_size: float = 0.2,
        random_seed: int = 42
    ):
        """Initialize the ML efficiency metric.

        Args:
            metric_id: Unique identifier for the metric
            name: Human-readable name
            description: Description of what this metric evaluates
            model_factory: Factory for creating ML models (defaults to DecisionTree)
            classification_metrics: Calculator for classification metrics
            regression_metrics: Calculator for regression metrics
            data_splitter: Splitter for train/test data
            test_size: Proportion of data to use for testing (0.0-1.0)
            random_seed: Random seed for reproducibility
        """
        self.metric_id = metric_id
        self.family = MetricFamily.UTILITY
        self.name = name
        self.description = description

        # Dependency injection with sensible defaults
        self.model_factory = model_factory or DecisionTreeModelFactory()
        self.classification_metrics = classification_metrics or ClassificationMetricsCalculator()
        self.regression_metrics = regression_metrics or RegressionMetricsCalculator()
        self.data_splitter = data_splitter or SklearnDataSplitter()

        self.test_size = test_size
        self.random_seed = random_seed

    def compute(self, context: MetricExecutionContext) -> MetricResult:
        """Compute the ML efficiency metric.

        This is the Template Method that orchestrates the evaluation process.

        Args:
            context: Execution context with real data, synthetic data, and parameters

        Returns:
            MetricResult with computed efficiency score and detailed metrics

        Raises:
            MetricComputationError: If computation fails
        """
        start_time = time.time()

        try:
            # Extract data from context
            real_df = context.real_data
            synth_df = context.synth_data

            # Validate input data
            self._validate_data(real_df, synth_df)

            # Override parameters from context if provided
            test_size = context.parameters.get('test_size', self.test_size)
            random_seed = context.parameters.get('random_seed', self.random_seed)

            # Separate features and target (assume last column is target)
            X_real, y_real = self._prepare_dataset(real_df)
            X_synth, y_synth = self._prepare_dataset(synth_df)

            # Encode categorical variables if present
            X_real, X_synth = self._encode_categorical_features(X_real, X_synth)

            # Detect task type (classification or regression)
            task_type = detect_task_type(y_real)

            # Prepare train and test data according to the specific technique
            train_X, train_y, test_X, test_y = self._prepare_train_test_data(
                X_real, y_real, X_synth, y_synth, test_size, random_seed
            )

            # Train model
            model = self._train_model(train_X, train_y, task_type, random_seed)

            # Make predictions
            y_pred, y_pred_proba = self._make_predictions(model, test_X, task_type)

            # Calculate metrics
            metrics_calculator = (
                self.classification_metrics if task_type == 'classification'
                else self.regression_metrics
            )
            performance_metrics = metrics_calculator.calculate_metrics(
                test_y, y_pred, y_pred_proba
            )

            # Get primary metric value
            primary_metric = metrics_calculator.get_primary_metric()
            primary_value = performance_metrics[primary_metric]

            # Build detailed results
            details = {
                'technique': self.metric_id,
                'task_type': task_type,
                'primary_metric': primary_metric,
                **performance_metrics,
                'train_samples': len(train_y),
                'test_samples': len(test_y),
                'n_features': train_X.shape[1]
            }

            # Add technique-specific details
            self._add_technique_specific_details(details, performance_metrics)

            computation_time = time.time() - start_time

            return MetricResult(
                metric_id=self.metric_id,
                family=self.family,
                value=primary_value,
                details=details,
                metadata={
                    'test_size': test_size,
                    'random_seed': random_seed,
                    'model_type': type(model).__name__
                },
                computation_time=computation_time
            )

        except Exception as e:
            raise MetricComputationError(
                metric_id=self.metric_id,
                reason=str(e)
            ) from e

    @abstractmethod
    def _prepare_train_test_data(
        self,
        X_real: pd.DataFrame,
        y_real: pd.Series,
        X_synth: pd.DataFrame,
        y_synth: pd.Series,
        test_size: float,
        random_seed: int
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Prepare training and testing data according to the specific technique.

        This is the Strategy pattern - each subclass implements its own data preparation.

        Args:
            X_real: Real features
            y_real: Real target
            X_synth: Synthetic features
            y_synth: Synthetic target
            test_size: Proportion for test split
            random_seed: Random seed

        Returns:
            Tuple of (train_X, train_y, test_X, test_y)
        """
        raise NotImplementedError

    def can_compute(
        self,
        real_df: pd.DataFrame,
        synth_df: pd.DataFrame
    ) -> bool:
        """Check if this metric can be computed for the given datasets.

        Args:
            real_df: Real dataset
            synth_df: Synthetic dataset

        Returns:
            True if metric can be computed, False otherwise
        """
        try:
            self._validate_data(real_df, synth_df)
            return True
        except Exception:
            return False

    def validate_parameters(self, parameters: dict) -> bool:
        """Validate that parameters are correct for this metric.

        Args:
            parameters: Dictionary of parameters

        Returns:
            True if parameters are valid, False otherwise
        """
        if 'test_size' in parameters:
            test_size = parameters['test_size']
            if not isinstance(test_size, (int, float)) or not 0 < test_size < 1:
                return False

        if 'random_seed' in parameters:
            random_seed = parameters['random_seed']
            if not isinstance(random_seed, int) or random_seed < 0:
                return False

        return True

    def _validate_data(self, real_df: pd.DataFrame, synth_df: pd.DataFrame) -> None:
        """Validate input data.

        Args:
            real_df: Real dataset
            synth_df: Synthetic dataset

        Raises:
            ValueError: If data is invalid
        """
        if real_df.empty:
            raise ValueError("Real data cannot be empty")
        if synth_df.empty:
            raise ValueError("Synthetic data cannot be empty")
        if real_df.shape[1] != synth_df.shape[1]:
            raise ValueError(
                f"Real and synthetic data must have the same number of columns. "
                f"Got {real_df.shape[1]} and {synth_df.shape[1]}"
            )
        if real_df.shape[1] < 2:
            raise ValueError("Data must have at least 2 columns (features + target)")
        if len(real_df) < 10:
            raise ValueError("Real data must have at least 10 samples")
        if len(synth_df) < 10:
            raise ValueError("Synthetic data must have at least 10 samples")

    @staticmethod
    def _prepare_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Separate features and target from dataset.

        Assumes the last column is the target variable.

        Args:
            df: Complete dataset with features and target

        Returns:
            Tuple of (features, target)
        """
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        return X, y

    @staticmethod
    def _encode_categorical_features(
        X_real: pd.DataFrame,
        X_synth: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Encode categorical variables using label encoding.

        This ensures sklearn models can handle non-numeric data.
        The same encoding is applied to both real and synthetic data
        based on the combined categories from both datasets.

        Args:
            X_real: Real features
            X_synth: Synthetic features

        Returns:
            Tuple of (encoded_X_real, encoded_X_synth)
        """
        from sklearn.preprocessing import LabelEncoder

        X_real_encoded = X_real.copy()
        X_synth_encoded = X_synth.copy()

        # Find columns with object/categorical dtypes
        categorical_cols = X_real.select_dtypes(include=['object', 'category']).columns

        for col in categorical_cols:
            # Combine categories from both datasets
            combined_categories = pd.concat([
                X_real[col].astype(str),
                X_synth[col].astype(str)
            ]).unique()

            # Create and fit encoder on combined categories
            encoder = LabelEncoder()
            encoder.fit(combined_categories)

            # Transform both datasets
            X_real_encoded[col] = encoder.transform(X_real[col].astype(str))
            X_synth_encoded[col] = encoder.transform(X_synth[col].astype(str))

        return X_real_encoded, X_synth_encoded

    def _train_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        task_type: str,
        random_seed: int
    ) -> Any:
        """Train the ML model.

        Args:
            X_train: Training features
            y_train: Training target
            task_type: 'classification' or 'regression'
            random_seed: Random seed

        Returns:
            Trained model

        Raises:
            ValueError: If model training fails
        """
        if task_type == 'classification':
            model = self.model_factory.create_classifier(random_state=random_seed)
        else:
            model = self.model_factory.create_regressor(random_state=random_seed)

        try:
            model.fit(X_train, y_train)
            return model
        except Exception as e:
            raise ValueError(f"Failed to train model: {str(e)}") from e

    def _make_predictions(
        self,
        model: Any,
        X_test: pd.DataFrame,
        task_type: str
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Make predictions using the trained model.

        Args:
            model: Trained model
            X_test: Test features
            task_type: 'classification' or 'regression'

        Returns:
            Tuple of (predictions, probabilities)
            probabilities is None for regression
        """
        y_pred = model.predict(X_test)

        y_pred_proba = None
        if task_type == 'classification' and hasattr(model, 'predict_proba'):
            try:
                y_pred_proba = model.predict_proba(X_test)
            except Exception:
                # Some models might not support predict_proba
                pass

        return y_pred, y_pred_proba

    def _add_technique_specific_details(
        self,
        details: dict,
        performance_metrics: dict
    ) -> None:
        """Add technique-specific details to results.

        Subclasses can override this to add custom information.

        Args:
            details: Dictionary to add details to (modified in-place)
            performance_metrics: Computed performance metrics
        """
        pass  # Default: no additional details
