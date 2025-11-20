"""
Membership Inference Attack (MIA) metric for privacy assessment.

This metric uses a classification-based approach to determine if synthetic data
reveals information about whether specific records were in the training data.
"""

import pandas as pd
import numpy as np
import time
from typing import Any, Optional, Literal

from ...contracts import Metric
from ...entities import MetricResult, MetricExecutionContext, MetricFamily
from ...exceptions import MetricComputationError, UnsupportedDataError


class MembershipInferenceMetric:
    """
    Membership Inference Attack (MIA) metric using classifier-based approach.

    This metric trains a binary classifier to distinguish between real and synthetic
    data. The classifier's performance (accuracy, AUC) indicates privacy risk:
    - High accuracy (close to 1.0) = High privacy risk (easy to distinguish)
    - Low accuracy (close to 0.5) = Low privacy risk (hard to distinguish)

    The attack model:
    1. Label real data as 1 (member) and synthetic data as 0 (non-member)
    2. Train a classifier to predict the label
    3. Evaluate classifier performance on held-out data
    4. High performance = privacy leak (synthetic data too similar to specific real records)

    Privacy score = 1 - accuracy (higher is better privacy)
    """

    metric_id: str = "privacy.mia"
    family: MetricFamily = MetricFamily.PRIVACY
    name: str = "Membership Inference Attack"
    description: str = "Classifier-based membership inference attack to assess privacy risk"

    def compute(self, context: MetricExecutionContext) -> MetricResult:
        """Compute MIA privacy metric."""
        start_time = time.time()

        try:
            real_df = context.real_data
            synth_df = context.synth_data

            # Get parameters
            params = context.parameters
            classifier_type = params.get('classifier', 'random_forest')
            test_size = params.get('test_size', 0.3)
            n_splits = params.get('n_splits', 5)
            random_seed = params.get('random_seed', context.random_seed)

            # Validate data
            if real_df.empty or synth_df.empty:
                raise UnsupportedDataError(
                    self.metric_id,
                    "Real or synthetic data is empty"
                )

            # Get numeric columns only
            numeric_cols = self._get_numeric_columns(real_df, synth_df)

            if not numeric_cols:
                raise UnsupportedDataError(
                    self.metric_id,
                    "No common numeric columns found for MIA"
                )

            # Prepare data for classification
            X_real = real_df[numeric_cols].copy()
            X_synth = synth_df[numeric_cols].copy()

            # Handle missing values
            X_real = X_real.fillna(X_real.mean())
            X_synth = X_synth.fillna(X_synth.mean())

            # Combine and create labels
            # Real data = 1 (member), Synthetic data = 0 (non-member)
            X = pd.concat([X_real, X_synth], axis=0, ignore_index=True)
            y = np.concatenate([
                np.ones(len(X_real)),   # Real data labeled as 1
                np.zeros(len(X_synth))  # Synthetic data labeled as 0
            ])

            # Perform attack evaluation
            attack_results = self._perform_attack(
                X, y,
                classifier_type=classifier_type,
                test_size=test_size,
                n_splits=n_splits,
                random_seed=random_seed
            )

            # Privacy score: 1 - accuracy
            # Higher score = better privacy (harder to distinguish)
            privacy_score = 1.0 - attack_results['mean_accuracy']

            computation_time = time.time() - start_time

            return MetricResult(
                metric_id=self.metric_id,
                family=self.family,
                value=privacy_score,
                details={
                    "attack_accuracy": attack_results['mean_accuracy'],
                    "attack_accuracy_std": attack_results['std_accuracy'],
                    "attack_precision": attack_results['mean_precision'],
                    "attack_recall": attack_results['mean_recall'],
                    "attack_f1": attack_results['mean_f1'],
                    "attack_auc": attack_results.get('mean_auc', None),
                    "privacy_score": privacy_score,
                    "num_features": len(numeric_cols),
                    "num_real_samples": len(X_real),
                    "num_synth_samples": len(X_synth),
                    "classifier_type": classifier_type,
                    "n_splits": n_splits,
                    "interpretation": self._interpret_results(privacy_score, attack_results['mean_accuracy'])
                },
                metadata={
                    "attack_type": "classification",
                    "classifier": classifier_type,
                    "test_size": test_size,
                    "n_splits": n_splits,
                    "features_used": numeric_cols,
                    "baseline_accuracy": 0.5  # Random guessing baseline
                },
                computation_time=computation_time
            )

        except Exception as e:
            raise MetricComputationError(self.metric_id, str(e))

    def can_compute(self, real_df: pd.DataFrame, synth_df: pd.DataFrame) -> bool:
        """Check if MIA can be computed for the given datasets."""
        try:
            # Need at least one common numeric column
            numeric_columns = self._get_numeric_columns(real_df, synth_df)

            # Need sufficient samples for train/test split
            min_samples = 20  # Minimum for meaningful evaluation

            return (
                len(numeric_columns) > 0 and
                len(real_df) >= min_samples and
                len(synth_df) >= min_samples
            )
        except Exception:
            return False

    def validate_parameters(self, parameters: dict) -> bool:
        """Validate parameters for MIA."""
        if not isinstance(parameters, dict):
            return False

        # Validate classifier type
        if "classifier" in parameters:
            valid_classifiers = ['random_forest', 'logistic_regression', 'mlp', 'gradient_boosting']
            if parameters['classifier'] not in valid_classifiers:
                return False

        # Validate test_size
        if "test_size" in parameters:
            test_size = parameters['test_size']
            if not isinstance(test_size, (int, float)) or not (0 < test_size < 1):
                return False

        # Validate n_splits
        if "n_splits" in parameters:
            n_splits = parameters['n_splits']
            if not isinstance(n_splits, int) or n_splits < 2:
                return False

        return True

    def _get_numeric_columns(self, real_df: pd.DataFrame, synth_df: pd.DataFrame) -> list[str]:
        """Get list of numeric columns that exist in both datasets."""
        real_numeric = real_df.select_dtypes(include=[np.number]).columns.tolist()
        synth_numeric = synth_df.select_dtypes(include=[np.number]).columns.tolist()

        # Return intersection of numeric columns
        common_numeric = list(set(real_numeric) & set(synth_numeric))
        return common_numeric

    def _perform_attack(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        classifier_type: str = 'random_forest',
        test_size: float = 0.3,
        n_splits: int = 5,
        random_seed: int = 42
    ) -> dict[str, float]:
        """
        Perform membership inference attack using cross-validation.

        Returns attack performance metrics.
        """
        try:
            from sklearn.model_selection import cross_validate, StratifiedKFold
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.neural_network import MLPClassifier
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline
        except ImportError:
            raise ImportError(
                "scikit-learn is required for MIA metric. "
                "Install with: pip install scikit-learn"
            )

        # Select classifier
        if classifier_type == 'random_forest':
            base_classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=random_seed,
                n_jobs=-1
            )
        elif classifier_type == 'logistic_regression':
            base_classifier = LogisticRegression(
                max_iter=1000,
                random_state=random_seed
            )
        elif classifier_type == 'mlp':
            base_classifier = MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                random_state=random_seed
            )
        elif classifier_type == 'gradient_boosting':
            base_classifier = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=random_seed
            )
        else:
            base_classifier = RandomForestClassifier(
                n_estimators=100,
                random_state=random_seed
            )

        # Create pipeline with scaling
        classifier = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', base_classifier)
        ])

        # Cross-validation setup
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

        # Scoring metrics
        scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

        # Perform cross-validation
        cv_results = cross_validate(
            classifier, X, y,
            cv=cv,
            scoring=scoring,
            return_train_score=False,
            n_jobs=-1
        )

        # Aggregate results
        results = {
            'mean_accuracy': np.mean(cv_results['test_accuracy']),
            'std_accuracy': np.std(cv_results['test_accuracy']),
            'mean_precision': np.mean(cv_results['test_precision']),
            'std_precision': np.std(cv_results['test_precision']),
            'mean_recall': np.mean(cv_results['test_recall']),
            'std_recall': np.std(cv_results['test_recall']),
            'mean_f1': np.mean(cv_results['test_f1']),
            'std_f1': np.std(cv_results['test_f1']),
            'mean_auc': np.mean(cv_results['test_roc_auc']),
            'std_auc': np.std(cv_results['test_roc_auc'])
        }

        return results

    def _interpret_results(self, privacy_score: float, attack_accuracy: float) -> str:
        """Provide human-readable interpretation of MIA results."""
        if privacy_score >= 0.4:  # Attack accuracy <= 0.6
            return (
                f"GOOD PRIVACY: Attack accuracy ({attack_accuracy:.3f}) is close to random guessing. "
                "Synthetic data does not reveal membership information."
            )
        elif privacy_score >= 0.2:  # Attack accuracy 0.6-0.8
            return (
                f"MODERATE PRIVACY RISK: Attack accuracy ({attack_accuracy:.3f}) is moderately high. "
                "Some membership information may be leaked."
            )
        else:  # Attack accuracy > 0.8
            return (
                f"HIGH PRIVACY RISK: Attack accuracy ({attack_accuracy:.3f}) is very high. "
                "Synthetic data reveals significant membership information."
            )
