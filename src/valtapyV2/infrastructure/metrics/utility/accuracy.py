"""Classification accuracy for utility measurement."""

import pandas as pd
import numpy as np
from typing import Self, Dict, Any

from ..registry import register
from ..base import MetricBase
from ....domain.entities import MetricResult


@register("utility.accuracy")
class AccuracyMetric(MetricBase):
    """
    Classification accuracy utility metric.
    
    Specifically designed for classification tasks, this metric trains classifiers
    on synthetic data and evaluates them on real data, comparing performance
    to real-data-trained models.
    
    This metric reuses cached train/test splits and features from StatsStore
    for efficiency across multiple utility metrics.
    """
    
    name: str = "accuracy"
    family: str = "utility"
    purpose_tags: set[str] = {"utility", "classification_performance"}
    
    def __init__(self):
        super().__init__()
    
    def fit(self, real_data: pd.DataFrame, synth_data: pd.DataFrame, context: Dict[str, Any]) -> Self:
        """Fit accuracy metric to data."""
        self._setup(real_data, synth_data, context)
        return self
    
    def compute(self) -> MetricResult:
        """Compute classification accuracy utility metric."""
        try:
            dataset_spec = self._context.get("dataset_spec")
            target_col = None

            if dataset_spec and hasattr(dataset_spec, 'target'):
                target_col = dataset_spec.target

            if not target_col or target_col not in self._real_data.columns:
                return MetricResult(
                    id="utility.accuracy",
                    value=0.0,
                    details={"error": "Target column not specified or not found"},
                    family="utility",
                    purpose_tags=self.purpose_tags
                )

            y_real = self._real_data[target_col]
            n_classes = y_real.nunique()
            
            if n_classes > min(50, len(y_real) // 10):
                return MetricResult(
                    id="utility.accuracy",
                    value=0.0,
                    details={"error": f"Target has too many unique values ({n_classes}) for classification"},
                    family="utility",
                    purpose_tags=self.purpose_tags
                )
            
            if n_classes < 2:
                return MetricResult(
                    id="utility.accuracy",
                    value=0.0,
                    details={"error": f"Target has only {n_classes} unique values, need at least 2 for classification"},
                    family="utility",
                    purpose_tags=self.purpose_tags
                )
            

            feature_cols = [col for col in self._real_data.columns if col != target_col]
            numeric_features = [col for col in feature_cols if pd.api.types.is_numeric_dtype(self._real_data[col])]
            
            if not numeric_features:
                return MetricResult(
                    id="utility.accuracy",
                    value=0.0,
                    details={"error": "No numeric feature columns found"},
                    family="utility",
                    purpose_tags=self.purpose_tags
                )
            
            X_real = self._real_data[numeric_features].fillna(0)
            X_synth = self._synth_data[numeric_features].fillna(0)
            y_synth = self._synth_data[target_col] if target_col in self._synth_data.columns else None
            
            if y_synth is None:
                return MetricResult(
                    id="utility.accuracy",
                    value=0.0,
                    details={"error": "Target column not found in synthetic data"},
                    family="utility",
                    purpose_tags=self.purpose_tags
                )
            
            try:
                splits = self._get_train_test_splits(n_splits=3, random_state=42)

                tstr_accuracies = []
                trtr_accuracies = []

                for train_real, test_real in splits:
                    X_train_real = train_real[numeric_features].fillna(0)
                    y_train_real = train_real[target_col]
                    X_test_real = test_real[numeric_features].fillna(0)
                    y_test_real = test_real[target_col]

                    n_train = len(X_train_real)
                    X_train_synth = X_synth.iloc[:n_train]
                    y_train_synth = y_synth.iloc[:n_train]

                    if len(X_test_real) < 5:
                        continue

                    train_real_classes = set(y_train_real.unique())
                    train_synth_classes = set(y_train_synth.unique())
                    test_classes = set(y_test_real.unique())

                    if len(train_real_classes) < 2 or len(train_synth_classes) < 2:
                        continue

                    from sklearn.linear_model import LogisticRegression
                    from sklearn.metrics import accuracy_score

                    try:
                        model_tstr = LogisticRegression(max_iter=1000, random_state=42)
                        model_tstr.fit(X_train_synth, y_train_synth)
                        y_pred_tstr = model_tstr.predict(X_test_real)
                        acc_tstr = accuracy_score(y_test_real, y_pred_tstr)
                        tstr_accuracies.append(acc_tstr)
                    except:
                        continue

                    try:
                        model_trtr = LogisticRegression(max_iter=1000, random_state=42)
                        model_trtr.fit(X_train_real, y_train_real)
                        y_pred_trtr = model_trtr.predict(X_test_real)
                        acc_trtr = accuracy_score(y_test_real, y_pred_trtr)
                        trtr_accuracies.append(acc_trtr)
                    except:
                        if tstr_accuracies:
                            tstr_accuracies.pop()
                        continue

                if not tstr_accuracies or not trtr_accuracies:
                    return MetricResult(
                        id="utility.accuracy",
                        value=0.0,
                        details={"error": "Could not complete any cross-validation folds"},
                        family="utility",
                        purpose_tags=self.purpose_tags
                    )

                mean_tstr = np.mean(tstr_accuracies)
                mean_trtr = np.mean(trtr_accuracies)

                if mean_trtr > 0:
                    utility_score = mean_tstr / mean_trtr
                else:
                    utility_score = 0.0

                utility_score = min(1.0, max(0.0, utility_score))

                details = {
                    "tstr_accuracies": [float(a) for a in tstr_accuracies],
                    "trtr_accuracies": [float(a) for a in trtr_accuracies],
                    "mean_tstr_accuracy": float(mean_tstr),
                    "mean_trtr_accuracy": float(mean_trtr),
                    "n_classes": int(n_classes),
                    "n_features": len(numeric_features),
                    "n_successful_folds": len(tstr_accuracies),
                    "class_distribution": y_real.value_counts().to_dict()
                }
                
            except ImportError:
                utility_score = 0.5
                details = {
                    "error": "sklearn not available, using fallback score",
                    "fallback_score": 0.5,
                    "n_classes": int(n_classes)
                }
            
            return MetricResult(
                id="utility.accuracy",
                value=float(utility_score),
                details=details,
                family="utility",
                purpose_tags=self.purpose_tags
            )
            
        except Exception as e:
            return MetricResult(
                id="utility.accuracy",
                value=0.0,
                details={"error": f"Accuracy computation failed: {str(e)}"},
                family="utility",
                purpose_tags=self.purpose_tags
            )