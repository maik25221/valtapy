"""Predictive Mean Squared Error for utility measurement."""

import pandas as pd
import numpy as np
from typing import Self, Dict, Any

from ...registry import register
from ...base import MetricBase
from ....domain.entities import MetricResult


@register("utility.pmse")
class PMSEMetric(MetricBase):
    """
    Predictive Mean Squared Error (PMSE) for utility measurement.
    
    This metric trains models on synthetic data and tests on real data (TSTR),
    comparing performance to training and testing on real data (TRTR).
    Higher utility scores indicate synthetic data enables better predictive models.
    
    TODO: This is a functional stub. Full implementation would:
    - Use multiple model types (linear, tree-based, neural networks)
    - Implement proper cross-validation
    - Handle both regression and classification targets
    - Add feature selection and preprocessing
    - Use cached train/test splits from StatsStore
    """
    
    name: str = "pmse"
    family: str = "utility"
    purpose_tags: set[str] = {"utility", "predictive_performance"}
    
    def __init__(self):
        super().__init__()
    
    def fit(self, real_data: pd.DataFrame, synth_data: pd.DataFrame, context: Dict[str, Any]) -> Self:
        """Fit PMSE metric to data."""
        self._setup(real_data, synth_data, context)
        return self
    
    def compute(self) -> MetricResult:
        """Compute predictive mean squared error utility metric."""
        try:
            dataset_spec = self._context.get("dataset_spec")
            target_col = None

            if dataset_spec and hasattr(dataset_spec, 'target'):
                target_col = dataset_spec.target

            if not target_col or target_col not in self._real_data.columns:
                return MetricResult(
                    id="utility.pmse",
                    value=0.0,
                    details={"error": "Target column not specified or not found"},
                    family="utility",
                    purpose_tags=self.purpose_tags
                )

            feature_cols = [col for col in self._real_data.columns if col != target_col]
            numeric_features = [col for col in feature_cols if pd.api.types.is_numeric_dtype(self._real_data[col])]
            
            if not numeric_features:
                return MetricResult(
                    id="utility.pmse",
                    value=0.0,
                    details={"error": "No numeric feature columns found"},
                    family="utility",
                    purpose_tags=self.purpose_tags
                )
            
            X_real = self._real_data[numeric_features].fillna(0)
            y_real = self._real_data[target_col]
            X_synth = self._synth_data[numeric_features].fillna(0)
            y_synth = self._synth_data[target_col] if target_col in self._synth_data.columns else None
            
            if y_synth is None:
                return MetricResult(
                    id="utility.pmse",
                    value=0.0,
                    details={"error": "Target column not found in synthetic data"},
                    family="utility",
                    purpose_tags=self.purpose_tags
                )
            
            is_regression = pd.api.types.is_numeric_dtype(y_real) and y_real.nunique() > 10
            
            try:
                splits = self._get_train_test_splits(n_splits=3, random_state=42)

                tstr_scores = []
                trtr_scores = []

                for train_real, test_real in splits:
                    train_indices = train_real.index
                    test_indices = test_real.index

                    X_train_real = train_real[numeric_features].fillna(0)
                    y_train_real = train_real[target_col]
                    X_test_real = test_real[numeric_features].fillna(0)
                    y_test_real = test_real[target_col]

                    n_train = len(X_train_real)
                    X_train_synth = X_synth.iloc[:n_train]
                    y_train_synth = y_synth.iloc[:n_train]

                    if is_regression:
                        from sklearn.linear_model import LinearRegression
                        from sklearn.metrics import mean_squared_error

                        model_tstr = LinearRegression()
                        model_tstr.fit(X_train_synth, y_train_synth)
                        y_pred_tstr = model_tstr.predict(X_test_real)
                        mse_tstr = mean_squared_error(y_test_real, y_pred_tstr)

                        model_trtr = LinearRegression()
                        model_trtr.fit(X_train_real, y_train_real)
                        y_pred_trtr = model_trtr.predict(X_test_real)
                        mse_trtr = mean_squared_error(y_test_real, y_pred_trtr)

                        tstr_scores.append(mse_tstr)
                        trtr_scores.append(mse_trtr)

                    else:
                        from sklearn.linear_model import LogisticRegression
                        from sklearn.metrics import accuracy_score

                        model_tstr = LogisticRegression(max_iter=1000)
                        model_tstr.fit(X_train_synth, y_train_synth)
                        y_pred_tstr = model_tstr.predict(X_test_real)
                        acc_tstr = accuracy_score(y_test_real, y_pred_tstr)

                        model_trtr = LogisticRegression(max_iter=1000)
                        model_trtr.fit(X_train_real, y_train_real)
                        y_pred_trtr = model_trtr.predict(X_test_real)
                        acc_trtr = accuracy_score(y_test_real, y_pred_trtr)

                        tstr_scores.append(1.0 - acc_tstr)
                        trtr_scores.append(1.0 - acc_trtr)

                mean_tstr = np.mean(tstr_scores)
                mean_trtr = np.mean(trtr_scores)

                if is_regression:
                    utility_score = max(0.0, 1.0 - self._safe_divide(mean_tstr, mean_trtr, default=2.0))
                else:
                    utility_score = max(0.0, 1.0 - self._safe_divide(mean_tstr, mean_trtr, default=2.0))

                details = {
                    "task_type": "regression" if is_regression else "classification",
                    "tstr_scores": [float(s) for s in tstr_scores],
                    "trtr_scores": [float(s) for s in trtr_scores],
                    "mean_tstr_score": float(mean_tstr),
                    "mean_trtr_score": float(mean_trtr),
                    "n_features": len(numeric_features),
                    "n_splits": len(splits),
                    "metric_type": "mse" if is_regression else "accuracy_error"
                }
                
            except ImportError:
                utility_score = 0.5
                details = {
                    "error": "sklearn not available, using fallback score",
                    "fallback_score": 0.5,
                    "task_type": "regression" if is_regression else "classification"
                }
            
            return MetricResult(
                id="utility.pmse",
                value=float(utility_score),
                details=details,
                family="utility",
                purpose_tags=self.purpose_tags
            )
            
        except Exception as e:
            return MetricResult(
                id="utility.pmse",
                value=0.0,
                details={"error": f"PMSE computation failed: {str(e)}"},
                family="utility",
                purpose_tags=self.purpose_tags
            )