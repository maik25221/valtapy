from typing import Any, Dict, List

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from valtapy.validators.utility.efficiency import EfficiencyValidator


class TTRRValidator(EfficiencyValidator):
    """Train and Test on Real - Baseline validation using only real data"""

    def __init__(self, random_seed: int = 42, test_size: float = 0.2):
        self.random_seed = random_seed
        self.test_size = test_size

    def validate_ml_technique(
        self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        TTRR: Train and test exclusively on real data
        Serves as baseline for comparison with synthetic data techniques
        """
        try:
            # Prepare real data (assuming last column is target)
            X_real = real_data.iloc[:, :-1]
            y_real = real_data.iloc[:, -1]

            # Split real data: 80% train, 20% test
            X_train, X_test, y_train, y_test = train_test_split(
                X_real,
                y_real,
                test_size=self.test_size,
                random_state=self.random_seed,
                stratify=y_real if len(y_real.unique()) > 1 else None,
            )

            # Train CatBoost model on real data
            model = CatBoostClassifier(
                random_seed=self.random_seed, verbose=False, eval_metric="F1"
            )
            model.fit(X_train, y_train)

            # Test on real data
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted")
            precision = precision_score(y_test, y_pred, average="weighted")

            # AUC-ROC (only for binary/multiclass with probabilities)
            try:
                if len(y_real.unique()) == 2:
                    auc_roc = roc_auc_score(y_test, y_pred_proba[:, 1])
                else:
                    auc_roc = roc_auc_score(
                        y_test, y_pred_proba, multi_class="ovr", average="weighted"
                    )
            except:
                auc_roc = None

            return {
                "technique": "TTRR",
                "description": "Train and Test on Real (Baseline)",
                "accuracy": accuracy,
                "f1_score": f1,
                "precision": precision,
                "auc_roc": auc_roc,
                "train_size": len(X_train),
                "test_size": len(X_test),
                "status": "success",
            }

        except Exception as e:
            return {
                "technique": "TTRR",
                "description": "Train and Test on Real (Baseline)",
                "accuracy": 0.0,
                "f1_score": 0.0,
                "precision": 0.0,
                "auc_roc": None,
                "error": str(e),
                "status": "failed",
            }

    def _get_ml_techniques(self) -> List[EfficiencyValidator]:
        return [self]
