from typing import Any, Dict, List

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from valtapy.validators.utility.efficiency import EfficiencyValidator


class TSTRValidator(EfficiencyValidator):
    """Train on Synthetic, Test on Real - Generalization capability validation"""

    def __init__(self, random_seed: int = 42, test_size: float = 0.2):
        self.random_seed = random_seed
        self.test_size = test_size

    def validate_ml_technique(
        self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        TSTR: Train on synthetic data, test on real data
        Measures generalization capability of synthetic data to real world
        """
        try:
            # Prepare data
            X_synthetic = synthetic_data.iloc[:, :-1]
            y_synthetic = synthetic_data.iloc[:, -1]
            X_real = real_data.iloc[:, :-1]
            y_real = real_data.iloc[:, -1]

            # Use synthetic data for training
            X_train = X_synthetic
            y_train = y_synthetic

            # Split real data for testing (or use all real data for testing)
            _, X_test, _, y_test = train_test_split(
                X_real,
                y_real,
                test_size=self.test_size,
                random_state=self.random_seed,
                stratify=y_real if len(y_real.unique()) > 1 else None,
            )

            # Train CatBoost model on synthetic data
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

            # AUC-ROC
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
                "technique": "TSTR",
                "description": "Train on Synthetic, Test on Real (Generalization)",
                "accuracy": accuracy,
                "f1_score": f1,
                "precision": precision,
                "auc_roc": auc_roc,
                "train_size": len(X_train),
                "test_size": len(X_test),
                "generalization_capability": (
                    "high" if accuracy > 0.7 else "medium" if accuracy > 0.5 else "low"
                ),
                "status": "success",
            }

        except Exception as e:
            return {
                "technique": "TSTR",
                "description": "Train on Synthetic, Test on Real (Generalization)",
                "accuracy": 0.0,
                "f1_score": 0.0,
                "precision": 0.0,
                "auc_roc": None,
                "error": str(e),
                "status": "failed",
            }

    def _get_ml_techniques(self) -> List[EfficiencyValidator]:
        return [self]
