from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ..base_quality import Quality


class Detection(Quality):
    """Clase base para detectar anomalías."""

    def __init__(
        self,
        original_data: Union[pd.DataFrame, np.ndarray],
        synthetic_data: Union[pd.DataFrame, np.ndarray],
        path: Optional[str] = None,
        seed: int = 42,
    ):
        super().__init__(original_data, synthetic_data, path, seed)
        self.good_ele = []
        self.bad_ele = []
        self.model = None
        self.anomaly_score = 0.0

    def detection_model(self):
        """Override this method to implement specific detection model"""
        raise NotImplementedError

    def predict(self) -> Dict[str, Any]:
        """Predict anomalies in synthetic data"""
        synthetic_array = self.to_numpy(self.synthetic_data)

        predictions = []
        for i, row in enumerate(synthetic_array):
            try:
                prediction = self.model.predict([row])
                if hasattr(prediction, "__iter__") and len(prediction) > 0:
                    pred_value = (
                        prediction[0]
                        if hasattr(prediction[0], "__iter__")
                        else prediction[0]
                    )
                else:
                    pred_value = prediction

                if pred_value == 1 or pred_value > 0:  # Normal/good element
                    self.good_ele.append(i)
                    predictions.append(1)
                else:  # Anomaly/bad element
                    self.bad_ele.append(i)
                    predictions.append(-1)
            except Exception as e:
                # If prediction fails, assume it's an anomaly
                self.bad_ele.append(i)
                predictions.append(-1)

        total_elements = len(synthetic_array)
        self.anomaly_score = (
            len(self.bad_ele) / total_elements if total_elements > 0 else 0.0
        )

        return {
            "good_elements": len(self.good_ele),
            "bad_elements": len(self.bad_ele),
            "total_elements": total_elements,
            "anomaly_rate": self.anomaly_score,
            "quality_score": 1.0 - self.anomaly_score,  # Higher is better
        }

    def save_results(self, results: Dict[str, Any]):
        """Save the results in a file"""
        if self.path:
            with open(self.path, "w") as file:
                file.write(f"Good elements: {results['good_elements']}\n")
                file.write(f"Bad elements: {results['bad_elements']}\n")
                file.write(f"Total elements: {results['total_elements']}\n")
                file.write(f"Anomaly rate: {results['anomaly_rate']:.4f}\n")
                file.write(f"Quality score: {results['quality_score']:.4f}\n")

    def execute(self) -> Dict[str, Any]:
        """Execute the detection process"""
        try:
            self.model = self.detection_model()
            results = self.predict()
            if self.path:
                self.save_results(results)
            return results
        except Exception as e:
            return {
                "error": str(e),
                "good_elements": 0,
                "bad_elements": 0,
                "total_elements": 0,
                "anomaly_rate": 1.0,
                "quality_score": 0.0,
            }
