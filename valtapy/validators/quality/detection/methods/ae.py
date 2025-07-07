from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from ..detection import Detection


class AEDetection(Detection):
    """Simplified Autoencoder-like detection using statistical methods."""

    def __init__(
        self,
        original_data: Union[pd.DataFrame, np.ndarray],
        synthetic_data: Union[pd.DataFrame, np.ndarray],
        path: Optional[str] = None,
        seed: int = 42,
        threshold_percentile: float = 95.0,
    ):
        super().__init__(original_data, synthetic_data, path, seed)
        self.threshold_percentile = threshold_percentile
        self.scaler = StandardScaler()
        self.threshold = None

    def detection_model(self):
        """
        Create a simplified autoencoder-like detector using reconstruction error simulation
        """
        # Normalize the original data
        original_array = self.to_numpy(self.original_data)
        normalized_original = self.scaler.fit_transform(original_array)

        # Calculate reconstruction errors as distance from mean for each feature
        feature_means = np.mean(normalized_original, axis=0)
        feature_stds = np.std(normalized_original, axis=0)

        # Calculate reconstruction errors for training data
        reconstruction_errors = []
        for row in normalized_original:
            # Simulate reconstruction error as normalized distance from feature means
            error = np.mean(((row - feature_means) / (feature_stds + 1e-8)) ** 2)
            reconstruction_errors.append(error)

        # Set threshold based on percentile of training errors
        self.threshold = np.percentile(reconstruction_errors, self.threshold_percentile)

        # Return a simple model that can predict based on reconstruction error
        return {
            "feature_means": feature_means,
            "feature_stds": feature_stds,
            "threshold": self.threshold,
        }

    def predict(self) -> Dict[str, Any]:
        """Predict anomalies based on reconstruction error simulation"""
        if self.model is None:
            raise ValueError("Model not trained. Call detection_model first.")

        synthetic_array = self.to_numpy(self.synthetic_data)
        normalized_synthetic = self.scaler.transform(synthetic_array)

        feature_means = self.model["feature_means"]
        feature_stds = self.model["feature_stds"]
        threshold = self.model["threshold"]

        self.good_ele = []
        self.bad_ele = []

        for i, row in enumerate(normalized_synthetic):
            # Calculate reconstruction error
            error = np.mean(((row - feature_means) / (feature_stds + 1e-8)) ** 2)

            if error <= threshold:
                self.good_ele.append(i)
            else:
                self.bad_ele.append(i)

        total_elements = len(synthetic_array)
        self.anomaly_score = (
            len(self.bad_ele) / total_elements if total_elements > 0 else 0.0
        )

        return {
            "good_elements": len(self.good_ele),
            "bad_elements": len(self.bad_ele),
            "total_elements": total_elements,
            "anomaly_rate": self.anomaly_score,
            "quality_score": 1.0 - self.anomaly_score,
            "threshold": threshold,
            "avg_reconstruction_error": np.mean(
                [
                    np.mean(
                        (
                            (normalized_synthetic[i] - feature_means)
                            / (feature_stds + 1e-8)
                        )
                        ** 2
                    )
                    for i in range(len(normalized_synthetic))
                ]
            ),
        }

    def execute(self) -> Dict[str, Any]:
        """Execute the autoencoder-like detection process"""
        try:
            self.model = self.detection_model()
            results = self.predict()
            if self.path:
                self.save_results(results)

            results["method"] = "autoencoder_simulation"
            results["threshold_percentile"] = self.threshold_percentile
            results["description"] = (
                "Autoencoder-like anomaly detection using reconstruction error simulation"
            )
            return results

        except Exception as e:
            return {
                "error": str(e),
                "good_elements": 0,
                "bad_elements": 0,
                "total_elements": 0,
                "anomaly_rate": 1.0,
                "quality_score": 0.0,
                "method": "autoencoder_simulation",
            }
