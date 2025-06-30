"""
Quality detection orchestrator - coordinates anomaly detection methods
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .methods.ae import AEDetection
from .methods.isolation_forest import IsolationForestDetection
from .methods.lof import LOFDetection


class QualityDetectionOrchestrator:
    """Orchestrates detection-based quality validation methods"""

    def __init__(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        output_path: Optional[str] = None,
        seed: int = 42,
    ):
        """
        Initialize the quality detection orchestrator

        Args:
            real_data: Original real dataset
            synthetic_data: Synthetic dataset to validate
            output_path: Optional base path for saving results
            seed: Random seed for reproducibility
        """
        self.real_data = real_data
        self.synthetic_data = synthetic_data
        self.output_path = output_path
        self.seed = seed

    def run_isolation_forest(self, contamination: float = 0.1) -> Dict[str, Any]:
        """Run Isolation Forest anomaly detection"""
        path = (
            f"{self.output_path}/isolation_forest_results.txt"
            if self.output_path
            else None
        )
        detector = IsolationForestDetection(
            self.real_data, self.synthetic_data, path, self.seed, contamination
        )
        return detector.execute()

    def run_lof(
        self, n_neighbors: int = 20, contamination: float = 0.1
    ) -> Dict[str, Any]:
        """Run Local Outlier Factor detection"""
        path = f"{self.output_path}/lof_results.txt" if self.output_path else None
        detector = LOFDetection(
            self.real_data,
            self.synthetic_data,
            path,
            self.seed,
            n_neighbors,
            contamination,
        )
        return detector.execute()

    def run_autoencoder_simulation(
        self, threshold_percentile: float = 95.0
    ) -> Dict[str, Any]:
        """Run Autoencoder-like detection"""
        path = (
            f"{self.output_path}/autoencoder_results.txt" if self.output_path else None
        )
        detector = AEDetection(
            self.real_data, self.synthetic_data, path, self.seed, threshold_percentile
        )
        return detector.execute()

    def run_all_detections(
        self,
        contamination: float = 0.1,
        n_neighbors: int = 20,
        threshold_percentile: float = 95.0,
    ) -> Dict[str, Any]:
        """
        Run all detection-based quality validation methods

        Args:
            contamination: Expected proportion of outliers for IF and LOF
            n_neighbors: Number of neighbors for LOF
            threshold_percentile: Threshold percentile for autoencoder simulation

        Returns:
            Combined results from all detection methods
        """
        results = {}

        try:
            results["isolation_forest"] = self.run_isolation_forest(contamination)
        except Exception as e:
            results["isolation_forest"] = {"error": str(e)}

        try:
            results["local_outlier_factor"] = self.run_lof(n_neighbors, contamination)
        except Exception as e:
            results["local_outlier_factor"] = {"error": str(e)}

        try:
            results["autoencoder_simulation"] = self.run_autoencoder_simulation(
                threshold_percentile
            )
        except Exception as e:
            results["autoencoder_simulation"] = {"error": str(e)}

        # Calculate overall detection quality score
        quality_scores = []

        for method_name, method_results in results.items():
            if isinstance(method_results, dict) and "quality_score" in method_results:
                quality_scores.append(method_results["quality_score"])

        overall_score = np.mean(quality_scores) if quality_scores else 0.0

        # Calculate average anomaly rate
        anomaly_rates = []
        for method_name, method_results in results.items():
            if isinstance(method_results, dict) and "anomaly_rate" in method_results:
                anomaly_rates.append(method_results["anomaly_rate"])

        avg_anomaly_rate = np.mean(anomaly_rates) if anomaly_rates else 1.0

        results["overall_detection_quality_score"] = overall_score
        results["average_anomaly_rate"] = avg_anomaly_rate
        results["description"] = (
            "Detection-based quality validation results using anomaly detection methods"
        )

        return results
