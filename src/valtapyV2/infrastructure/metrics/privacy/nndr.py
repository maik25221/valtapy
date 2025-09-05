"""Nearest Neighbor Distance Ratio for privacy measurement."""

import pandas as pd
import numpy as np
from typing import Self, Dict, Any

from ...registry import register
from ...base import MetricBase
from ....domain.entities import MetricResult


@register("privacy.nndr")
class NNDRMetric(MetricBase):
    """
    Nearest Neighbor Distance Ratio (NNDR) for privacy measurement.
    
    This metric measures privacy by computing the ratio of distances from synthetic
    points to their nearest neighbors in the real data versus distances to second
    nearest neighbors. Lower ratios indicate potential privacy leakage.
    
    The metric uses cached KNN computations from StatsStore for efficiency.
    """
    
    name: str = "nndr"
    family: str = "privacy"
    purpose_tags: set[str] = {"privacy", "distance_based"}
    
    def __init__(self):
        super().__init__()
    
    def fit(self, real_data: pd.DataFrame, synth_data: pd.DataFrame, context: Dict[str, Any]) -> Self:
        """Fit NNDR metric to data."""
        self._setup(real_data, synth_data, context)
        return self
    
    def compute(self) -> MetricResult:
        """Compute nearest neighbor distance ratio."""
        try:
            # Get cached KNN distances (k=5 for NNDR computation)
            knn_data = self._get_knn_distances(k=5)
            
            if "error" in knn_data:
                return MetricResult(
                    id="privacy.nndr",
                    value=0.0,
                    details={"error": knn_data["error"]},
                    family="privacy",
                    purpose_tags=self.purpose_tags
                )
            
            distances = knn_data["distances"]
            
            if distances.shape[1] < 2:
                return MetricResult(
                    id="privacy.nndr",
                    value=0.0,
                    details={"error": "Need at least k=2 for NNDR computation"},
                    family="privacy",
                    purpose_tags=self.purpose_tags
                )
            
            first_nn_dist = distances[:, 0]
            second_nn_dist = distances[:, 1]

            valid_mask = second_nn_dist > 1e-10
            
            if not np.any(valid_mask):
                privacy_score = 0.0
                details = {"error": "All second nearest neighbor distances are zero"}
            else:
                ratios = np.full(len(first_nn_dist), 1.0)
                ratios[valid_mask] = first_nn_dist[valid_mask] / second_nn_dist[valid_mask]

                mean_ratio = np.mean(ratios)
                median_ratio = np.median(ratios)
                min_ratio = np.min(ratios)

                low_ratio_threshold = 0.1
                n_low_ratios = np.sum(ratios < low_ratio_threshold)
                fraction_low_ratios = n_low_ratios / len(ratios)

                privacy_score = median_ratio * (1.0 - fraction_low_ratios * 0.5)
                privacy_score = min(1.0, max(0.0, privacy_score))

                details = {
                    "mean_nndr": float(mean_ratio),
                    "median_nndr": float(median_ratio),
                    "min_nndr": float(min_ratio),
                    "max_nndr": float(np.max(ratios)),
                    "std_nndr": float(np.std(ratios)),
                    "n_synthetic_points": len(ratios),
                    "n_valid_ratios": int(np.sum(valid_mask)),
                    "n_low_ratios": int(n_low_ratios),
                    "fraction_low_ratios": float(fraction_low_ratios),
                    "low_ratio_threshold": low_ratio_threshold,
                    "k_neighbors": knn_data["k"]
                }

                percentiles = [10, 25, 75, 90]
                for p in percentiles:
                    details[f"nndr_p{p}"] = float(np.percentile(ratios, p))
            
            return MetricResult(
                id="privacy.nndr",
                value=float(privacy_score),
                details=details,
                family="privacy",
                purpose_tags=self.purpose_tags
            )
            
        except Exception as e:
            return MetricResult(
                id="privacy.nndr",
                value=0.0,
                details={"error": f"NNDR computation failed: {str(e)}"},
                family="privacy",
                purpose_tags=self.purpose_tags
            )