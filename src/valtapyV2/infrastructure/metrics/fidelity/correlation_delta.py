"""Correlation matrix comparison for fidelity measurement."""

import pandas as pd
import numpy as np
from typing import Self, Dict, Any

from ...registry import register
from ...base import MetricBase
from ....domain.entities import MetricResult


@register("fidelity.correlation_delta")
class CorrelationDeltaMetric(MetricBase):
    """
    Measures fidelity by comparing correlation matrices between real and synthetic data.
    
    This metric computes the difference in pairwise correlations and provides
    an overall similarity score. Lower correlation differences indicate better fidelity.
    """
    
    name: str = "correlation_delta"
    family: str = "fidelity" 
    purpose_tags: set[str] = {"fidelity", "correlation_preservation"}
    
    def __init__(self):
        super().__init__()
    
    def fit(self, real_data: pd.DataFrame, synth_data: pd.DataFrame, context: Dict[str, Any]) -> Self:
        """Fit correlation delta metric to data."""
        self._setup(real_data, synth_data, context)
        return self
    
    def compute(self) -> MetricResult:
        """Compute correlation matrix differences."""
        try:
            # Get cached correlation matrices
            real_corr = self._get_correlation_matrix("real")
            synth_corr = self._get_correlation_matrix("synth")
            
            if real_corr.empty or synth_corr.empty:
                return MetricResult(
                    id="fidelity.correlation_delta",
                    value=0.0,
                    details={"error": "No numeric columns found for correlation analysis"},
                    family="fidelity",
                    purpose_tags=self.purpose_tags
                )
            
            # Ensure same columns in both matrices
            common_cols = real_corr.columns.intersection(synth_corr.columns)
            if len(common_cols) < 2:
                return MetricResult(
                    id="fidelity.correlation_delta", 
                    value=0.0,
                    details={"error": "Need at least 2 common numeric columns"},
                    family="fidelity",
                    purpose_tags=self.purpose_tags
                )
            
            real_corr = real_corr.loc[common_cols, common_cols]
            synth_corr = synth_corr.loc[common_cols, common_cols]
            
            # Calculate correlation differences
            corr_diff = real_corr - synth_corr
            
            # Extract upper triangle (excluding diagonal) to avoid double counting
            mask = np.triu(np.ones_like(corr_diff, dtype=bool), k=1)
            upper_triangle_diff = corr_diff.values[mask]
            
            # Calculate metrics
            mean_abs_diff = np.mean(np.abs(upper_triangle_diff))
            max_abs_diff = np.max(np.abs(upper_triangle_diff))
            rmse = np.sqrt(np.mean(upper_triangle_diff**2))
            
            # Calculate correlation between correlation matrices
            real_upper = real_corr.values[mask]
            synth_upper = synth_corr.values[mask]
            
            if len(real_upper) > 1:
                correlation_similarity = np.corrcoef(real_upper, synth_upper)[0, 1]
                if np.isnan(correlation_similarity):
                    correlation_similarity = 0.0
            else:
                correlation_similarity = 1.0
            
            # Convert to fidelity score (1 - normalized error)
            # Use mean absolute difference as primary metric
            fidelity_score = max(0.0, 1.0 - mean_abs_diff)
            
            details = {
                "mean_absolute_difference": float(mean_abs_diff),
                "max_absolute_difference": float(max_abs_diff),
                "rmse": float(rmse),
                "correlation_similarity": float(correlation_similarity),
                "n_correlations_compared": len(upper_triangle_diff),
                "n_variables": len(common_cols),
                "common_variables": list(common_cols)
            }
            
            return MetricResult(
                id="fidelity.correlation_delta",
                value=float(fidelity_score),
                details=details,
                family="fidelity",
                purpose_tags=self.purpose_tags
            )
            
        except Exception as e:
            return MetricResult(
                id="fidelity.correlation_delta",
                value=0.0,
                details={"error": f"Correlation delta computation failed: {str(e)}"},
                family="fidelity",
                purpose_tags=self.purpose_tags
            )