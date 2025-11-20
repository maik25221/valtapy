"""Kolmogorov-Smirnov test for fidelity measurement."""

import pandas as pd
from typing import Self, Dict, Any
from scipy.stats import ks_2samp
import numpy as np

from ..registry import register
from ..base import MetricBase
from ....domain.entities import MetricResult


@register("fidelity.ks")
class KSTestMetric(MetricBase):
    """
    Kolmogorov-Smirnov test for comparing distributions between real and synthetic data.
    
    This metric measures how well the synthetic data preserves the marginal 
    distributions of the real data. Higher p-values indicate better fidelity.
    """
    
    name: str = "ks_test"
    family: str = "fidelity"
    purpose_tags: set[str] = {"fidelity", "distribution_matching"}
    
    def __init__(self):
        super().__init__()
        self._results = {}
    
    def fit(self, real_data: pd.DataFrame, synth_data: pd.DataFrame, context: Dict[str, Any]) -> Self:
        """Fit KS test metric to data."""
        self._setup(real_data, synth_data, context)
        return self
    
    def compute(self) -> MetricResult:
        """Compute KS test statistics for all numeric columns."""
        try:
            numeric_columns = self._get_numeric_columns()

            if not numeric_columns:
                return MetricResult(
                    id="fidelity.ks",
                    value=0.0,
                    details={"error": "No numeric columns found for KS test"},
                    family="fidelity",
                    purpose_tags=self.purpose_tags
                )

            column_results = {}
            p_values = []
            statistics = []

            for col in numeric_columns:
                try:
                    real_col = self._real_data[col].dropna()
                    synth_col = self._synth_data[col].dropna()

                    if len(real_col) < 10 or len(synth_col) < 10:
                        column_results[col] = {
                            "error": f"Insufficient data points (real: {len(real_col)}, synth: {len(synth_col)})"
                        }
                        continue

                    statistic, p_value = ks_2samp(real_col, synth_col)

                    column_results[col] = {
                        "ks_statistic": float(statistic),
                        "p_value": float(p_value),
                        "significant": p_value < 0.05,
                        "real_samples": len(real_col),
                        "synth_samples": len(synth_col)
                    }

                    p_values.append(p_value)
                    statistics.append(statistic)

                except Exception as e:
                    column_results[col] = {"error": str(e)}

            if p_values:
                overall_p_value = np.exp(np.mean(np.log(np.maximum(p_values, 1e-10))))
                mean_statistic = np.mean(statistics)
                fidelity_score = overall_p_value
            else:
                overall_p_value = 0.0
                mean_statistic = 1.0
                fidelity_score = 0.0
            
            details = {
                "column_results": column_results,
                "overall_p_value": float(overall_p_value),
                "mean_ks_statistic": float(mean_statistic),
                "n_columns_tested": len(p_values),
                "n_columns_failed": len([r for r in column_results.values() if "error" in r])
            }
            
            return MetricResult(
                id="fidelity.ks",
                value=float(fidelity_score),
                details=details,
                family="fidelity",
                purpose_tags=self.purpose_tags
            )
            
        except Exception as e:
            return MetricResult(
                id="fidelity.ks",
                value=0.0,
                details={"error": f"KS test computation failed: {str(e)}"},
                family="fidelity",
                purpose_tags=self.purpose_tags
            )