"""Kolmogorov-Smirnov test for fidelity measurement."""

import pandas as pd
import numpy as np
import time
from typing import Any

from ...contracts import Metric
from ...entities import MetricResult, MetricExecutionContext, MetricFamily
from ...exceptions import MetricComputationError, UnsupportedDataError


class KSTestMetric:
    """
    Kolmogorov-Smirnov test for comparing distributions between real and synthetic data.
    
    This metric measures how well the synthetic data preserves the marginal 
    distributions of the real data. Higher p-values indicate better fidelity.
    Lower KS statistics indicate more similar distributions.
    """
    
    metric_id: str = "fidelity.ks"
    family: MetricFamily = MetricFamily.FIDELITY
    name: str = "KS Test"
    description: str = "Kolmogorov-Smirnov test for distribution similarity"
    
    def compute(self, context: MetricExecutionContext) -> MetricResult:
        """Compute KS test statistics for all numeric columns."""
        start_time = time.time()
        
        try:
            real_df = context.real_data
            synth_df = context.synth_data
            
            # Get numeric columns that exist in both datasets
            numeric_columns = self._get_numeric_columns(real_df, synth_df)
            
            if not numeric_columns:
                raise UnsupportedDataError(
                    self.metric_id,
                    "No common numeric columns found for KS test"
                )
            
            # Compute KS test for each numeric column
            column_results = {}
            p_values = []
            statistics = []
            
            for col in numeric_columns:
                try:
                    result = self._compute_column_ks(real_df[col], synth_df[col])
                    column_results[col] = result
                    
                    if "p_value" in result:
                        p_values.append(result["p_value"])
                        statistics.append(result["ks_statistic"])
                        
                except Exception as e:
                    column_results[col] = {"error": str(e)}
            
            # Aggregate results
            if p_values:
                # Use geometric mean of p-values as overall score
                # Higher values indicate better fidelity (distributions are similar)
                overall_p_value = np.exp(np.mean(np.log(np.maximum(p_values, 1e-10))))
                overall_statistic = np.mean(statistics)
                overall_score = overall_p_value  # Use p-value as the main score
            else:
                overall_score = 0.0
                overall_p_value = 0.0
                overall_statistic = 1.0
            
            computation_time = time.time() - start_time
            
            return MetricResult(
                metric_id=self.metric_id,
                family=self.family,
                value=overall_score,
                details={
                    "column_results": column_results,
                    "overall_p_value": overall_p_value,
                    "overall_ks_statistic": overall_statistic,
                    "num_columns_tested": len([r for r in column_results.values() if "p_value" in r]),
                    "num_columns_failed": len([r for r in column_results.values() if "error" in r])
                },
                metadata={
                    "test_type": "two_sample_ks",
                    "significance_level": 0.05,
                    "columns_tested": list(numeric_columns)
                },
                computation_time=computation_time
            )
            
        except Exception as e:
            raise MetricComputationError(self.metric_id, str(e))
    
    def can_compute(self, real_df: pd.DataFrame, synth_df: pd.DataFrame) -> bool:
        """Check if KS test can be computed for the given datasets."""
        try:
            # Need at least one common numeric column
            numeric_columns = self._get_numeric_columns(real_df, synth_df)
            return len(numeric_columns) > 0
        except Exception:
            return False
    
    def validate_parameters(self, parameters: dict) -> bool:
        """Validate parameters for KS test."""
        # KS test doesn't require special parameters for basic implementation
        # Future parameters could include: significance_level, min_samples, etc.
        if not isinstance(parameters, dict):
            return False
        
        # Check optional parameters if present
        if "significance_level" in parameters:
            alpha = parameters["significance_level"]
            if not isinstance(alpha, (int, float)) or not (0 < alpha < 1):
                return False
        
        if "min_samples" in parameters:
            min_samples = parameters["min_samples"]
            if not isinstance(min_samples, int) or min_samples < 1:
                return False
        
        return True
    
    def _get_numeric_columns(self, real_df: pd.DataFrame, synth_df: pd.DataFrame) -> list[str]:
        """Get list of numeric columns that exist in both datasets."""
        real_numeric = real_df.select_dtypes(include=[np.number]).columns.tolist()
        synth_numeric = synth_df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Return intersection of numeric columns
        common_numeric = list(set(real_numeric) & set(synth_numeric))
        return common_numeric
    
    def _compute_column_ks(self, real_col: pd.Series, synth_col: pd.Series) -> dict[str, Any]:
        """Compute KS test for a single column."""
        # Import scipy here to make it optional
        try:
            from scipy.stats import ks_2samp
        except ImportError:
            return {"error": "scipy not available for KS test"}
        
        # Clean data
        real_clean = real_col.dropna()
        synth_clean = synth_col.dropna()
        
        # Check minimum sample sizes
        min_samples = 10  # Default minimum
        if len(real_clean) < min_samples or len(synth_clean) < min_samples:
            return {
                "error": f"Insufficient data points (real: {len(real_clean)}, synth: {len(synth_clean)})"
            }
        
        # Compute KS test
        try:
            statistic, p_value = ks_2samp(real_clean, synth_clean)
            
            return {
                "ks_statistic": float(statistic),
                "p_value": float(p_value),
                "significant": p_value < 0.05,
                "real_samples": len(real_clean),
                "synth_samples": len(synth_clean),
                "interpretation": self._interpret_ks_result(statistic, p_value)
            }
            
        except Exception as e:
            return {"error": f"KS computation failed: {str(e)}"}
    
    def _interpret_ks_result(self, statistic: float, p_value: float) -> str:
        """Provide human-readable interpretation of KS test results."""
        if p_value >= 0.05:
            return "Distributions are not significantly different (good fidelity)"
        elif p_value >= 0.01:
            return "Distributions are somewhat different (moderate fidelity)"
        else:
            return "Distributions are significantly different (poor fidelity)"