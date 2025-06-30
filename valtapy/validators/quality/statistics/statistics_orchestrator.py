"""
Quality statistics orchestrator - coordinates statistical analysis methods
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .methods.correlation import Correlation
from .methods.metrics import Metrics
from .methods.tests import Tests


class QualityStatisticsOrchestrator:
    """Orchestrates statistics-based quality validation methods"""

    def __init__(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        output_path: Optional[str] = None,
        seed: int = 42,
    ):
        """
        Initialize the quality statistics orchestrator

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

    def run_statistical_tests(self, all_data: bool = True) -> Dict[str, Any]:
        """Run statistical tests (KS test and KL divergence)"""
        path = (
            f"{self.output_path}/statistical_tests_results.txt"
            if self.output_path
            else None
        )
        tests = Tests(self.real_data, self.synthetic_data, path, self.seed, all_data)
        return tests.execute()

    def run_correlation_analysis(self) -> Dict[str, Any]:
        """Run correlation analysis"""
        path = f"{self.output_path}/correlation" if self.output_path else None
        correlation = Correlation(self.real_data, self.synthetic_data, path, self.seed)
        return correlation.execute()

    def run_statistical_metrics(self) -> Dict[str, Any]:
        """Run statistical metrics analysis"""
        path = f"{self.output_path}/metrics" if self.output_path else None
        metrics = Metrics(self.real_data, self.synthetic_data, path, self.seed)
        return metrics.execute()

    def run_all_statistics(self, all_data: bool = True) -> Dict[str, Any]:
        """
        Run all statistics-based quality validation methods

        Args:
            all_data: Whether to analyze all data together or column by column

        Returns:
            Combined results from all statistical methods
        """
        results = {}

        try:
            results["statistical_tests"] = self.run_statistical_tests(all_data)
        except Exception as e:
            results["statistical_tests"] = {"error": str(e)}

        try:
            results["correlation_analysis"] = self.run_correlation_analysis()
        except Exception as e:
            results["correlation_analysis"] = {"error": str(e)}

        try:
            results["statistical_metrics"] = self.run_statistical_metrics()
        except Exception as e:
            results["statistical_metrics"] = {"error": str(e)}

        # Calculate overall statistics quality score
        quality_scores = []

        # From statistical tests
        if "statistical_tests" in results and isinstance(
            results["statistical_tests"], dict
        ):
            tests_result = results["statistical_tests"]
            if "overall_statistical_quality" in tests_result:
                quality_scores.append(tests_result["overall_statistical_quality"])

        # From correlation analysis
        if "correlation_analysis" in results and isinstance(
            results["correlation_analysis"], dict
        ):
            corr_result = results["correlation_analysis"]
            if "correlation_quality_score" in corr_result:
                quality_scores.append(corr_result["correlation_quality_score"])

        # From statistical metrics
        if "statistical_metrics" in results and isinstance(
            results["statistical_metrics"], dict
        ):
            metrics_result = results["statistical_metrics"]
            if "overall_statistical_quality" in metrics_result:
                quality_scores.append(metrics_result["overall_statistical_quality"])

        overall_score = np.mean(quality_scores) if quality_scores else 0.0

        results["overall_statistics_quality_score"] = overall_score
        results["description"] = (
            "Statistics-based quality validation results using distribution and correlation analysis"
        )

        return results
