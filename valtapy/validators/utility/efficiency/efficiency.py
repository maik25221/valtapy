from abc import ABC, abstractmethod
from typing import Any, Dict, List

import pandas as pd


class EfficiencyValidator(ABC):
    """Abstract base class for machine learning efficiency validation techniques"""

    def validate_utility(
        self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Validate utility through efficiency metrics"""
        results = {}

        # Get all ML efficiency techniques
        ml_techniques = self._get_ml_techniques()

        for technique in ml_techniques:
            technique_name = technique.__class__.__name__
            technique_results = technique.validate_ml_technique(
                real_data, synthetic_data
            )
            results[technique_name] = technique_results

        # Calculate overall efficiency score
        overall_score = self._calculate_efficiency_score(results)
        results["efficiency_score"] = overall_score

        return results

    @abstractmethod
    def validate_ml_technique(
        self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Abstract method for specific ML technique validation"""
        raise NotImplementedError

    @abstractmethod
    def _get_ml_techniques(self) -> List["EfficiencyValidator"]:
        """Get list of ML techniques to execute"""
        raise NotImplementedError

    def _calculate_efficiency_score(self, results: Dict[str, Any]) -> float:
        """Calculate overall efficiency score as difference from baseline (TTRR)"""
        try:
            # Find baseline (TTRR) accuracy
            baseline_acc = 0.0
            for _, value in results.items():
                if isinstance(value, dict) and "technique" in value:
                    if value["technique"] == "TTRR" and "accuracy" in value:
                        baseline_acc = value["accuracy"]
                        break

            if baseline_acc <= 0:
                return 0.0

            # Calculate efficiency scores as differences from baseline
            efficiency_scores = []
            technique_weights = {"TTSS": 0.3, "TSTR": 0.4, "TRTS": 0.3}

            for _, value in results.items():
                if (
                    isinstance(value, dict)
                    and "technique" in value
                    and "accuracy" in value
                ):
                    technique = value["technique"]
                    if technique in technique_weights:
                        technique_acc = value["accuracy"]
                        efficiency = technique_acc - baseline_acc
                        weight = technique_weights[technique]
                        efficiency_scores.append(efficiency * weight)

            return sum(efficiency_scores) if efficiency_scores else 0.0

        except (ValueError, TypeError, ZeroDivisionError):
            return 0.0
