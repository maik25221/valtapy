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
        pass

    @abstractmethod
    def _get_ml_techniques(self) -> List["EfficiencyValidator"]:
        """Get list of ML techniques to execute"""
        pass

    def _calculate_efficiency_score(self, results: Dict[str, Any]) -> float:
        """Calculate overall efficiency score from ML technique results"""
        scores = []
        for key, value in results.items():
            if isinstance(value, dict) and "accuracy" in value:
                scores.append(value["accuracy"])

        return sum(scores) / len(scores) if scores else 0.0
