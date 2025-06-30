from typing import Any, Dict, List

import pandas as pd

from valtapy.interfaces import IValidator, ValidationBranch
from valtapy.validators.utility.efficiency import EfficiencyValidator
from valtapy.validators.utility.efficiency.ml_effiency.methods import (
    TRTSValidator,
    TSTRValidator,
    TTRRValidator,
    TTSSValidator,
)


class MLEfficiencyValidator(EfficiencyValidator, IValidator):
    """
    Concrete implementation that runs all ML efficiency techniques for utility validation.

    Implements the four standard techniques:
    - TTRR: Train and Test on Real (Baseline) - serves as reference point
    - TTSS: Train and Test on Synthetic (Internal Consistency)
    - TSTR: Train on Synthetic, Test on Real (Generalization)
    - TRTS: Train on Real, Test on Synthetic (Fidelity)

    EFFICIENCY CALCULATION METHODOLOGY:
    Efficiency is calculated as the difference between each technique's accuracy
    and the baseline (TTRR) accuracy:

    Efficiency = Technique_Accuracy - TTRR_Accuracy

    - Positive values: Synthetic data performs better than real data baseline
    - Negative values: Synthetic data performs worse than real data baseline
    - Values near zero: Synthetic data performs equivalently to real data

    This approach provides a direct measure of how much utility is gained or lost
    when using synthetic data instead of real data for ML tasks.
    """

    def __init__(self, random_seed: int = 42, test_size: float = 0.2):
        self.random_seed = random_seed
        self.test_size = test_size
        self._techniques = [
            TTRRValidator(random_seed, test_size),
            TTSSValidator(random_seed, test_size),
            TSTRValidator(random_seed, test_size),
            TRTSValidator(random_seed, test_size),
        ]

    def validate(
        self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Main validate method required by IValidator interface"""
        return self.validate_utility(real_data, synthetic_data)

    def get_branch_name(self) -> ValidationBranch:
        """Return the branch this validator belongs to"""
        return ValidationBranch.UTILITY

    def validate_ml_technique(
        self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Run all ML efficiency techniques and provide comprehensive analysis"""
        results = {}

        # Execute all techniques
        for technique in self._techniques:
            technique_result = technique.validate_ml_technique(
                real_data, synthetic_data
            )
            technique_name = technique_result.get(
                "technique", technique.__class__.__name__
            )
            results[technique_name] = technique_result

        # Calculate comparative metrics
        results["comparative_analysis"] = self._calculate_comparative_analysis(results)

        # Calculate overall utility score
        overall_score = self._calculate_ml_efficiency_score(results)
        results["overall_utility_score"] = overall_score

        return results

    def _get_ml_techniques(self) -> List[EfficiencyValidator]:
        return self._techniques

    def _calculate_comparative_analysis(
        self, results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate comparative analysis between techniques using TTRR as baseline"""
        try:
            ttrr_acc = results.get("TTRR", {}).get("accuracy", 0)
            ttss_acc = results.get("TTSS", {}).get("accuracy", 0)
            tstr_acc = results.get("TSTR", {}).get("accuracy", 0)
            trts_acc = results.get("TRTS", {}).get("accuracy", 0)

            # Calculate efficiency as difference from baseline (TTRR)
            return {
                "baseline_performance": ttrr_acc,
                "ttss_efficiency": ttss_acc - ttrr_acc if ttrr_acc > 0 else None,
                "tstr_efficiency": tstr_acc - ttrr_acc if ttrr_acc > 0 else None,
                "trts_efficiency": trts_acc - ttrr_acc if ttrr_acc > 0 else None,
                # Keep legacy metrics for compatibility
                "synthetic_consistency": ttss_acc,
                "generalization_gap": (
                    abs(ttrr_acc - tstr_acc) if ttrr_acc > 0 else None
                ),
                "fidelity_gap": abs(ttrr_acc - trts_acc) if ttrr_acc > 0 else None,
                "utility_preservation": (tstr_acc / ttrr_acc) if ttrr_acc > 0 else 0,
                "synthetic_quality": (trts_acc / ttrr_acc) if ttrr_acc > 0 else 0,
            }
        except (ValueError, TypeError, KeyError) as e:
            return {"error": str(e)}

    def _calculate_ml_efficiency_score(self, results: Dict[str, Any]) -> float:
        """Calculate overall ML efficiency score based on differences from baseline (TTRR)"""
        try:
            ttrr_acc = results.get("TTRR", {}).get("accuracy", 0)

            if ttrr_acc <= 0:
                return 0.0

            # Calculate efficiency scores as differences from baseline
            efficiency_scores = []
            technique_weights = {"TTSS": 0.3, "TSTR": 0.4, "TRTS": 0.3}

            for technique, weight in technique_weights.items():
                if technique in results and "accuracy" in results[technique]:
                    technique_acc = results[technique]["accuracy"]
                    efficiency = technique_acc - ttrr_acc  # Difference from baseline
                    efficiency_scores.append(efficiency * weight)

            # Overall efficiency is weighted average of individual efficiencies
            overall_efficiency = sum(efficiency_scores) if efficiency_scores else 0.0

            return overall_efficiency

        except (ValueError, TypeError, ZeroDivisionError):
            return 0.0
