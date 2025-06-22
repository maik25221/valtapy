from typing import TYPE_CHECKING, Any, Dict

import pandas as pd

from valtapy.interfaces import IValidator, ValidationBranch, ValidationSubBranch

# Importación condicional para evitar circular imports
if TYPE_CHECKING:
    from valtapy.factory.validator_factory import ValidatorFactory


class UtilityOrchestrator(IValidator):
    """Orchestrator that manages all utility sub-validators"""

    def __init__(self):
        # Importación lazy para evitar circular imports
        from valtapy.factory.validator_factory import ValidatorFactory

        self._factory = ValidatorFactory()

    def validate(
        self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Execute all utility sub-validators"""
        results = {}

        # Get all sub-branches for Utility
        utility_subbranches = ValidationSubBranch.get_subbranches_for_branch(
            ValidationBranch.UTILITY
        )

        # Execute each sub-branch validator
        for subbranch in utility_subbranches:
            try:
                validator = self._factory.create_subbranch_validator(subbranch)
                subbranch_result = validator.validate(real_data, synthetic_data)
                results[subbranch.value] = subbranch_result
            except Exception as e:
                results[subbranch.value] = {"error": str(e), "status": "failed"}

        # Calculate overall utility score
        overall_score = self._calculate_overall_utility_score(results)
        results["overall_utility_score"] = overall_score

        return results

    def get_branch_name(self) -> ValidationBranch:
        return ValidationBranch.UTILITY

    def _calculate_overall_utility_score(self, results: Dict[str, Any]) -> float:
        """Calculate overall utility score from all sub-branch results"""
        scores = []

        for key, value in results.items():
            if isinstance(value, dict):
                # Try to get overall score from sub-branch
                if "overall_utility_score" in value:
                    scores.append(value["overall_utility_score"])
                elif "efficiency_score" in value:
                    scores.append(value["efficiency_score"])
                elif "score" in value:
                    scores.append(value["score"])

        return sum(scores) / len(scores) if scores else 0.0
