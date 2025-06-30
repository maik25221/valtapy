from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from valtapy.interfaces import IValidator, ValidationBranch

from .extraction.extraction import PrivacyExtractionOrchestrator
from .inference.inference import PrivacyInferenceOrchestrator


class PrivacyValidator(IValidator):
    """Privacy validation implementation with extraction and inference methods"""

    def __init__(
        self,
        output_path: Optional[str] = None,
        epsilon: float = 1.0,
        delta: float = 1e-5,
    ):
        """
        Initialize privacy validator

        Args:
            output_path: Base path for saving detailed results
            epsilon: Privacy parameter for differential privacy
            delta: Privacy parameter for differential privacy
        """
        self.output_path = output_path
        self.epsilon = epsilon
        self.delta = delta

    def validate(
        self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Execute comprehensive privacy validation

        Args:
            real_data: Original real dataset
            synthetic_data: Synthetic dataset to validate

        Returns:
            Comprehensive privacy validation results
        """
        results = {
            "privacy_validator": "comprehensive",
            "extraction_methods": {},
            "inference_methods": {},
            "overall_privacy_score": 0.0,
        }

        # Run extraction-based privacy methods
        try:
            extraction_orchestrator = PrivacyExtractionOrchestrator(
                real_data,
                synthetic_data,
                f"{self.output_path}/extraction" if self.output_path else None,
            )
            results["extraction_methods"] = extraction_orchestrator.run_all_extractions(
                self.epsilon, self.delta
            )
        except Exception as e:
            results["extraction_methods"] = {
                "error": f"Extraction methods failed: {str(e)}"
            }

        # Run inference-based privacy methods
        try:
            inference_orchestrator = PrivacyInferenceOrchestrator(
                real_data,
                synthetic_data,
                f"{self.output_path}/inference" if self.output_path else None,
            )
            results["inference_methods"] = inference_orchestrator.run_all_inferences()
        except Exception as e:
            results["inference_methods"] = {
                "error": f"Inference methods failed: {str(e)}"
            }

        # Calculate overall privacy score
        scores = []

        if "overall_extraction_privacy_score" in results["extraction_methods"]:
            scores.append(
                results["extraction_methods"]["overall_extraction_privacy_score"]
            )

        if "overall_inference_privacy_score" in results["inference_methods"]:
            scores.append(
                results["inference_methods"]["overall_inference_privacy_score"]
            )

        if scores:
            results["overall_privacy_score"] = np.mean(scores)
        else:
            results["overall_privacy_score"] = 0.0

        # Add summary metrics for compatibility
        results["privacy_score"] = results["overall_privacy_score"]
        results["k_anonymity"] = self._estimate_k_anonymity(real_data, synthetic_data)
        results["differential_privacy"] = (
            results["extraction_methods"]
            .get("differential_privacy", {})
            .get("differential_privacy", {})
            .get("privacy_score", 0.0)
        )

        return results

    def _estimate_k_anonymity(
        self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame
    ) -> bool:
        """
        Estimate k-anonymity by checking for unique combinations

        Returns:
            True if estimated k-anonymity is satisfied, False otherwise
        """
        try:
            # Simple heuristic: check if synthetic data has fewer unique combinations than real data
            real_unique = len(real_data.drop_duplicates())
            synthetic_unique = len(synthetic_data.drop_duplicates())

            # If synthetic data has significantly fewer unique records, it likely has some k-anonymity
            return synthetic_unique < real_unique * 0.8
        except Exception:
            return False

    def get_branch_name(self) -> ValidationBranch:
        return ValidationBranch.PRIVACY
