"""
Privacy extraction orchestrator - coordinates extraction-based privacy attacks
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .methods.dcr import DCR
from .methods.differential_privacy import DifferentialPrivacy
from .methods.identity_attribute_disclosure import IdentityAttributeDisclosure


class PrivacyExtractionOrchestrator:
    """Orchestrates extraction-based privacy validation methods"""

    def __init__(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        output_path: Optional[str] = None,
    ):
        """
        Initialize the privacy extraction orchestrator

        Args:
            real_data: Original real dataset
            synthetic_data: Synthetic dataset to validate
            output_path: Optional base path for saving results
        """
        self.real_data = real_data
        self.synthetic_data = synthetic_data
        self.output_path = output_path

    def run_dcr(self) -> Dict[str, Any]:
        """Run Disclosure Control Ratio analysis"""
        path = f"{self.output_path}/dcr_results.txt" if self.output_path else None
        dcr = DCR(self.real_data, self.synthetic_data, path)
        return dcr.execute()

    def run_differential_privacy(
        self, epsilon: float = 1.0, delta: float = 1e-5
    ) -> Dict[str, Any]:
        """Run differential privacy analysis"""
        path = (
            f"{self.output_path}/differential_privacy_results.txt"
            if self.output_path
            else None
        )
        dp = DifferentialPrivacy(
            self.real_data, self.synthetic_data, path, epsilon, delta
        )
        return dp.execute()

    def run_identity_attribute_disclosure(self) -> Dict[str, Any]:
        """Run identity and attribute disclosure analysis"""
        path = (
            f"{self.output_path}/identity_attribute_disclosure_results.txt"
            if self.output_path
            else None
        )
        iad = IdentityAttributeDisclosure(self.real_data, self.synthetic_data, path)
        return iad.execute()

    def run_all_extractions(
        self, epsilon: float = 1.0, delta: float = 1e-5
    ) -> Dict[str, Any]:
        """
        Run all extraction-based privacy validation methods

        Args:
            epsilon: Privacy parameter for differential privacy
            delta: Privacy parameter for differential privacy

        Returns:
            Combined results from all extraction methods
        """
        results = {}

        try:
            results["dcr"] = self.run_dcr()
        except Exception as e:
            results["dcr"] = {"error": str(e)}

        try:
            results["differential_privacy"] = self.run_differential_privacy(
                epsilon, delta
            )
        except Exception as e:
            results["differential_privacy"] = {"error": str(e)}

        try:
            results["identity_attribute_disclosure"] = (
                self.run_identity_attribute_disclosure()
            )
        except Exception as e:
            results["identity_attribute_disclosure"] = {"error": str(e)}

        # Calculate overall extraction privacy score
        scores = []
        if "dcr" in results and "dcr" in results["dcr"]:
            # DCR closer to 1 is better (good frequency preservation)
            dcr_score = min(results["dcr"]["dcr"], 1.0)
            scores.append(dcr_score)

        if (
            "differential_privacy" in results
            and "differential_privacy" in results["differential_privacy"]
        ):
            dp_results = results["differential_privacy"]["differential_privacy"]
            if "privacy_score" in dp_results:
                scores.append(dp_results["privacy_score"])

        if "identity_attribute_disclosure" in results:
            iad_results = results["identity_attribute_disclosure"]
            if "identity_disclosure_rate" in iad_results:
                # Lower disclosure rate is better for privacy
                identity_privacy_score = 1.0 - iad_results["identity_disclosure_rate"]
                scores.append(identity_privacy_score)

        overall_score = np.mean(scores) if scores else 0.0

        results["overall_extraction_privacy_score"] = overall_score
        results["description"] = "Extraction-based privacy validation results"

        return results
