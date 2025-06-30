"""
Privacy inference orchestrator - coordinates inference-based privacy attacks
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .methods.identity_attribute_disclosure import IdentityAttributeDisclosureInference
from .methods.membership_inference_attack import MembershipInferenceAttack


class PrivacyInferenceOrchestrator:
    """Orchestrates inference-based privacy validation methods"""

    def __init__(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        output_path: Optional[str] = None,
    ):
        """
        Initialize the privacy inference orchestrator

        Args:
            real_data: Original real dataset
            synthetic_data: Synthetic dataset to validate
            output_path: Optional base path for saving results
        """
        self.real_data = real_data
        self.synthetic_data = synthetic_data
        self.output_path = output_path

    def run_identity_attribute_disclosure_inference(self) -> Dict[str, Any]:
        """Run inference-based identity and attribute disclosure analysis"""
        path = (
            f"{self.output_path}/identity_attribute_disclosure_inference_results.txt"
            if self.output_path
            else None
        )
        iad_inference = IdentityAttributeDisclosureInference(
            self.real_data, self.synthetic_data, path
        )
        return iad_inference.execute()

    def run_membership_inference_attack(self) -> Dict[str, Any]:
        """Run membership inference attack"""
        path = (
            f"{self.output_path}/membership_inference_attack_results.txt"
            if self.output_path
            else None
        )
        mia = MembershipInferenceAttack(self.real_data, self.synthetic_data, path)
        return mia.execute()

    def run_all_inferences(self) -> Dict[str, Any]:
        """
        Run all inference-based privacy validation methods

        Returns:
            Combined results from all inference methods
        """
        results = {}

        try:
            results["identity_attribute_disclosure_inference"] = (
                self.run_identity_attribute_disclosure_inference()
            )
        except Exception as e:
            results["identity_attribute_disclosure_inference"] = {"error": str(e)}

        try:
            results["membership_inference_attack"] = (
                self.run_membership_inference_attack()
            )
        except Exception as e:
            results["membership_inference_attack"] = {"error": str(e)}

        # Calculate overall inference privacy score
        scores = []

        if "identity_attribute_disclosure_inference" in results:
            iad_results = results["identity_attribute_disclosure_inference"]
            if "identity_disclosure_rate" in iad_results:
                # Lower disclosure rate is better for privacy
                identity_privacy_score = 1.0 - iad_results["identity_disclosure_rate"]
                scores.append(identity_privacy_score)
            if "attribute_disclosure_rate" in iad_results:
                # Lower disclosure rate is better for privacy
                attribute_privacy_score = 1.0 - iad_results["attribute_disclosure_rate"]
                scores.append(attribute_privacy_score)

        if "membership_inference_attack" in results:
            mia_results = results["membership_inference_attack"]
            if "membership_inference_accuracy" in mia_results:
                # Lower attack accuracy is better for privacy
                # Random guessing would be 0.5, so we normalize around that
                mia_accuracy = mia_results["membership_inference_accuracy"]
                privacy_score = max(0.0, 1.0 - (mia_accuracy - 0.5) / 0.5)
                scores.append(privacy_score)

        overall_score = np.mean(scores) if scores else 0.0

        results["overall_inference_privacy_score"] = overall_score
        results["description"] = "Inference-based privacy validation results"

        return results
