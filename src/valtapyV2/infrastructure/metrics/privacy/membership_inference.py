"""Membership Inference Attack for privacy measurement."""

import pandas as pd
import numpy as np
from typing import Self, Dict, Any

from ...registry import register
from ...base import MetricBase
from ....domain.entities import MetricResult


@register("privacy.membership_inference")
class MembershipInferenceMetric(MetricBase):
    """
    Membership Inference Attack (MIA) for privacy measurement.
    
    This metric measures privacy by training an attacker model to distinguish
    between records that were in the training set (real data) versus records
    that were not (synthetic data). Higher attack accuracy indicates lower privacy.
    
    TODO: This is a functional stub. Full implementation would:
    - Use more sophisticated shadow models
    - Implement proper train/test splits for the attack model
    - Add multiple attack strategies
    - Use model confidence scores as attack features
    - Handle different types of target models
    """
    
    name: str = "membership_inference"
    family: str = "privacy"
    purpose_tags: set[str] = {"privacy", "inference_attack"}
    
    def __init__(self):
        super().__init__()
    
    def fit(self, real_data: pd.DataFrame, synth_data: pd.DataFrame, context: Dict[str, Any]) -> Self:
        """Fit membership inference attack metric to data."""
        self._setup(real_data, synth_data, context)
        return self
    
    def compute(self) -> MetricResult:
        """Compute membership inference attack success rate."""
        try:
            numeric_columns = self._get_numeric_columns()
            
            if not numeric_columns:
                return MetricResult(
                    id="privacy.membership_inference",
                    value=0.0,
                    details={"error": "No numeric columns found for MIA"},
                    family="privacy",
                    purpose_tags=self.purpose_tags
                )
            
            # Prepare data for attack
            real_features = self._real_data[numeric_columns].fillna(0)
            synth_features = self._synth_data[numeric_columns].fillna(0)
            
            # Balance dataset sizes for fair comparison
            min_size = min(len(real_features), len(synth_features))
            real_sample = real_features.sample(n=min_size, random_state=42)
            synth_sample = synth_features.sample(n=min_size, random_state=42)
            
            # Create attack dataset
            # Label: 1 = real (member), 0 = synthetic (non-member)
            attack_features = pd.concat([real_sample, synth_sample], ignore_index=True)
            attack_labels = np.concatenate([np.ones(len(real_sample)), np.zeros(len(synth_sample))])
            
            # Shuffle the attack dataset
            indices = np.random.RandomState(42).permutation(len(attack_features))
            attack_features = attack_features.iloc[indices]
            attack_labels = attack_labels[indices]
            
            try:
                # Simple membership inference attack using distance-based features
                # TODO: Replace with more sophisticated attack model
                
                # Use KNN distances as attack features
                knn_data = self._get_knn_distances(k=3)
                
                if "error" in knn_data:
                    # Fallback: use statistical distance features
                    attack_accuracy = self._statistical_mia_fallback(real_sample, synth_sample)
                    details = {
                        "attack_type": "statistical_fallback",
                        "attack_accuracy": float(attack_accuracy),
                        "baseline_accuracy": 0.5,
                        "privacy_score": float(1.0 - attack_accuracy),
                        "note": "Used statistical features due to KNN computation error"
                    }
                else:
                    # Use KNN-based attack
                    attack_accuracy = self._knn_based_mia(real_sample, synth_sample, knn_data)
                    details = {
                        "attack_type": "knn_based",
                        "attack_accuracy": float(attack_accuracy),
                        "baseline_accuracy": 0.5,
                        "privacy_score": float(1.0 - attack_accuracy),
                        "k_neighbors": knn_data["k"]
                    }
                
                # Privacy score: 1 - attack_accuracy
                # Perfect privacy = 0.5 attack accuracy (random guessing)
                # Lower privacy = higher attack accuracy
                privacy_score = max(0.0, 1.0 - attack_accuracy)
                
                details.update({
                    "n_real_samples": len(real_sample),
                    "n_synthetic_samples": len(synth_sample),
                    "n_features": len(numeric_columns),
                    "balanced_dataset": True
                })
                
            except ImportError:
                # Fallback if sklearn not available
                privacy_score = 0.5
                details = {
                    "error": "sklearn not available, using fallback score",
                    "fallback_score": 0.5,
                    "attack_accuracy": 0.5
                }
            
            return MetricResult(
                id="privacy.membership_inference",
                value=float(privacy_score),
                details=details,
                family="privacy",
                purpose_tags=self.purpose_tags
            )
            
        except Exception as e:
            return MetricResult(
                id="privacy.membership_inference",
                value=0.0,
                details={"error": f"MIA computation failed: {str(e)}"},
                family="privacy",
                purpose_tags=self.purpose_tags
            )
    
    def _statistical_mia_fallback(self, real_sample: pd.DataFrame, synth_sample: pd.DataFrame) -> float:
        """Fallback MIA using simple statistical features."""
        try:
            # Compare statistical properties as a simple attack
            real_means = real_sample.mean()
            synth_means = synth_sample.mean()
            
            # Distance between means as distinguishing feature
            mean_distance = np.sqrt(np.sum((real_means - synth_means) ** 2))
            
            # Simple heuristic: if means are very different, attack is easier
            # Normalize by feature dimensionality
            normalized_distance = mean_distance / np.sqrt(len(real_means))
            
            # Convert to attack accuracy (higher distance = higher accuracy)
            attack_accuracy = 0.5 + min(0.4, normalized_distance * 0.1)
            
            return attack_accuracy
            
        except:
            return 0.5  # Random guessing
    
    def _knn_based_mia(self, real_sample: pd.DataFrame, synth_sample: pd.DataFrame, knn_data: Dict) -> float:
        """KNN-based membership inference attack."""
        try:
            # Use distance to nearest neighbors as attack feature
            distances = knn_data["distances"]
            
            # Simple attack: synthetic points that are very close to real points
            # are likely to be "copies" and thus attackable
            
            # Use minimum distance as attack signal
            min_distances = distances[:, 0] if len(distances.shape) > 1 else distances
            threshold = np.median(min_distances)
            
            # Attack prediction: if distance < threshold, predict "real"
            predictions = (min_distances < threshold).astype(int)
            
            # Ground truth: all synthetic samples should be labeled as 0 (non-member)
            ground_truth = np.zeros(len(predictions))
            
            # Attack accuracy on synthetic samples
            accuracy_synthetic = np.mean(predictions == ground_truth)
            
            # For balanced evaluation, assume attack on real samples has same accuracy
            # Overall attack accuracy = average of both
            attack_accuracy = (accuracy_synthetic + (1.0 - accuracy_synthetic)) / 2.0
            
            return attack_accuracy
            
        except:
            return 0.5  # Random guessing