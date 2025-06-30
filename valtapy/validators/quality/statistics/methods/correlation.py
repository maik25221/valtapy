import os
from typing import Any, Dict, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ..stadistics import Statistics


class Correlation(Statistics):
    def __init__(
        self,
        original_data: Union[pd.DataFrame, np.ndarray],
        synthetic_data: Union[pd.DataFrame, np.ndarray],
        path: Optional[str] = None,
        seed: int = 42,
    ):
        super().__init__(original_data, synthetic_data, path, seed)

    def matrix_correlation(self, df: pd.DataFrame, type_data: str) -> np.ndarray:
        """Calculate and optionally plot correlation matrix"""
        corr = df.corr()

        if self.path:
            try:
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(self.path), exist_ok=True)

                fig, ax = plt.subplots(figsize=(10, 10))
                cax = ax.matshow(corr, cmap="coolwarm")

                # Add numbers to correlation matrix
                for i in range(len(corr.columns)):
                    for j in range(len(corr.columns)):
                        text = ax.text(
                            j,
                            i,
                            round(corr.iloc[i, j], 2),
                            ha="center",
                            va="center",
                            color="black",
                        )

                fig.colorbar(cax)
                plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
                plt.yticks(range(len(corr.columns)), corr.columns)
                plt.title(f"Correlation Matrix - {type_data}")
                plt.tight_layout()
                plt.savefig(
                    self.path + f"_{type_data}.png", dpi=150, bbox_inches="tight"
                )
                plt.close()
            except Exception as e:
                print(f"Could not save correlation plot: {e}")

        return corr.values

    def calculate_correlation_similarity(
        self, corr_orig: np.ndarray, corr_synth: np.ndarray
    ) -> float:
        """Calculate similarity between correlation matrices"""
        # Flatten upper triangular parts (excluding diagonal)
        mask = np.triu(np.ones_like(corr_orig, dtype=bool), k=1)
        orig_upper = corr_orig[mask]
        synth_upper = corr_synth[mask]

        # Calculate correlation between the correlation values
        if len(orig_upper) > 1:
            correlation_similarity = np.corrcoef(orig_upper, synth_upper)[0, 1]
            # Handle NaN case
            if np.isnan(correlation_similarity):
                correlation_similarity = 0.0
        else:
            correlation_similarity = 1.0  # Perfect if only one feature

        return correlation_similarity

    def execute(self) -> Dict[str, Any]:
        """Execute correlation analysis"""
        try:
            self.standardize_data(method="standard")

            # Calculate correlation matrices
            corr_original = self.matrix_correlation(self.original_data, "original")
            corr_synthetic = self.matrix_correlation(self.synthetic_data, "synthetic")

            # Calculate similarity score
            correlation_similarity = self.calculate_correlation_similarity(
                corr_original, corr_synthetic
            )

            # Calculate mean absolute difference in correlations
            diff_matrix = np.abs(corr_original - corr_synthetic)
            mask = np.triu(np.ones_like(diff_matrix, dtype=bool), k=1)
            mean_abs_diff = np.mean(diff_matrix[mask]) if np.any(mask) else 0.0

            # Quality score: high correlation similarity and low difference is better
            quality_score = max(0.0, correlation_similarity) * max(
                0.0, 1.0 - mean_abs_diff
            )

            return {
                "correlation_similarity": correlation_similarity,
                "mean_absolute_correlation_difference": mean_abs_diff,
                "correlation_quality_score": quality_score,
                "original_correlation_matrix": corr_original.tolist(),
                "synthetic_correlation_matrix": corr_synthetic.tolist(),
                "description": "Correlation analysis comparing relationship patterns between features",
            }

        except Exception as e:
            return {
                "error": str(e),
                "correlation_similarity": 0.0,
                "mean_absolute_correlation_difference": 1.0,
                "correlation_quality_score": 0.0,
            }
