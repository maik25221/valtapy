from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import entropy, ks_2samp
from sklearn.preprocessing import StandardScaler

from ..stadistics import Statistics


class Tests(Statistics):
    """Class to perform statistical tests between two datasets."""

    def __init__(
        self,
        original_data: Union[pd.DataFrame, np.ndarray],
        synthetic_data: Union[pd.DataFrame, np.ndarray],
        path: Optional[str] = None,
        seed: int = 42,
        all_data: bool = True,
    ):
        super().__init__(original_data, synthetic_data, path, seed)
        self.statistic = None
        self.p_value = None
        self.kl_divergence = None
        self.all_data = all_data

    def test_ks(self):
        """Perform the Kolmogorov-Smirnov test."""
        if self.all_data:
            self.statistic, self.p_value = ks_2samp(
                self.original_data.values.flatten(),
                self.synthetic_data.values.flatten(),
            )
        else:
            # For single column data
            if isinstance(self.original_data, pd.DataFrame):
                orig_values = self.original_data.iloc[:, 0].values
                synth_values = self.synthetic_data.iloc[:, 0].values
            else:
                orig_values = self.original_data
                synth_values = self.synthetic_data
            self.statistic, self.p_value = ks_2samp(orig_values, synth_values)

    def obtener_histogramas_normalizados(
        self, datos: pd.DataFrame, num_bins: int = 50
    ) -> Dict[str, np.ndarray]:
        """Get normalized histograms for each column"""
        histogramas = {}
        for columna in datos.columns:
            hist, bin_edges = np.histogram(datos[columna], bins=num_bins, density=True)
            hist += 1e-10  # Add small value to avoid zeros
            histogramas[columna] = hist
        return histogramas

    def test_kl(self, base=None):
        """Calculate the KL divergence for the entire dataset."""
        if self.all_data:
            histogramas_orig = self.obtener_histogramas_normalizados(self.original_data)
            histogramas_sint = self.obtener_histogramas_normalizados(
                self.synthetic_data
            )

            kl_divergencias = []
            for columna in histogramas_orig.keys():
                if columna in histogramas_sint:
                    kl_div = entropy(
                        histogramas_orig[columna], histogramas_sint[columna]
                    )
                    kl_divergencias.append(kl_div)

            self.kl_divergence = (
                np.mean(kl_divergencias) if kl_divergencias else float("inf")
            )
        else:
            # For single column analysis
            if isinstance(self.original_data, pd.DataFrame):
                orig_col = self.original_data.iloc[:, 0]
                synth_col = self.synthetic_data.iloc[:, 0]

                hist_orig, _ = np.histogram(orig_col, bins=50, density=True)
                hist_synth, _ = np.histogram(synth_col, bins=50, density=True)

                hist_orig += 1e-10
                hist_synth += 1e-10

                self.kl_divergence = entropy(hist_orig, hist_synth)
            else:
                self.kl_divergence = float("inf")

    def write_tests(self):
        """Write test results to a file."""
        if self.path:
            try:
                with open(self.path, "w") as file:
                    file.write(f"Kolmogorov-Smirnov Test Results:\n")
                    file.write(f"KS statistic: {self.statistic:.6f}\n")
                    file.write(f"KS p-value: {self.p_value:.6f}\n")
                    file.write(f"KL divergence: {self.kl_divergence:.6f}\n")
                    file.write(f"\nInterpretation:\n")
                    file.write(
                        f"KS statistic closer to 0 = more similar distributions\n"
                    )
                    file.write(
                        f"KS p-value > 0.05 = distributions are not significantly different\n"
                    )
                    file.write(
                        f"KL divergence closer to 0 = more similar distributions\n"
                    )
            except IOError as e:
                print(f"Error writing to file: {e}")

    def execute(self) -> Dict[str, Any]:
        """Execute all tests and return results."""
        try:
            self.standardize_data(method="standard")
            self.test_ks()
            self.test_kl()

            if self.path:
                self.write_tests()

            # Calculate quality scores
            # KS p-value: higher is better (distributions are similar)
            ks_quality = (
                min(self.p_value * 10, 1.0) if self.p_value is not None else 0.0
            )

            # KL divergence: lower is better (closer to 0)
            kl_quality = (
                max(0.0, 1.0 - min(self.kl_divergence / 10.0, 1.0))
                if self.kl_divergence != float("inf")
                else 0.0
            )

            overall_quality = (ks_quality + kl_quality) / 2.0

            return {
                "ks_statistic": self.statistic,
                "ks_p_value": self.p_value,
                "kl_divergence": self.kl_divergence,
                "ks_quality_score": ks_quality,
                "kl_quality_score": kl_quality,
                "overall_statistical_quality": overall_quality,
                "description": "Statistical tests comparing distributions between real and synthetic data",
            }

        except Exception as e:
            return {
                "error": str(e),
                "ks_statistic": None,
                "ks_p_value": None,
                "kl_divergence": float("inf"),
                "ks_quality_score": 0.0,
                "kl_quality_score": 0.0,
                "overall_statistical_quality": 0.0,
            }
