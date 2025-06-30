import os
from typing import Any, Dict, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ..stadistics import Statistics

# Try to import seaborn, if not available use matplotlib only
try:
    import seaborn as sns

    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


class Metrics(Statistics):
    def __init__(
        self,
        original_data: Union[pd.DataFrame, np.ndarray],
        synthetic_data: Union[pd.DataFrame, np.ndarray],
        path: Optional[str] = None,
        seed: int = 42,
    ):
        super().__init__(original_data, synthetic_data, path, seed)

    def plot_density_per_variable(self):
        """Generate density plots for each variable"""
        if not self.path:
            return

        try:
            # Create directory for density plots
            density_dir = os.path.join(os.path.dirname(self.path), "density_plots")
            os.makedirs(density_dir, exist_ok=True)

            # Generate density plot for each column
            for column in self.original_data.columns:
                fig, ax = plt.subplots(figsize=(10, 6))

                # Plot densities
                if HAS_SEABORN:
                    sns.kdeplot(
                        self.original_data[column],
                        ax=ax,
                        fill=True,
                        label="Original",
                        alpha=0.7,
                        color="blue",
                    )
                    sns.kdeplot(
                        self.synthetic_data[column],
                        ax=ax,
                        fill=True,
                        label="Synthetic",
                        alpha=0.7,
                        color="red",
                    )
                else:
                    # Fallback to matplotlib histogram if seaborn not available
                    ax.hist(
                        self.original_data[column],
                        bins=50,
                        alpha=0.7,
                        label="Original",
                        color="blue",
                        density=True,
                    )
                    ax.hist(
                        self.synthetic_data[column],
                        bins=50,
                        alpha=0.7,
                        label="Synthetic",
                        color="red",
                        density=True,
                    )

                # Calculate statistics
                median_orig = self.original_data[column].median()
                mean_orig = self.original_data[column].mean()
                median_syn = self.synthetic_data[column].median()
                mean_syn = self.synthetic_data[column].mean()

                # Add vertical lines for statistics
                ax.axvline(
                    median_orig,
                    color="darkblue",
                    linestyle="-",
                    alpha=0.8,
                    label=f"Original Median: {median_orig:.2f}",
                )
                ax.axvline(
                    mean_orig,
                    color="darkblue",
                    linestyle="--",
                    alpha=0.8,
                    label=f"Original Mean: {mean_orig:.2f}",
                )
                ax.axvline(
                    median_syn,
                    color="darkred",
                    linestyle="-",
                    alpha=0.8,
                    label=f"Synthetic Median: {median_syn:.2f}",
                )
                ax.axvline(
                    mean_syn,
                    color="darkred",
                    linestyle="--",
                    alpha=0.8,
                    label=f"Synthetic Mean: {mean_syn:.2f}",
                )

                # Configure plot
                ax.set_title(f"Density Plot of {column}")
                ax.set_xlabel("Value")
                ax.set_ylabel("Density")
                ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

                plt.tight_layout()
                plt.savefig(
                    os.path.join(density_dir, f"{column}_density.png"),
                    dpi=150,
                    bbox_inches="tight",
                )
                plt.close()

        except Exception as e:
            print(f"Could not generate density plots: {e}")

    def calculate_statistical_metrics(self) -> Dict[str, Any]:
        """Calculate statistical metrics comparing original and synthetic data"""
        metrics = {}

        for column in self.original_data.columns:
            orig_col = self.original_data[column]
            synth_col = self.synthetic_data[column]

            # Basic statistics
            orig_mean = orig_col.mean()
            synth_mean = synth_col.mean()
            orig_std = orig_col.std()
            synth_std = synth_col.std()
            orig_median = orig_col.median()
            synth_median = synth_col.median()

            # Calculate differences
            mean_diff = abs(orig_mean - synth_mean)
            std_diff = abs(orig_std - synth_std)
            median_diff = abs(orig_median - synth_median)

            # Normalize differences by original values to get relative errors
            mean_rel_error = mean_diff / (abs(orig_mean) + 1e-8)
            std_rel_error = std_diff / (abs(orig_std) + 1e-8)
            median_rel_error = median_diff / (abs(orig_median) + 1e-8)

            metrics[column] = {
                "mean_relative_error": mean_rel_error,
                "std_relative_error": std_rel_error,
                "median_relative_error": median_rel_error,
                "combined_error": (mean_rel_error + std_rel_error + median_rel_error)
                / 3.0,
            }

        # Calculate overall metrics
        all_errors = [metrics[col]["combined_error"] for col in metrics.keys()]
        overall_error = np.mean(all_errors)
        overall_quality = max(0.0, 1.0 - min(overall_error, 1.0))

        return {
            "per_column_metrics": metrics,
            "overall_statistical_error": overall_error,
            "overall_statistical_quality": overall_quality,
        }

    def execute(self) -> Dict[str, Any]:
        """Execute metrics analysis"""
        try:
            # Don't standardize for metrics calculation to preserve original scale relationships
            # self.standardize_data(method="none")

            # Calculate statistical metrics
            statistical_results = self.calculate_statistical_metrics()

            # Generate plots if path is provided
            if self.path:
                self.plot_density_per_variable()

            return {
                **statistical_results,
                "description": "Statistical metrics comparing distributions and summary statistics",
            }

        except Exception as e:
            return {
                "error": str(e),
                "per_column_metrics": {},
                "overall_statistical_error": 1.0,
                "overall_statistical_quality": 0.0,
            }
            # print(self.original_data.head())
            self.original_data.boxplot(column, ax=ax)  # Plot synthetic data

            # Obtener estadísticas descriptivas para ambos conjuntos de datos
            median_orig = self.original_data[column].median()
            mean_orig = self.original_data[column].mean()
            std_orig = self.original_data[column].std()
            median_syn = self.synthetic_data[column].median()
            mean_syn = self.synthetic_data[column].mean()
            std_syn = self.synthetic_data[column].std()

            # Añadir un texto para la desviación estándar de los datos sintéticos
            ax.text(
                0.95,
                0.01,
                f"Synthetic Std: {std_syn:.2f}",
                verticalalignment="bottom",
                horizontalalignment="right",
                transform=ax.transAxes,
                color="blue",
                fontsize=10,
            )

            # Añadir líneas horizontales para la media y la mediana de ambos conjuntos de datos
            ax.axhline(
                median_orig,
                color="red",
                linestyle="-",
                label=f"Original Median: {median_orig:.2f}",
            )
            ax.axhline(
                mean_orig,
                color="green",
                linestyle="--",
                label=f"Original Mean: {mean_orig:.2f}",
            )
            ax.axhline(
                median_syn,
                color="purple",
                linestyle="-",
                label=f"Synthetic Median: {median_syn:.2f}",
            )
            ax.axhline(
                mean_syn,
                color="orange",
                linestyle="--",
                label=f"Synthetic Mean: {mean_syn:.2f}",
            )

            # Mostrar la leyenda
            plt.legend()

            plt.tight_layout()  # Ajustar automáticamente los subplots para que encajen en la figura
            plt.savefig(self.path + "boxplot/" + f"{column}.png")
            plt.close()

    def execute(self):
        self.standardize_data()
        self.plot_density_per_variable()
        self.plot_statistics()
