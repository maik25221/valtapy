from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer, StandardScaler

from ..base_quality import Quality


class Statistics(Quality):
    """Clase base para realizar pruebas estadísticas entre dos conjuntos de datos."""

    def __init__(
        self,
        original_data: Union[pd.DataFrame, np.ndarray],
        synthetic_data: Union[pd.DataFrame, np.ndarray],
        path: Optional[str] = None,
        seed: int = 42,
    ):
        super().__init__(original_data, synthetic_data, path, seed)
        self.original_data_all = None
        self.synthetic_data_all = None

    def standardize_data(self, method: str = "quantile"):
        """
        Standardize the data using different methods

        Args:
            method: 'quantile', 'standard', or 'none'
        """
        self.original_data_all = self.original_data.copy()
        self.synthetic_data_all = self.synthetic_data.copy()

        if method == "quantile":
            transformer = QuantileTransformer(
                output_distribution="normal", random_state=self.seed
            )
            original_transformed = transformer.fit_transform(self.original_data)
            synthetic_transformed = transformer.transform(self.synthetic_data)

            self.original_data = pd.DataFrame(
                original_transformed, columns=self.original_data.columns
            )
            self.synthetic_data = pd.DataFrame(
                synthetic_transformed, columns=self.synthetic_data.columns
            )

        elif method == "standard":
            scaler = StandardScaler()
            original_transformed = scaler.fit_transform(self.original_data)
            synthetic_transformed = scaler.transform(self.synthetic_data)

            self.original_data = pd.DataFrame(
                original_transformed, columns=self.original_data.columns
            )
            self.synthetic_data = pd.DataFrame(
                synthetic_transformed, columns=self.synthetic_data.columns
            )

        # If method == "none", keep data as is
        print(f"Data has been standardized using {method} method.")

    def execute(self) -> Dict[str, Any]:
        """Execute the statistical analysis - to be overridden by subclasses"""
        raise NotImplementedError("Subclasses must implement the execute method")
