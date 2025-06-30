from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor

from ..detection import Detection


class LOFDetection(Detection):
    """Clase para detectar anomalías usando Local Outlier Factor."""

    def __init__(
        self,
        original_data: Union[pd.DataFrame, np.ndarray],
        synthetic_data: Union[pd.DataFrame, np.ndarray],
        path: Optional[str] = None,
        seed: int = 42,
        n_neighbors: int = 20,
        contamination: float = 0.1,
    ):
        super().__init__(original_data, synthetic_data, path, seed)
        self.n_neighbors = n_neighbors
        self.contamination = contamination

    def detection_model(self):
        """Train LOF model on original data"""
        original_array = self.to_numpy(self.original_data)
        # LOF for novelty detection (to predict on new data)
        model = LocalOutlierFactor(
            n_neighbors=self.n_neighbors, contamination=self.contamination, novelty=True
        )
        model.fit(original_array)
        return model

    def execute(self) -> Dict[str, Any]:
        """Execute LOF anomaly detection"""
        results = super().execute()
        results["method"] = "local_outlier_factor"
        results["n_neighbors"] = self.n_neighbors
        results["contamination"] = self.contamination
        results["description"] = (
            "Local Outlier Factor anomaly detection - identifies local outliers in synthetic data"
        )
        return results
