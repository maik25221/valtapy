from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from ..detection import Detection


class IsolationForestDetection(Detection):
    """Clase para detectar anomalías usando Isolation Forest."""

    def __init__(
        self,
        original_data: Union[pd.DataFrame, np.ndarray],
        synthetic_data: Union[pd.DataFrame, np.ndarray],
        path: Optional[str] = None,
        seed: int = 42,
        contamination: float = 0.1,
    ):
        super().__init__(original_data, synthetic_data, path, seed)
        self.contamination = contamination

    def detection_model(self):
        """Train Isolation Forest model on original data"""
        original_array = self.to_numpy(self.original_data)
        model = IsolationForest(
            random_state=self.seed, contamination=self.contamination, n_estimators=100
        )
        model.fit(original_array)
        return model

    def execute(self) -> Dict[str, Any]:
        """Execute Isolation Forest anomaly detection"""
        results = super().execute()
        results["method"] = "isolation_forest"
        results["contamination"] = self.contamination
        results["description"] = (
            "Isolation Forest anomaly detection - identifies outliers in synthetic data"
        )
        return results
