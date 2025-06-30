"""
Base class for quality validation methods
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd


class Quality(ABC):
    """Base class for quality validation methods"""

    def __init__(
        self,
        original_data: Union[pd.DataFrame, np.ndarray],
        synthetic_data: Union[pd.DataFrame, np.ndarray],
        path: Optional[str] = None,
        seed: int = 42,
    ):
        """
        Initialize quality validator

        Args:
            original_data: Original/real data
            synthetic_data: Generated/synthetic data
            path: Optional path for saving results
            seed: Random seed for reproducibility
        """
        self.original_data = self._ensure_dataframe(original_data)
        self.synthetic_data = self._ensure_dataframe(synthetic_data)
        self.path = path
        self.seed = seed
        np.random.seed(seed)

    def _ensure_dataframe(self, data: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """Ensure data is a pandas DataFrame"""
        if isinstance(data, pd.DataFrame):
            return data
        elif isinstance(data, np.ndarray):
            return pd.DataFrame(data)
        else:
            return pd.DataFrame(np.array(data))

    def to_numpy(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Convert data to NumPy array if it is not already."""
        if isinstance(data, np.ndarray):
            return data
        elif hasattr(data, "detach"):  # torch.Tensor
            return data.detach().cpu().numpy()
        elif isinstance(data, pd.DataFrame):
            return data.values
        else:
            return np.array(data)

    @abstractmethod
    def execute(self) -> Dict[str, Any]:
        """Execute the quality validation method"""
        pass
