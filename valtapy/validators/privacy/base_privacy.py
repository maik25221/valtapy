"""
Base class for privacy validation methods
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Union

import numpy as np
import pandas as pd


class Privacy(ABC):
    """Base class for privacy validation methods"""

    def __init__(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        gen_data: Union[pd.DataFrame, np.ndarray],
        path: str = None,
        seed: int = 42,
    ):
        """
        Initialize privacy validator

        Args:
            data: Original/real data
            gen_data: Generated/synthetic data
            path: Optional path for saving results
            seed: Random seed for reproducibility
        """
        self.data = data
        self.gen_data = gen_data
        self.path = path
        self.seed = seed
        np.random.seed(seed)

    @abstractmethod
    def execute(self) -> Dict[str, Any]:
        """Execute the privacy validation method"""
        pass

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
