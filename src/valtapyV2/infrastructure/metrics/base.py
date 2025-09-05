"""Base utilities for metric implementations."""

import pandas as pd
from typing import Any, Dict, List, Tuple, Optional
from ...infrastructure.runtime.cache import StatsStore
from ...domain.entities import DatasetSpec


class MetricBase:
    """Base class providing common utilities for metric implementations."""
    
    def __init__(self):
        self._real_data = None
        self._synth_data = None
        self._context = {}
        self._stats_store = None
    
    def _setup(self, real_data: pd.DataFrame, synth_data: pd.DataFrame, context: Dict[str, Any]):
        """Initialize metric with data and context."""
        self._real_data = real_data
        self._synth_data = synth_data
        self._context = context
        self._stats_store = context.get("stats_store")
    
    def _get_numeric_columns(self, exclude_target: bool = True) -> List[str]:
        """Get list of numeric columns from the real dataset."""
        numeric_cols = self._real_data.select_dtypes(include=['number']).columns.tolist()
        
        if exclude_target and "target" in self._context:
            target = self._context["target"]
            if target in numeric_cols:
                numeric_cols.remove(target)
        
        return numeric_cols
    
    def _get_categorical_columns(self, exclude_target: bool = True) -> List[str]:
        """Get list of categorical columns from the real dataset."""
        categorical_cols = self._real_data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if exclude_target and "target" in self._context:
            target = self._context["target"]
            if target in categorical_cols:
                categorical_cols.remove(target)
        
        return categorical_cols
    
    def _get_cached_or_compute(self, cache_key: str, compute_fn) -> Any:
        """Get value from cache or compute using provided function."""
        if self._stats_store:
            return self._stats_store.get_or_compute(cache_key, compute_fn)
        else:
            return compute_fn()
    
    def _get_univariate_stats(self, column: str, dataset: str = "real") -> Dict[str, Any]:
        """Get cached univariate statistics for a column."""
        data = self._real_data if dataset == "real" else self._synth_data
        cache_key = f"univariate:{dataset}:{column}"
        
        def compute_stats():
            series = data[column]
            
            if pd.api.types.is_numeric_dtype(series):
                return {
                    "mean": series.mean(),
                    "std": series.std(),
                    "min": series.min(),
                    "max": series.max(),
                    "median": series.median(),
                    "q25": series.quantile(0.25),
                    "q75": series.quantile(0.75),
                    "skew": series.skew(),
                    "kurtosis": series.kurtosis()
                }
            else:
                value_counts = series.value_counts()
                return {
                    "unique_count": series.nunique(),
                    "mode": series.mode().iloc[0] if len(series.mode()) > 0 else None,
                    "most_frequent_value": value_counts.index[0] if len(value_counts) > 0 else None,
                    "most_frequent_count": value_counts.iloc[0] if len(value_counts) > 0 else 0,
                    "value_counts": value_counts.to_dict()
                }
        
        return self._get_cached_or_compute(cache_key, compute_stats)
    
    def _get_correlation_matrix(self, dataset: str = "real") -> pd.DataFrame:
        """Get cached correlation matrix for numeric columns."""
        data = self._real_data if dataset == "real" else self._synth_data
        cache_key = f"correlation:{dataset}"
        
        def compute_correlation():
            numeric_data = data.select_dtypes(include=['number'])
            if numeric_data.empty:
                return pd.DataFrame()
            return numeric_data.corr()
        
        return self._get_cached_or_compute(cache_key, compute_correlation)
    
    def _get_train_test_splits(self, n_splits: int = 3, random_state: int = 42) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Get cached train/test splits for cross-validation."""
        cache_key = f"splits:n{n_splits}:seed{random_state}"
        
        def compute_splits():
            from sklearn.model_selection import KFold
            import numpy as np
            
            splits = []
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            
            # Use real data for splitting indices
            indices = np.arange(len(self._real_data))
            
            for train_idx, test_idx in kf.split(indices):
                train_real = self._real_data.iloc[train_idx]
                test_real = self._real_data.iloc[test_idx]
                splits.append((train_real, test_real))
            
            return splits
        
        return self._get_cached_or_compute(cache_key, compute_splits)
    
    def _get_knn_distances(self, k: int = 5) -> Dict[str, Any]:
        """Get cached k-nearest neighbor distances for privacy metrics."""
        cache_key = f"knn:k{k}"
        
        def compute_knn():
            from sklearn.neighbors import NearestNeighbors
            from sklearn.preprocessing import StandardScaler
            import numpy as np
            
            # Use only numeric columns for KNN
            numeric_cols = self._get_numeric_columns()
            if not numeric_cols:
                return {"error": "No numeric columns found for KNN computation"}
            
            real_numeric = self._real_data[numeric_cols].fillna(0)
            synth_numeric = self._synth_data[numeric_cols].fillna(0)
            
            # Standardize data
            scaler = StandardScaler()
            real_scaled = scaler.fit_transform(real_numeric)
            synth_scaled = scaler.transform(synth_numeric)
            
            # Fit KNN on real data
            knn = NearestNeighbors(n_neighbors=k)
            knn.fit(real_scaled)
            
            # Find distances from synthetic to real
            distances, indices = knn.kneighbors(synth_scaled)
            
            return {
                "distances": distances,
                "indices": indices,
                "mean_distance": np.mean(distances),
                "min_distance": np.min(distances),
                "max_distance": np.max(distances),
                "k": k
            }
        
        return self._get_cached_or_compute(cache_key, compute_knn)
    
    def _select_columns_by_type(self, column_type: str) -> List[str]:
        """Select columns based on type specification."""
        if column_type == "numeric":
            return self._get_numeric_columns()
        elif column_type == "categorical":
            return self._get_categorical_columns()
        elif column_type == "all":
            return self._real_data.columns.tolist()
        else:
            raise ValueError(f"Unknown column type: {column_type}")
    
    def _safe_divide(self, numerator: float, denominator: float, default: float = 0.0) -> float:
        """Safely divide two numbers, returning default if denominator is zero."""
        if abs(denominator) < 1e-10:
            return default
        return numerator / denominator