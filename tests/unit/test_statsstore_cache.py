"""Test StatsStore caching functionality."""

import pytest
import time
from src.valtapyV2.infrastructure.runtime.cache import StatsStore


class TestStatsStore:
    """Test the StatsStore caching system."""
    
    def test_basic_cache_operations(self):
        """Test basic cache get/set operations."""
        cache = StatsStore()
        
        # Test cache miss and computation
        def expensive_computation():
            return 42
        
        result = cache.get_or_compute("test_key", expensive_computation)
        assert result == 42
        
        # Test cache hit
        result2 = cache.get_or_compute("test_key", lambda: 99)
        assert result2 == 42  # Should return cached value, not recompute
        
        # Verify cache statistics
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5
    
    def test_cache_with_different_keys(self):
        """Test cache with different keys."""
        cache = StatsStore()
        
        # Store different values with different keys
        result1 = cache.get_or_compute("key1", lambda: "value1")
        result2 = cache.get_or_compute("key2", lambda: "value2")
        
        assert result1 == "value1"
        assert result2 == "value2"
        
        # Verify both are cached
        assert cache.get_or_compute("key1", lambda: "changed") == "value1"
        assert cache.get_or_compute("key2", lambda: "changed") == "value2"
        
        stats = cache.get_stats()
        assert stats["size"] == 2
        assert stats["hits"] == 2
        assert stats["misses"] == 2
    
    def test_cache_size_limit_and_eviction(self):
        """Test cache size limits and LRU eviction."""
        cache = StatsStore(max_size=3)
        
        # Fill cache to capacity
        cache.get_or_compute("key1", lambda: "value1")
        cache.get_or_compute("key2", lambda: "value2")
        cache.get_or_compute("key3", lambda: "value3")
        
        assert cache.size() == 3
        assert cache.contains("key1")
        assert cache.contains("key2")
        assert cache.contains("key3")
        
        # Access key1 to make it more recently used
        cache.get_or_compute("key1", lambda: "changed")
        
        # Add another key, should evict least recently used (key2)
        cache.get_or_compute("key4", lambda: "value4")
        
        assert cache.size() == 3
        assert cache.contains("key1")  # Recently accessed
        assert not cache.contains("key2")  # Should be evicted
        assert cache.contains("key3")
        assert cache.contains("key4")  # Newly added
    
    def test_cache_invalidation(self):
        """Test manual cache invalidation."""
        cache = StatsStore()
        
        # Add some values
        cache.get_or_compute("key1", lambda: "value1")
        cache.get_or_compute("key2", lambda: "value2")
        
        assert cache.contains("key1")
        assert cache.contains("key2")
        
        # Invalidate one key
        result = cache.invalidate("key1")
        assert result is True
        assert not cache.contains("key1")
        assert cache.contains("key2")
        
        # Try to invalidate non-existent key
        result = cache.invalidate("nonexistent")
        assert result is False
    
    def test_cache_clear(self):
        """Test cache clearing."""
        cache = StatsStore()
        
        # Add some values
        cache.get_or_compute("key1", lambda: "value1")
        cache.get_or_compute("key2", lambda: "value2")
        
        assert cache.size() == 2
        
        # Clear cache
        cache.clear()
        
        assert cache.size() == 0
        assert not cache.contains("key1")
        assert not cache.contains("key2")
        
        # Stats should be reset
        stats = cache.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
    
    def test_cache_with_complex_objects(self):
        """Test caching complex objects."""
        cache = StatsStore()
        
        # Test with different object types
        import pandas as pd
        import numpy as np
        
        def compute_dataframe():
            return pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        
        def compute_array():
            return np.array([1, 2, 3, 4, 5])
        
        def compute_dict():
            return {'nested': {'key': 'value'}, 'list': [1, 2, 3]}
        
        # Cache different object types
        df_result = cache.get_or_compute("dataframe", compute_dataframe)
        array_result = cache.get_or_compute("array", compute_array)
        dict_result = cache.get_or_compute("dict", compute_dict)
        
        # Verify objects are correctly cached and returned
        assert isinstance(df_result, pd.DataFrame)
        assert df_result.equals(pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]}))
        
        assert isinstance(array_result, np.ndarray)
        assert np.array_equal(array_result, np.array([1, 2, 3, 4, 5]))
        
        assert isinstance(dict_result, dict)
        assert dict_result == {'nested': {'key': 'value'}, 'list': [1, 2, 3]}
        
        # Verify cache hits return same objects
        df_result2 = cache.get_or_compute("dataframe", lambda: "changed")
        assert df_result2.equals(df_result)
    
    def test_cache_error_handling(self):
        """Test cache behavior when computation fails."""
        cache = StatsStore()
        
        def failing_computation():
            raise ValueError("Test error")
        
        # Error should propagate, not be cached
        with pytest.raises(ValueError, match="Test error"):
            cache.get_or_compute("error_key", failing_computation)
        
        # Key should not be cached after error
        assert not cache.contains("error_key")
        
        # Next call should attempt computation again
        def working_computation():
            return "success"
        
        result = cache.get_or_compute("error_key", working_computation)
        assert result == "success"
        assert cache.contains("error_key")
    
    def test_cache_key_generators(self):
        """Test the cache key generation utilities."""
        from src.valtapyV2.infrastructure.runtime.cache import (
            make_univariate_key,
            make_bivariate_key,
            make_dataset_key
        )
        
        # Test univariate key generator
        key1 = make_univariate_key("age", "mean")
        assert key1 == "univariate:age:mean"
        
        # Test bivariate key generator (should handle ordering)
        key2a = make_bivariate_key("age", "income", "correlation")
        key2b = make_bivariate_key("income", "age", "correlation")
        assert key2a == key2b  # Should be order-independent
        assert "bivariate:" in key2a
        
        # Test dataset key generator
        key3a = make_dataset_key("splits")
        key3b = make_dataset_key("splits", seed=42)
        assert key3a == "dataset:splits"
        assert key3b == "dataset:splits:seed42"