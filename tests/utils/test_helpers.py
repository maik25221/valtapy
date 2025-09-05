"""Test helper utilities."""

import tempfile
import os
from pathlib import Path
from typing import Dict, Any, List
import json

from ..utils.data_generators import dict_list_to_csv_string


def create_temp_csv_files(real_data: List[Dict], synth_data: List[Dict]) -> tuple[str, str]:
    """
    Create temporary CSV files from data lists.
    
    Returns:
        Tuple of (real_csv_path, synth_csv_path)
    """
    # Create temporary files
    real_fd, real_path = tempfile.mkstemp(suffix='.csv', prefix='test_real_')
    synth_fd, synth_path = tempfile.mkstemp(suffix='.csv', prefix='test_synth_')
    
    try:
        # Write real data
        with os.fdopen(real_fd, 'w') as f:
            f.write(dict_list_to_csv_string(real_data))
        
        # Write synthetic data
        with os.fdopen(synth_fd, 'w') as f:
            f.write(dict_list_to_csv_string(synth_data))
        
        return real_path, synth_path
        
    except:
        # Clean up on error
        try:
            os.unlink(real_path)
            os.unlink(synth_path)
        except:
            pass
        raise


def cleanup_temp_files(*file_paths: str) -> None:
    """Clean up temporary files."""
    for path in file_paths:
        try:
            if path and os.path.exists(path):
                os.unlink(path)
        except:
            pass


def assert_metric_result_structure(result, expected_id: str, expected_family: str):
    """Assert that a metric result has the expected structure."""
    from src.valtapyV2.domain.entities import MetricResult
    
    assert isinstance(result, MetricResult)
    assert result.id == expected_id
    assert result.family == expected_family
    assert isinstance(result.value, (int, float))
    assert isinstance(result.details, dict)
    assert isinstance(result.purpose_tags, set)


def assert_value_in_range(value: float, min_val: float = 0.0, max_val: float = 1.0):
    """Assert that a metric value is in the expected range."""
    assert min_val <= value <= max_val, f"Value {value} not in range [{min_val}, {max_val}]"


def assert_no_error_in_details(details: Dict[str, Any]):
    """Assert that result details don't contain error information."""
    assert "error" not in details, f"Unexpected error in details: {details.get('error')}"


def assert_has_required_details(details: Dict[str, Any], required_keys: List[str]):
    """Assert that details contain all required keys."""
    missing_keys = [key for key in required_keys if key not in details]
    assert not missing_keys, f"Missing required details: {missing_keys}"


def create_mock_context(include_stats_store: bool = True, **kwargs) -> Dict[str, Any]:
    """Create a mock context for metric testing."""
    context = {"seed": 42}
    
    if include_stats_store:
        from src.valtapyV2.infrastructure.runtime.cache import StatsStore
        context["stats_store"] = StatsStore()
    
    context.update(kwargs)
    return context


def create_test_config(real_path: str, synth_path: str, **overrides) -> Dict[str, Any]:
    """Create a test configuration dictionary."""
    config = {
        "data": {
            "real": real_path,
            "synthetic": synth_path,
            "target": "target"
        },
        "evaluation": {
            "purpose": "privacy_hardening",
            "seed": 42,
            "cv_splits": 3
        },
        "aggregation": {
            "weights": {"fidelity": 0.4, "utility": 0.3, "privacy": 0.3},
            "composite_index": True
        }
    }
    
    # Apply overrides
    for key, value in overrides.items():
        if "." in key:
            # Handle nested keys like "data.target"
            parts = key.split(".")
            current = config
            for part in parts[:-1]:
                current = current[part]
            current[parts[-1]] = value
        else:
            config[key] = value
    
    return config


def save_temp_config(config: Dict[str, Any]) -> str:
    """Save config to temporary YAML file and return path."""
    try:
        import yaml
    except ImportError:
        # Fallback to JSON if PyYAML not available
        import json
        fd, path = tempfile.mkstemp(suffix='.json', prefix='test_config_')
        with os.fdopen(fd, 'w') as f:
            json.dump(config, f, indent=2)
        return path
    
    fd, path = tempfile.mkstemp(suffix='.yaml', prefix='test_config_')
    with os.fdopen(fd, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    return path


class MetricTestCase:
    """Base class for metric test cases."""
    
    def __init__(self, name: str, real_data: List[Dict], synth_data: List[Dict], 
                 expected_value_range: tuple = (0.0, 1.0), 
                 should_succeed: bool = True,
                 description: str = ""):
        self.name = name
        self.real_data = real_data
        self.synth_data = synth_data
        self.expected_value_range = expected_value_range
        self.should_succeed = should_succeed
        self.description = description


def convert_to_pandas_mock(data: List[Dict]):
    """
    Create a mock pandas DataFrame-like object for testing without pandas dependency.
    
    This is a simple mock that provides basic DataFrame functionality for testing.
    """
    class MockDataFrame:
        def __init__(self, data: List[Dict]):
            self.data = data
            self.columns = list(data[0].keys()) if data else []
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, key):
            if isinstance(key, str):
                # Return column as MockSeries
                values = [row.get(key) for row in self.data]
                return MockSeries(values, name=key)
            return self.data[key]
        
        def select_dtypes(self, include=None):
            # Simple dtype detection
            numeric_cols = []
            for col in self.columns:
                values = [row.get(col) for row in self.data]
                if any(isinstance(v, (int, float)) for v in values if v is not None):
                    numeric_cols.append(col)
            
            return MockDataFrame([{col: row[col] for col in numeric_cols} for row in self.data])
        
        @property
        def shape(self):
            return (len(self.data), len(self.columns))
        
        def fillna(self, value):
            new_data = []
            for row in self.data:
                new_row = {}
                for k, v in row.items():
                    new_row[k] = value if v is None else v
                new_data.append(new_row)
            return MockDataFrame(new_data)
        
        def empty(self):
            return len(self.data) == 0
        
        def dropna(self):
            return MockDataFrame([row for row in self.data if None not in row.values()])
    
    class MockSeries:
        def __init__(self, values: List, name: str = None):
            self.values = [v for v in values if v is not None]
            self.name = name
        
        def __len__(self):
            return len(self.values)
        
        def nunique(self):
            return len(set(self.values))
        
        def dropna(self):
            return MockSeries(self.values, self.name)
        
        def mean(self):
            numeric_vals = [v for v in self.values if isinstance(v, (int, float))]
            return sum(numeric_vals) / len(numeric_vals) if numeric_vals else 0
        
        def std(self):
            numeric_vals = [v for v in self.values if isinstance(v, (int, float))]
            if len(numeric_vals) < 2:
                return 0
            mean_val = sum(numeric_vals) / len(numeric_vals)
            variance = sum((x - mean_val) ** 2 for x in numeric_vals) / (len(numeric_vals) - 1)
            return variance ** 0.5
    
    return MockDataFrame(data)