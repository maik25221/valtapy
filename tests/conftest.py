"""Pytest configuration and fixtures."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os

from src.valtapyV2.domain.entities import DatasetSpec, EvalPlan


@pytest.fixture
def sample_real_data():
    """Generate sample real data for testing."""
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'age': np.random.randint(18, 80, n_samples),
        'income': np.random.normal(50000, 15000, n_samples),
        'score': np.random.uniform(0, 100, n_samples),
        'category': np.random.choice(['A', 'B', 'C'], n_samples),
        'is_member': np.random.choice([0, 1], n_samples),
        'target': np.random.choice(['low', 'medium', 'high'], n_samples)
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_synthetic_data():
    """Generate sample synthetic data for testing."""
    np.random.seed(123)  # Different seed for variation
    n_samples = 100
    
    data = {
        'age': np.random.randint(20, 75, n_samples),
        'income': np.random.normal(48000, 16000, n_samples),
        'score': np.random.uniform(5, 95, n_samples),
        'category': np.random.choice(['A', 'B', 'C'], n_samples),
        'is_member': np.random.choice([0, 1], n_samples),
        'target': np.random.choice(['low', 'medium', 'high'], n_samples)
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_dataset_spec():
    """Generate sample dataset specification."""
    return DatasetSpec(
        target="target",
        dtypes={
            "age": "int64",
            "income": "float64", 
            "score": "float64",
            "category": "object",
            "is_member": "int64",
            "target": "object"
        }
    )


@pytest.fixture
def sample_eval_plan():
    """Generate sample evaluation plan."""
    return EvalPlan(
        metric_ids=["fidelity.ks", "utility.pmse", "privacy.nndr"],
        seed=42,
        cv_splits=3,
        purpose="privacy_hardening"
    )


@pytest.fixture
def temp_data_files(sample_real_data, sample_synthetic_data):
    """Create temporary CSV files with sample data."""
    temp_dir = tempfile.mkdtemp()
    
    real_path = Path(temp_dir) / "real.csv"
    synth_path = Path(temp_dir) / "synth.csv"
    
    sample_real_data.to_csv(real_path, index=False)
    sample_synthetic_data.to_csv(synth_path, index=False)
    
    yield str(real_path), str(synth_path)
    
    # Cleanup
    try:
        os.unlink(real_path)
        os.unlink(synth_path)
        os.rmdir(temp_dir)
    except:
        pass


@pytest.fixture
def sample_config(temp_data_files):
    """Generate sample configuration dictionary."""
    real_path, synth_path = temp_data_files
    
    return {
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
        },
        "report": {
            "formats": ["json"],
            "output_dir": "test_reports"
        }
    }


@pytest.fixture
def temp_config_file(sample_config):
    """Create temporary configuration file."""
    import yaml
    
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    yaml.dump(sample_config, temp_file, default_flow_style=False)
    temp_file.close()
    
    yield temp_file.name
    
    # Cleanup
    try:
        os.unlink(temp_file.name)
    except:
        pass