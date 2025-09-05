"""Data generation utilities for testing."""

import random
from typing import Dict, List, Tuple, Optional
import math


def generate_tabular_data(
    n_samples: int = 100,
    n_numeric: int = 3,
    n_categorical: int = 2,
    seed: int = 42,
    include_target: bool = True,
    missing_rate: float = 0.0
) -> List[Dict]:
    """
    Generate synthetic tabular data for testing.
    
    Args:
        n_samples: Number of samples to generate
        n_numeric: Number of numeric columns
        n_categorical: Number of categorical columns  
        seed: Random seed for reproducibility
        include_target: Whether to include a target column
        missing_rate: Fraction of values to make missing (0.0 to 1.0)
        
    Returns:
        List of dictionaries representing rows
    """
    random.seed(seed)
    data = []
    
    # Define column templates
    numeric_templates = [
        ('age', lambda: random.randint(18, 80)),
        ('income', lambda: random.gauss(50000, 15000)),
        ('score', lambda: random.uniform(0, 100)),
        ('rating', lambda: random.uniform(1, 5)),
        ('price', lambda: random.lognormvariate(3, 1)),
    ]
    
    categorical_templates = [
        ('category', lambda: random.choice(['A', 'B', 'C', 'D'])),
        ('region', lambda: random.choice(['North', 'South', 'East', 'West'])),
        ('status', lambda: random.choice(['active', 'inactive', 'pending'])),
        ('type', lambda: random.choice(['type1', 'type2', 'type3'])),
        ('grade', lambda: random.choice(['excellent', 'good', 'fair', 'poor'])),
    ]
    
    # Select columns to use
    numeric_cols = numeric_templates[:n_numeric]
    categorical_cols = categorical_templates[:n_categorical]
    
    for i in range(n_samples):
        row = {}
        
        # Generate numeric columns
        for col_name, generator in numeric_cols:
            value = generator()
            if random.random() < missing_rate:
                value = None
            row[col_name] = value
        
        # Generate categorical columns
        for col_name, generator in categorical_cols:
            value = generator()
            if random.random() < missing_rate:
                value = None
            row[col_name] = value
        
        # Generate target column if requested
        if include_target:
            # Create target based on some of the features for realism
            if n_numeric > 0:
                first_numeric = next(iter([v for v in row.values() if isinstance(v, (int, float))]), 50)
                if first_numeric > 60:
                    target = random.choice(['high', 'high', 'medium'])  # Biased towards high
                elif first_numeric > 30:
                    target = random.choice(['medium', 'medium', 'high', 'low'])
                else:
                    target = random.choice(['low', 'low', 'medium'])
            else:
                target = random.choice(['low', 'medium', 'high'])
            
            if random.random() < missing_rate:
                target = None
            row['target'] = target
        
        data.append(row)
    
    return data


def generate_correlated_datasets(
    n_samples: int = 100,
    correlation: float = 0.8,
    noise_level: float = 0.1,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    """
    Generate two correlated datasets for testing fidelity metrics.
    
    Args:
        n_samples: Number of samples to generate
        correlation: Correlation level between datasets (0.0 to 1.0)
        noise_level: Amount of noise to add to synthetic data
        seed: Random seed
        
    Returns:
        Tuple of (real_data, synthetic_data) as lists of dictionaries
    """
    random.seed(seed)
    
    real_data = generate_tabular_data(n_samples, seed=seed)
    synthetic_data = []
    
    for row in real_data:
        synth_row = {}
        
        for key, value in row.items():
            if value is None:
                synth_value = None
            elif isinstance(value, (int, float)):
                # Add correlated noise to numeric values
                base_value = value * correlation
                noise = random.gauss(0, abs(value) * noise_level)
                synth_value = base_value + noise
                
                if isinstance(value, int):
                    synth_value = int(round(synth_value))
            else:
                # For categorical, sometimes change the value
                if random.random() > correlation:
                    # Get a different value from the same column in real data
                    possible_values = list(set(r[key] for r in real_data if r[key] is not None))
                    if possible_values:
                        synth_value = random.choice(possible_values)
                    else:
                        synth_value = value
                else:
                    synth_value = value
            
            synth_row[key] = synth_value
        
        synthetic_data.append(synth_row)
    
    return real_data, synthetic_data


def dict_list_to_csv_string(data: List[Dict]) -> str:
    """Convert list of dictionaries to CSV string."""
    if not data:
        return ""
    
    # Get headers
    headers = list(data[0].keys())
    lines = [",".join(headers)]
    
    # Add data rows
    for row in data:
        values = []
        for header in headers:
            value = row.get(header)
            if value is None:
                values.append("")
            else:
                values.append(str(value))
        lines.append(",".join(values))
    
    return "\n".join(lines)


def create_edge_case_data() -> Tuple[List[Dict], List[Dict]]:
    """Create edge case datasets for robust testing."""
    # Dataset with extreme values
    real_data = [
        {'numeric1': 0, 'numeric2': 1000000, 'category': 'A', 'target': 'high'},
        {'numeric1': -999, 'numeric2': 0.001, 'category': 'B', 'target': 'low'},
        {'numeric1': 50, 'numeric2': 50, 'category': 'A', 'target': 'medium'},
    ]
    
    # Synthetic with similar structure but different distributions
    synthetic_data = [
        {'numeric1': 1, 'numeric2': 999999, 'category': 'A', 'target': 'high'},
        {'numeric1': -998, 'numeric2': 0.002, 'category': 'B', 'target': 'low'},
        {'numeric1': 51, 'numeric2': 49, 'category': 'C', 'target': 'medium'},
    ]
    
    return real_data, synthetic_data


def create_identical_datasets(n_samples: int = 50, seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
    """Create two identical datasets for testing perfect fidelity cases."""
    data = generate_tabular_data(n_samples, seed=seed)
    return data, [row.copy() for row in data]


def create_completely_different_datasets(n_samples: int = 50) -> Tuple[List[Dict], List[Dict]]:
    """Create completely different datasets for testing worst-case scenarios."""
    real_data = generate_tabular_data(n_samples, seed=42)
    synthetic_data = generate_tabular_data(n_samples, seed=123)
    
    # Make them even more different by scaling synthetic data
    for row in synthetic_data:
        for key, value in row.items():
            if isinstance(value, (int, float)):
                row[key] = value * 10 + 1000
    
    return real_data, synthetic_data