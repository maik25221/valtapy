"""Input validation for evaluation pipeline."""

import pandas as pd
from typing import Any
from ..domain.entities import EvalPlan
from ..domain.errors import SchemaError


def validate_inputs(real_data: pd.DataFrame, synth_data: pd.DataFrame, 
                   dataset_spec: Any, plan: EvalPlan) -> None:
    """
    Validate inputs for evaluation pipeline.
    
    Args:
        real_data: Real dataset
        synth_data: Synthetic dataset  
        dataset_spec: Dataset specification
        plan: Evaluation plan
        
    Raises:
        SchemaError: If validation fails
    """
    # Check basic data requirements
    _validate_basic_data_requirements(real_data, synth_data)
    
    # Check schema compatibility
    _validate_schema_compatibility(real_data, synth_data)
    
    # Check target column if specified
    if dataset_spec.target:
        _validate_target_column(real_data, synth_data, dataset_spec.target)
    
    # Check utility metrics requirements
    _validate_utility_requirements(real_data, synth_data, plan, dataset_spec)


def _validate_basic_data_requirements(real_data: pd.DataFrame, synth_data: pd.DataFrame) -> None:
    """Validate basic data requirements."""
    if real_data.empty:
        raise SchemaError("Real data cannot be empty")
    
    if synth_data.empty:
        raise SchemaError("Synthetic data cannot be empty")
    
    if len(real_data.columns) == 0:
        raise SchemaError("Real data must have at least one column")
    
    if len(synth_data.columns) == 0:
        raise SchemaError("Synthetic data must have at least one column")
    
    # Check for minimum sample sizes
    if len(real_data) < 10:
        raise SchemaError("Real data must have at least 10 samples")
    
    if len(synth_data) < 10:
        raise SchemaError("Synthetic data must have at least 10 samples")


def _validate_schema_compatibility(real_data: pd.DataFrame, synth_data: pd.DataFrame) -> None:
    """Validate that datasets have compatible schemas."""
    real_columns = set(real_data.columns)
    synth_columns = set(synth_data.columns)
    
    # Check for missing columns
    missing_in_synth = real_columns - synth_columns
    if missing_in_synth:
        raise SchemaError(f"Columns missing in synthetic data: {missing_in_synth}")
    
    extra_in_synth = synth_columns - real_columns
    if extra_in_synth:
        raise SchemaError(f"Extra columns in synthetic data: {extra_in_synth}")
    
    # Check data types compatibility (basic check)
    for col in real_columns:
        real_dtype = real_data[col].dtype
        synth_dtype = synth_data[col].dtype
        
        # Allow some flexibility in numeric types
        if _are_compatible_dtypes(real_dtype, synth_dtype):
            continue
        else:
            raise SchemaError(f"Incompatible dtypes for column {col}: "
                            f"real={real_dtype}, synth={synth_dtype}", column=col)


def _validate_target_column(real_data: pd.DataFrame, synth_data: pd.DataFrame, target: str) -> None:
    """Validate target column for utility metrics."""
    if target not in real_data.columns:
        raise SchemaError(f"Target column '{target}' not found in real data", column=target)
    
    if target not in synth_data.columns:
        raise SchemaError(f"Target column '{target}' not found in synthetic data", column=target)
    
    # Check for sufficient non-null values
    real_non_null = real_data[target].notna().sum()
    synth_non_null = synth_data[target].notna().sum()
    
    if real_non_null < 10:
        raise SchemaError(f"Target column '{target}' has too few non-null values in real data: {real_non_null}",
                         column=target)
    
    if synth_non_null < 10:
        raise SchemaError(f"Target column '{target}' has too few non-null values in synthetic data: {synth_non_null}",
                         column=target)


def _validate_utility_requirements(real_data: pd.DataFrame, synth_data: pd.DataFrame,
                                 plan: EvalPlan, dataset_spec: Any) -> None:
    """Validate requirements for utility metrics."""
    # Check if any utility metrics are requested
    utility_metrics = [m for m in plan.metric_ids if m.startswith("utility.")]
    
    if utility_metrics and not dataset_spec.target:
        raise SchemaError("Target column must be specified for utility metrics")
    
    # Additional checks for specific utility metrics
    for metric_id in utility_metrics:
        if metric_id in ["utility.accuracy"] and dataset_spec.target:
            # Check if target is suitable for classification
            target_col = dataset_spec.target
            unique_values = real_data[target_col].nunique()
            
            # Heuristic: if too many unique values, might not be classification
            if unique_values > min(100, len(real_data) // 10):
                # This is just a warning, not an error
                pass


def _are_compatible_dtypes(dtype1, dtype2) -> bool:
    """Check if two pandas dtypes are compatible for comparison."""
    import numpy as np
    
    # Convert to numpy dtypes for easier comparison
    np_dtype1 = np.dtype(dtype1)
    np_dtype2 = np.dtype(dtype2)
    
    # Same dtype
    if np_dtype1 == np_dtype2:
        return True
    
    # Both numeric
    if (np.issubdtype(np_dtype1, np.number) and np.issubdtype(np_dtype2, np.number)):
        return True
    
    # Both object/string
    if (np_dtype1 == np.object_ or np_dtype1.kind in ['U', 'S']) and \
       (np_dtype2 == np.object_ or np_dtype2.kind in ['U', 'S']):
        return True
    
    # Both datetime
    if (np.issubdtype(np_dtype1, np.datetime64) and np.issubdtype(np_dtype2, np.datetime64)):
        return True
    
    return False