"""Schema validation and alignment utilities."""

import pandas as pd
from typing import Tuple, Optional, Dict, Any
from ...domain.entities import DatasetSpec
from ...domain.errors import SchemaError


def align_schema(real_data: pd.DataFrame, synth_data: pd.DataFrame, 
                spec: DatasetSpec) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.Series]]:
    """
    Align schemas of real and synthetic datasets according to specification.
    
    Current implementation is a passthrough placeholder.
    TODO: Implement proper schema alignment including:
    - Column type casting and harmonization
    - Missing value handling standardization
    - Feature encoding for categorical variables
    - Target column extraction and validation
    
    Args:
        real_data: Real dataset
        synth_data: Synthetic dataset
        spec: Dataset specification with target, dtypes, constraints
        
    Returns:
        Tuple of (aligned_real_data, aligned_synth_data, target_column)
        
    Raises:
        SchemaError: If alignment fails
    """
    # TODO: Implement proper alignment logic

    target_column = None
    if spec.target:
        if spec.target not in real_data.columns:
            raise SchemaError(f"Target column '{spec.target}' not found in real data")
        if spec.target not in synth_data.columns:
            raise SchemaError(f"Target column '{spec.target}' not found in synthetic data")
        
        target_column = real_data[spec.target].copy()
    
    return real_data, synth_data, target_column


def infer_column_types(df: pd.DataFrame) -> Dict[str, str]:
    """
    Infer semantic column types (numeric, categorical, datetime, etc.).
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary mapping column names to semantic types
    """
    column_types = {}
    
    for col in df.columns:
        dtype = df[col].dtype
        
        if pd.api.types.is_datetime64_any_dtype(dtype):
            column_types[col] = "datetime"

        elif pd.api.types.is_numeric_dtype(dtype):
            unique_count = df[col].nunique()
            total_count = len(df[col].dropna())

            if unique_count <= min(10, total_count * 0.05):
                column_types[col] = "categorical_numeric"
            else:
                column_types[col] = "numeric"

        elif pd.api.types.is_bool_dtype(dtype):
            column_types[col] = "boolean"

        else:
            unique_count = df[col].nunique()
            total_count = len(df[col].dropna())

            if unique_count > total_count * 0.5:
                column_types[col] = "text"
            else:
                column_types[col] = "categorical"
    
    return column_types


def validate_schema_compatibility(real_data: pd.DataFrame, synth_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate that two datasets have compatible schemas.
    
    Args:
        real_data: Real dataset
        synth_data: Synthetic dataset
        
    Returns:
        Validation report with compatibility status and issues
    """
    report = {
        "compatible": True,
        "issues": [],
        "warnings": []
    }
    
    real_cols = set(real_data.columns)
    synth_cols = set(synth_data.columns)
    
    missing_in_synth = real_cols - synth_cols
    if missing_in_synth:
        report["compatible"] = False
        report["issues"].append(f"Columns missing in synthetic: {missing_in_synth}")
    
    extra_in_synth = synth_cols - real_cols
    if extra_in_synth:
        report["warnings"].append(f"Extra columns in synthetic: {extra_in_synth}")
    
    common_cols = real_cols.intersection(synth_cols)
    
    for col in common_cols:
        real_dtype = real_data[col].dtype
        synth_dtype = synth_data[col].dtype
        
        if not _are_compatible_types(real_dtype, synth_dtype):
            report["issues"].append(f"Incompatible types for {col}: {real_dtype} vs {synth_dtype}")
            report["compatible"] = False

        if pd.api.types.is_numeric_dtype(real_dtype) and pd.api.types.is_numeric_dtype(synth_dtype):
            real_range = (real_data[col].min(), real_data[col].max())
            synth_range = (synth_data[col].min(), synth_data[col].max())

            if (synth_range[1] - synth_range[0]) > 2 * (real_range[1] - real_range[0]):
                report["warnings"].append(f"Synthetic range much larger for {col}")
    
    return report


def standardize_missing_values(df: pd.DataFrame, strategy: str = "pandas") -> pd.DataFrame:
    """
    Standardize missing value representation across dataset.
    
    Args:
        df: Input DataFrame
        strategy: Missing value strategy ("pandas", "zero", "mode")
        
    Returns:
        DataFrame with standardized missing values
    """
    df_clean = df.copy()
    
    if strategy == "pandas":
        pass
    elif strategy == "zero":
        for col in df_clean.columns:
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col] = df_clean[col].fillna(0)
            else:
                df_clean[col] = df_clean[col].fillna("missing")
    elif strategy == "mode":
        for col in df_clean.columns:
            mode_val = df_clean[col].mode()
            if len(mode_val) > 0:
                df_clean[col] = df_clean[col].fillna(mode_val[0])
    
    return df_clean


def _are_compatible_types(dtype1, dtype2) -> bool:
    """Check if two pandas dtypes are compatible."""
    import numpy as np
    
    np_dtype1 = np.dtype(dtype1)
    np_dtype2 = np.dtype(dtype2)

    if np_dtype1 == np_dtype2:
        return True

    if np.issubdtype(np_dtype1, np.number) and np.issubdtype(np_dtype2, np.number):
        return True

    if (np_dtype1 == np.object_ or np_dtype1.kind in ['U', 'S']) and (
        np_dtype2 == np.object_ or np_dtype2.kind in ['U', 'S']
    ):
        return True

    if np.issubdtype(np_dtype1, np.datetime64) and np.issubdtype(np_dtype2, np.datetime64):
        return True

    return False