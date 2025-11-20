"""Basic data preprocessor with common cleaning and preparation operations."""

import pandas as pd
import time
from typing import Any

from ..contracts import DataPreprocessor
from ..entities import PreprocessingConfig, PreprocessingResult, PreprocessingReport
from ..exceptions import (
    IncompatibleDatasetsError,
    PreprocessingFailedError,
    UnsupportedDataTypeError
)


class BasicDataPreprocessor:
    """Basic preprocessor for cleaning and preparing datasets for comparison."""
    
    def can_handle(self, real_df: pd.DataFrame, synth_df: pd.DataFrame) -> bool:
        """Check if this preprocessor can handle the given datasets."""
        # Basic checks - both should be DataFrames with some data
        if not isinstance(real_df, pd.DataFrame) or not isinstance(synth_df, pd.DataFrame):
            return False
        
        if real_df.empty or synth_df.empty:
            return False
            
        return True
    
    def preprocess(
        self,
        real_df: pd.DataFrame,
        synth_df: pd.DataFrame,
        config: PreprocessingConfig
    ) -> PreprocessingResult:
        """
        Preprocess both datasets according to configuration.
        
        NOTE: Many operations are commented out to avoid making decisions
        about data handling. Uncomment and configure as needed.
        """
        start_time = time.time()
        
        # Make copies to avoid modifying originals
        real_processed = real_df.copy()
        synth_processed = synth_df.copy()
        
        # Initialize report tracking
        operations_applied = []
        real_changes = {}
        synth_changes = {}
        compatibility_issues = []
        
        try:
            # 1. BASIC COMPATIBILITY CHECKS
            real_processed, synth_processed, issues = self._check_basic_compatibility(
                real_processed, synth_processed
            )
            compatibility_issues.extend(issues)
            
            # 2. CLEANING OPERATIONS (commented - no decisions made)
            if config.drop_empty_rows or config.drop_empty_columns:
                real_processed, synth_processed, changes = self._clean_empty_data(
                    real_processed, synth_processed, config
                )
                operations_applied.append("empty_data_cleaning")
                real_changes.update(changes.get("real", {}))
                synth_changes.update(changes.get("synth", {}))
            
            # TODO: Uncomment and implement as needed
            # # 3. MISSING VALUE HANDLING
            # if config.na_strategy != "keep":
            #     real_processed, synth_processed = self._handle_missing_values(
            #         real_processed, synth_processed, config
            #     )
            #     operations_applied.append("missing_value_handling")
            
            # # 4. TYPE CONSISTENCY
            # if config.enforce_consistent_types:
            #     real_processed, synth_processed = self._ensure_type_consistency(
            #         real_processed, synth_processed
            #     )
            #     operations_applied.append("type_consistency")
            
            # # 5. NORMALIZATION
            # if config.normalize_numeric or config.normalize_text:
            #     real_processed, synth_processed = self._normalize_data(
            #         real_processed, synth_processed, config
            #     )
            #     operations_applied.append("normalization")
            
            # # 6. ENCODING
            # if config.encode_categorical:
            #     real_processed, synth_processed = self._encode_categorical(
            #         real_processed, synth_processed, config
            #     )
            #     operations_applied.append("categorical_encoding")
            
            # Create report
            report = PreprocessingReport(
                real_changes=real_changes,
                synth_changes=synth_changes,
                compatibility_issues=compatibility_issues,
                operations_applied=operations_applied
            )
            
            processing_time = time.time() - start_time
            
            return PreprocessingResult(
                real_data=real_processed,
                synth_data=synth_processed,
                config=config,
                report=report,
                processing_time=processing_time
            )
            
        except Exception as e:
            raise PreprocessingFailedError("preprocessing", str(e))
    
    def _check_basic_compatibility(
        self,
        real_df: pd.DataFrame,
        synth_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
        """Check and report basic compatibility issues."""
        issues = []
        
        # Check column names
        real_cols = set(real_df.columns)
        synth_cols = set(synth_df.columns)
        
        if real_cols != synth_cols:
            missing_in_synth = real_cols - synth_cols
            missing_in_real = synth_cols - real_cols
            
            if missing_in_synth:
                issues.append(f"Columns missing in synthetic: {missing_in_synth}")
            if missing_in_real:
                issues.append(f"Columns missing in real: {missing_in_real}")
        
        # Check basic shapes
        if real_df.shape[1] != synth_df.shape[1]:
            issues.append(
                f"Different number of columns: real={real_df.shape[1]}, "
                f"synth={synth_df.shape[1]}"
            )
        
        return real_df, synth_df, issues
    
    def _clean_empty_data(
        self,
        real_df: pd.DataFrame,
        synth_df: pd.DataFrame,
        config: PreprocessingConfig
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        """Clean empty rows and columns (minimal implementation)."""
        changes = {"real": {}, "synth": {}}
        
        if config.drop_empty_rows:
            # Only drop rows that are completely empty
            real_before = len(real_df)
            synth_before = len(synth_df)
            
            real_df = real_df.dropna(how='all')
            synth_df = synth_df.dropna(how='all')
            
            changes["real"]["rows_dropped"] = real_before - len(real_df)
            changes["synth"]["rows_dropped"] = synth_before - len(synth_df)
        
        if config.drop_empty_columns:
            # Only drop columns that are completely empty
            real_cols_before = len(real_df.columns)
            synth_cols_before = len(synth_df.columns)
            
            real_df = real_df.dropna(axis=1, how='all')
            synth_df = synth_df.dropna(axis=1, how='all')
            
            changes["real"]["columns_dropped"] = real_cols_before - len(real_df.columns)
            changes["synth"]["columns_dropped"] = synth_cols_before - len(synth_df.columns)
        
        return real_df, synth_df, changes
    
    # TODO: Implement these methods as needed
    # def _handle_missing_values(self, real_df, synth_df, config):
    #     """Handle missing values according to strategy."""
    #     # Strategy: "drop", "fill"
    #     # Implementation depends on business requirements
    #     pass
    
    # def _ensure_type_consistency(self, real_df, synth_df):
    #     """Ensure columns have consistent types between datasets."""
    #     # Align data types between real and synthetic
    #     # Handle numeric/string mismatches
    #     pass
    
    # def _normalize_data(self, real_df, synth_df, config):
    #     """Normalize numeric and text data."""
    #     # Numeric: min-max, z-score, etc.
    #     # Text: lowercase, strip whitespace, etc.
    #     pass
    
    # def _encode_categorical(self, real_df, synth_df, config):
    #     """Encode categorical variables consistently."""
    #     # Label encoding, one-hot encoding, etc.
    #     # Ensure same encoding applied to both datasets
    #     pass