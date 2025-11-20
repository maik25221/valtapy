"""Contracts for the preprocessing phase."""

from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    from .entities import PreprocessingConfig, PreprocessingResult


class DataPreprocessor(Protocol):
    """Protocol for preprocessing data to ensure comparability."""
    
    def preprocess(
        self,
        real_df: "pd.DataFrame",
        synth_df: "pd.DataFrame",
        config: "PreprocessingConfig"
    ) -> "PreprocessingResult":
        """
        Clean, type, normalize and encode both datasets for comparability.
        
        Operations include:
        - Cleaning: Handle missing values, empty rows/columns
        - Typing: Ensure consistent data types between datasets
        - Normalization: Standardize numeric ranges, text formats
        - Encoding: Convert categorical data to standard formats
        
        Args:
            real_df: Original dataset
            synth_df: Synthetic dataset
            config: Preprocessing configuration
            
        Returns:
            PreprocessingResult with cleaned datasets and report
            
        Raises:
            ValueError: When datasets are incompatible for preprocessing
        """
        ...
    
    def can_handle(
        self,
        real_df: "pd.DataFrame",
        synth_df: "pd.DataFrame"
    ) -> bool:
        """Check if this preprocessor can handle the given datasets."""
        ...


class PreprocessingOrchestrator(Protocol):
    """Protocol for orchestrating the preprocessing process."""
    
    def preprocess_datasets(
        self,
        real_df: "pd.DataFrame",
        synth_df: "pd.DataFrame",
        config: "PreprocessingConfig"
    ) -> tuple["pd.DataFrame", "pd.DataFrame"]:
        """
        Orchestrate complete preprocessing of both datasets.
        
        Returns tuple of (processed_real_df, processed_synth_df).
        """
        ...