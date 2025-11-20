"""Entities for the preprocessing phase."""

from dataclasses import dataclass, field
from typing import Any, Optional
from pathlib import Path


@dataclass(frozen=True)
class PreprocessingConfig:
    """Configuration for the preprocessing process."""
    
    # Cleaning configuration
    drop_empty_rows: bool = True
    drop_empty_columns: bool = True
    na_strategy: str = "keep"  # "keep", "drop", "fill"
    na_fill_value: Any = None
    
    # Typing configuration
    enforce_consistent_types: bool = True
    auto_detect_types: bool = True
    
    # Normalization configuration
    normalize_numeric: bool = False  # Future: min-max, z-score, etc.
    normalize_text: bool = False     # Future: lowercase, strip, etc.
    
    # Encoding configuration
    encode_categorical: bool = False  # Future: label encoding, one-hot, etc.
    
    def __post_init__(self) -> None:
        valid_strategies = {"keep", "drop", "fill"}
        if self.na_strategy not in valid_strategies:
            raise ValueError(f"na_strategy must be one of {valid_strategies}")


@dataclass(frozen=True)
class PreprocessingReport:
    """Report of preprocessing operations applied."""
    
    real_changes: dict[str, Any] = field(default_factory=dict)
    synth_changes: dict[str, Any] = field(default_factory=dict)
    compatibility_issues: list[str] = field(default_factory=list)
    operations_applied: list[str] = field(default_factory=list)
    
    def __post_init__(self) -> None:
        if not isinstance(self.real_changes, dict):
            raise ValueError("real_changes must be a dictionary")
        if not isinstance(self.synth_changes, dict):
            raise ValueError("synth_changes must be a dictionary")


@dataclass(frozen=True)
class PreprocessingResult:
    """Result of the preprocessing process."""
    
    real_data: Any  # Will be pd.DataFrame in practice
    synth_data: Any  # Will be pd.DataFrame in practice
    config: PreprocessingConfig
    report: PreprocessingReport
    processing_time: float = 0.0
    
    def __post_init__(self) -> None:
        if self.real_data is None:
            raise ValueError("real_data cannot be None")
        if self.synth_data is None:
            raise ValueError("synth_data cannot be None")