"""Entities for the ingestion phase."""

from dataclasses import dataclass, field
from typing import Any, Optional
from pathlib import Path


@dataclass(frozen=True)
class DataSource:
    """Represents a data source for ingestion."""
    
    path: str
    format: str  # 'csv', 'parquet', 'json', etc.
    read_params: dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        if not self.path:
            raise ValueError("Path cannot be empty")
        if not self.format:
            raise ValueError("Format cannot be empty")
    
    @property
    def file_path(self) -> Path:
        """Get Path object for the file."""
        return Path(self.path)


@dataclass(frozen=True)
class ReadResult:
    """Result of reading a data source."""
    
    data: Any  # Will be pd.DataFrame in practice
    source: DataSource
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        if self.data is None:
            raise ValueError("Data cannot be None")


@dataclass(frozen=True)
class IngestionResult:
    """Result of the complete ingestion process."""
    
    real_data: ReadResult
    synth_data: ReadResult
    processing_time: float = 0.0
    issues: list[str] = field(default_factory=list)