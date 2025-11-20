"""Contracts for the ingestion phase."""

from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    from .entities import DataSource, ReadResult


class DataReader(Protocol):
    """Protocol for reading data sources and converting to DataFrames."""
    
    def can_handle(self, source: "DataSource") -> bool:
        """Check if this reader can handle the given data source format."""
        ...
    
    def read(self, source: "DataSource") -> "ReadResult":
        """
        Read data from source and return ReadResult with DataFrame.
        
        Handles:
        - File not found errors
        - Format/parsing errors  
        - Encoding issues
        - Corrupted file errors
        
        Raises:
        - FileNotFoundError: When source file doesn't exist
        - ValueError: When file format is invalid or corrupted
        - UnicodeDecodeError: When encoding issues occur
        """
        ...


class IngestionOrchestrator(Protocol):
    """Protocol for orchestrating the ingestion process."""
    
    def ingest(
        self, 
        real_source: "DataSource", 
        synth_source: "DataSource"
    ) -> tuple["pd.DataFrame", "pd.DataFrame"]:
        """
        Ingest both real and synthetic datasets.
        
        Returns tuple of (real_df, synth_df).
        """
        ...