"""CSV data reader implementation."""

import pandas as pd
from pathlib import Path
from typing import Any

from ..contracts import DataReader
from ..entities import DataSource, ReadResult
from ..exceptions import (
    FileNotFoundError, 
    CorruptedFileError, 
    EncodingError, 
    EmptyFileError
)


class CSVReader:
    """CSV file reader with robust error handling."""
    
    def can_handle(self, source: DataSource) -> bool:
        """Check if this reader can handle CSV format."""
        return source.format.lower() == 'csv'
    
    def read(self, source: DataSource) -> ReadResult:
        """
        Read CSV file and return ReadResult with DataFrame.
        
        Args:
            source: DataSource with path and read parameters
            
        Returns:
            ReadResult with pandas DataFrame and metadata
            
        Raises:
            FileNotFoundError: When CSV file doesn't exist
            ValueError: When file format is invalid or corrupted
            UnicodeDecodeError: When encoding issues occur
        """
        file_path = source.file_path
        
        # Check file exists
        if not file_path.exists():
            raise FileNotFoundError(source.path)
        
        # Check file is actually a file
        if not file_path.is_file():
            raise CorruptedFileError(source.path, "Path is not a file")
        
        # Prepare read parameters with defaults
        read_params = self._prepare_read_params(source.read_params)
        
        try:
            # Read CSV with pandas
            data = pd.read_csv(file_path, **read_params)
            
            # Extract basic metadata
            metadata = self._extract_metadata(data, file_path)
            
            return ReadResult(
                data=data,
                source=source,
                metadata=metadata
            )
            
        except pd.errors.EmptyDataError:
            raise EmptyFileError(source.path)
        
        except pd.errors.ParserError as e:
            raise CorruptedFileError(source.path, f"CSV parsing error: {str(e)}")
        
        except UnicodeDecodeError as e:
            encoding = read_params.get('encoding', 'unknown')
            raise EncodingError(source.path, encoding, e.reason)
        
        except Exception as e:
            raise CorruptedFileError(source.path, f"Unexpected error: {str(e)}")
    
    def _prepare_read_params(self, user_params: dict[str, Any]) -> dict[str, Any]:
        """Prepare pandas read_csv parameters with sensible defaults."""
        # Default parameters for robust CSV reading
        defaults = {
            'encoding': 'utf-8',
            'na_values': ['', 'NA', 'N/A', 'null', 'NULL', 'None'],
            'keep_default_na': True,
            'skipinitialspace': True,
        }
        
        # Merge user params over defaults
        params = {**defaults, **user_params}
        
        return params
    
    def _extract_metadata(self, data: pd.DataFrame, file_path: Path) -> dict[str, Any]:
        """Extract basic metadata from DataFrame and file."""
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        
        return {
            'file_size_mb': round(file_size_mb, 2),
            'shape': data.shape,
            'dtypes': data.dtypes.to_dict(),
            'columns': data.columns.tolist(),
            'memory_usage_mb': round(data.memory_usage(deep=True).sum() / (1024 * 1024), 2)
        }