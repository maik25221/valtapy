"""Custom exceptions for the ingestion phase."""


class IngestionError(Exception):
    """Base exception for all ingestion-related errors."""
    pass


class FileNotFoundError(IngestionError):
    """Raised when a data source file cannot be found."""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        super().__init__(f"Data file not found: {file_path}")


class UnsupportedFormatError(IngestionError):
    """Raised when trying to read an unsupported file format."""
    
    def __init__(self, format_name: str, supported_formats: list[str]):
        self.format_name = format_name
        self.supported_formats = supported_formats
        super().__init__(
            f"Unsupported format '{format_name}'. "
            f"Supported formats: {', '.join(supported_formats)}"
        )


class CorruptedFileError(IngestionError):
    """Raised when a file is corrupted or has invalid format."""
    
    def __init__(self, file_path: str, reason: str):
        self.file_path = file_path
        self.reason = reason
        super().__init__(f"Corrupted file {file_path}: {reason}")


class EncodingError(IngestionError):
    """Raised when file encoding cannot be detected or is invalid."""
    
    def __init__(self, file_path: str, encoding: str, reason: str):
        self.file_path = file_path
        self.encoding = encoding
        self.reason = reason
        super().__init__(
            f"Encoding error in {file_path} with {encoding}: {reason}"
        )


class EmptyFileError(IngestionError):
    """Raised when a data file is empty or has no valid data."""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        super().__init__(f"Data file is empty: {file_path}")


class DatasetMismatchError(IngestionError):
    """Raised when real and synthetic datasets are incompatible."""
    
    def __init__(self, reason: str, real_info: dict, synth_info: dict):
        self.reason = reason
        self.real_info = real_info
        self.synth_info = synth_info
        super().__init__(f"Dataset mismatch: {reason}")