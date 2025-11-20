"""Custom exceptions for the preprocessing phase."""


class PreprocessingError(Exception):
    """Base exception for all preprocessing-related errors."""
    pass


class IncompatibleDatasetsError(PreprocessingError):
    """Raised when datasets are incompatible for preprocessing."""
    
    def __init__(self, reason: str, real_info: dict, synth_info: dict):
        self.reason = reason
        self.real_info = real_info
        self.synth_info = synth_info
        super().__init__(f"Incompatible datasets: {reason}")


class InvalidConfigurationError(PreprocessingError):
    """Raised when preprocessing configuration is invalid."""
    
    def __init__(self, config_field: str, value: str, valid_options: list[str]):
        self.config_field = config_field
        self.value = value
        self.valid_options = valid_options
        super().__init__(
            f"Invalid {config_field}: '{value}'. "
            f"Valid options: {', '.join(valid_options)}"
        )


class PreprocessingFailedError(PreprocessingError):
    """Raised when preprocessing operation fails."""
    
    def __init__(self, operation: str, reason: str):
        self.operation = operation
        self.reason = reason
        super().__init__(f"Preprocessing failed during {operation}: {reason}")


class UnsupportedDataTypeError(PreprocessingError):
    """Raised when data type is not supported for preprocessing."""
    
    def __init__(self, column_name: str, data_type: str, supported_types: list[str]):
        self.column_name = column_name
        self.data_type = data_type
        self.supported_types = supported_types
        super().__init__(
            f"Unsupported data type for column '{column_name}': {data_type}. "
            f"Supported types: {', '.join(supported_types)}"
        )