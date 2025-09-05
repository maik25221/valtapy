"""Domain-specific exceptions for error handling."""


class ValtaPyV2Error(Exception):
    """Base exception for ValtaPyV2 framework."""
    pass


class ConfigError(ValtaPyV2Error):
    """Raised when configuration is invalid or malformed."""
    
    def __init__(self, message: str, config_path: str | None = None):
        self.config_path = config_path
        full_message = f"Configuration error: {message}"
        if config_path:
            full_message += f" (in {config_path})"
        super().__init__(full_message)


class SchemaError(ValtaPyV2Error):
    """Raised when data schema validation fails."""
    
    def __init__(self, message: str, column: str | None = None):
        self.column = column
        full_message = f"Schema error: {message}"
        if column:
            full_message += f" (column: {column})"
        super().__init__(full_message)


class MetricExecutionError(ValtaPyV2Error):
    """Raised when metric computation fails."""
    
    def __init__(self, message: str, metric_id: str | None = None, original_error: Exception | None = None):
        self.metric_id = metric_id
        self.original_error = original_error
        
        full_message = f"Metric execution error: {message}"
        if metric_id:
            full_message += f" (metric: {metric_id})"
        if original_error:
            full_message += f" (caused by: {type(original_error).__name__}: {original_error})"
            
        super().__init__(full_message)


class RegistryError(ValtaPyV2Error):
    """Raised when registry operations fail."""
    
    def __init__(self, message: str, registry_type: str | None = None, item_id: str | None = None):
        self.registry_type = registry_type
        self.item_id = item_id
        
        full_message = f"Registry error: {message}"
        if registry_type and item_id:
            full_message += f" ({registry_type} registry, item: {item_id})"
        elif registry_type:
            full_message += f" ({registry_type} registry)"
            
        super().__init__(full_message)


class PreprocessingError(ValtaPyV2Error):
    """Raised when data preprocessing fails."""
    
    def __init__(self, message: str, step: str | None = None, original_error: Exception | None = None):
        self.step = step
        self.original_error = original_error
        
        full_message = f"Preprocessing error: {message}"
        if step:
            full_message += f" (step: {step})"
        if original_error:
            full_message += f" (caused by: {type(original_error).__name__}: {original_error})"
            
        super().__init__(full_message)