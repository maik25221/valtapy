"""Custom exceptions for the evaluation phase."""


class EvaluationError(Exception):
    """Base exception for all evaluation-related errors."""
    pass


class MetricNotFoundError(EvaluationError):
    """Raised when trying to use a metric that is not registered."""
    
    def __init__(self, metric_id: str, available_metrics: list[str]):
        self.metric_id = metric_id
        self.available_metrics = available_metrics
        super().__init__(
            f"Metric '{metric_id}' not found. "
            f"Available metrics: {', '.join(available_metrics)}"
        )


class MetricComputationError(EvaluationError):
    """Raised when metric computation fails."""
    
    def __init__(self, metric_id: str, reason: str):
        self.metric_id = metric_id
        self.reason = reason
        super().__init__(f"Metric '{metric_id}' computation failed: {reason}")


class InvalidParametersError(EvaluationError):
    """Raised when metric parameters are invalid."""
    
    def __init__(self, metric_id: str, parameter: str, reason: str):
        self.metric_id = metric_id
        self.parameter = parameter
        self.reason = reason
        super().__init__(
            f"Invalid parameter '{parameter}' for metric '{metric_id}': {reason}"
        )


class UnsupportedDataError(EvaluationError):
    """Raised when metric cannot handle the provided data."""
    
    def __init__(self, metric_id: str, reason: str):
        self.metric_id = metric_id
        self.reason = reason
        super().__init__(f"Metric '{metric_id}' cannot handle data: {reason}")


class MetricRegistrationError(EvaluationError):
    """Raised when metric registration fails."""
    
    def __init__(self, metric_id: str, reason: str):
        self.metric_id = metric_id
        self.reason = reason
        super().__init__(f"Failed to register metric '{metric_id}': {reason}")


class EvaluationConfigurationError(EvaluationError):
    """Raised when evaluation configuration is invalid."""
    
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(f"Invalid evaluation configuration: {reason}")


class CacheError(EvaluationError):
    """Raised when metric caching operations fail."""
    
    def __init__(self, operation: str, reason: str):
        self.operation = operation
        self.reason = reason
        super().__init__(f"Cache {operation} failed: {reason}")