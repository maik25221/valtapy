"""Custom exceptions for the validation phase."""


class ValidationError(Exception):
    """Base exception for all validation-related errors."""
    pass


class DomainRuleViolationError(ValidationError):
    """Raised when critical domain rules are violated."""
    
    def __init__(self, rule_id: str, column_name: str, violation_count: int):
        self.rule_id = rule_id
        self.column_name = column_name
        self.violation_count = violation_count
        super().__init__(
            f"Domain rule '{rule_id}' violated in column '{column_name}': "
            f"{violation_count} violations found"
        )


class CoherenceError(ValidationError):
    """Raised when datasets fail coherence checks."""
    
    def __init__(self, dataset_type: str, issues: list[str]):
        self.dataset_type = dataset_type
        self.issues = issues
        super().__init__(
            f"Coherence errors in {dataset_type} dataset: {'; '.join(issues)}"
        )


class CorrespondenceError(ValidationError):
    """Raised when datasets fail correspondence checks."""
    
    def __init__(self, issues: list[str]):
        self.issues = issues
        super().__init__(
            f"Correspondence errors between datasets: {'; '.join(issues)}"
        )


class InvalidRuleError(ValidationError):
    """Raised when a domain rule is malformed or invalid."""
    
    def __init__(self, rule_id: str, reason: str):
        self.rule_id = rule_id
        self.reason = reason
        super().__init__(f"Invalid rule '{rule_id}': {reason}")


class ParseError(ValidationError):
    """Raised when domain rules cannot be parsed from external format."""
    
    def __init__(self, source: str, format_type: str, reason: str):
        self.source = source
        self.format_type = format_type
        self.reason = reason
        super().__init__(
            f"Failed to parse {format_type} rules from {source}: {reason}"
        )


class UnsupportedFormatError(ValidationError):
    """Raised when trying to parse an unsupported rule format."""
    
    def __init__(self, format_type: str, supported_formats: list[str]):
        self.format_type = format_type
        self.supported_formats = supported_formats
        super().__init__(
            f"Unsupported format '{format_type}'. "
            f"Supported formats: {', '.join(supported_formats)}"
        )


class ValidationConfigurationError(ValidationError):
    """Raised when validation configuration is invalid."""
    
    def __init__(self, parameter: str, value: str, reason: str):
        self.parameter = parameter
        self.value = value
        self.reason = reason
        super().__init__(
            f"Invalid validation configuration for '{parameter}' = '{value}': {reason}"
        )