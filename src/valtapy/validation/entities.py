"""Entities for the validation phase."""

from dataclasses import dataclass, field
from typing import Any, Optional, Literal, Union
from enum import Enum


class RuleType(Enum):
    """Types of domain rules that can be applied."""
    
    RANGE = "range"                    # Numeric ranges (min/max)
    CATEGORICAL = "categorical"        # Allowed categorical values
    FORMAT = "format"                  # String format patterns (regex, email, etc.)
    UNIQUENESS = "uniqueness"          # Unique values constraint
    NOT_NULL = "not_null"             # Non-null constraint
    RELATIONSHIP = "relationship"      # Inter-column relationships


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    
    ERROR = "error"       # Critical issues that prevent evaluation
    WARNING = "warning"   # Issues that should be noted but don't block evaluation
    INFO = "info"        # Informational notices


@dataclass(frozen=True)
class DomainRule:
    """Represents a single domain rule that data should satisfy."""
    
    rule_id: str
    column_name: str
    rule_type: RuleType
    parameters: dict[str, Any]
    description: str = ""
    severity: ValidationSeverity = ValidationSeverity.ERROR
    
    def __post_init__(self) -> None:
        if not self.rule_id:
            raise ValueError("rule_id cannot be empty")
        if not self.column_name:
            raise ValueError("column_name cannot be empty")
        if not isinstance(self.rule_type, RuleType):
            raise ValueError("rule_type must be a RuleType enum")
    
    @classmethod
    def range_rule(
        cls, 
        rule_id: str, 
        column_name: str, 
        min_value: Optional[float] = None, 
        max_value: Optional[float] = None,
        description: str = "",
        severity: ValidationSeverity = ValidationSeverity.ERROR
    ) -> "DomainRule":
        """Create a numeric range rule."""
        parameters = {}
        if min_value is not None:
            parameters["min_value"] = min_value
        if max_value is not None:
            parameters["max_value"] = max_value
            
        return cls(
            rule_id=rule_id,
            column_name=column_name,
            rule_type=RuleType.RANGE,
            parameters=parameters,
            description=description or f"{column_name} must be in range [{min_value}, {max_value}]",
            severity=severity
        )
    
    @classmethod
    def categorical_rule(
        cls,
        rule_id: str,
        column_name: str,
        allowed_values: list[Any],
        description: str = "",
        severity: ValidationSeverity = ValidationSeverity.ERROR
    ) -> "DomainRule":
        """Create a categorical values rule."""
        return cls(
            rule_id=rule_id,
            column_name=column_name,
            rule_type=RuleType.CATEGORICAL,
            parameters={"allowed_values": allowed_values},
            description=description or f"{column_name} must be one of {allowed_values}",
            severity=severity
        )
    
    @classmethod
    def not_null_rule(
        cls,
        rule_id: str,
        column_name: str,
        description: str = "",
        severity: ValidationSeverity = ValidationSeverity.ERROR
    ) -> "DomainRule":
        """Create a not-null rule."""
        return cls(
            rule_id=rule_id,
            column_name=column_name,
            rule_type=RuleType.NOT_NULL,
            parameters={},
            description=description or f"{column_name} cannot be null",
            severity=severity
        )


@dataclass(frozen=True)
class DomainRuleSet:
    """Collection of domain rules with metadata."""
    
    name: str
    description: str
    rules: list[DomainRule]
    version: str = "1.0"
    source_format: str = "python"  # "json", "yaml", "xml", etc.
    
    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name cannot be empty")
        if not isinstance(self.rules, list):
            raise ValueError("rules must be a list")
    
    def get_rules_for_column(self, column_name: str) -> list[DomainRule]:
        """Get all rules that apply to a specific column."""
        return [rule for rule in self.rules if rule.column_name == column_name]
    
    def get_rules_by_type(self, rule_type: RuleType) -> list[DomainRule]:
        """Get all rules of a specific type."""
        return [rule for rule in self.rules if rule.rule_type == rule_type]


@dataclass(frozen=True)
class ValidationIssue:
    """Represents a single validation issue found."""
    
    rule_id: str
    column_name: str
    severity: ValidationSeverity
    message: str
    affected_rows: Optional[list[int]] = None
    affected_values: Optional[list[Any]] = None
    dataset_type: Literal["real", "synthetic"] = "real"
    
    def __post_init__(self) -> None:
        if not self.rule_id:
            raise ValueError("rule_id cannot be empty")
        if not self.message:
            raise ValueError("message cannot be empty")


@dataclass(frozen=True)
class ValidationReport:
    """Report of all validation results."""
    
    rule_set_name: str
    total_rules_checked: int
    issues: list[ValidationIssue] = field(default_factory=list)
    coherence_issues: list[str] = field(default_factory=list)
    correspondence_issues: list[str] = field(default_factory=list)
    processing_time: float = 0.0
    
    def __post_init__(self) -> None:
        if not self.rule_set_name:
            raise ValueError("rule_set_name cannot be empty")
        if self.total_rules_checked < 0:
            raise ValueError("total_rules_checked cannot be negative")
    
    @property
    def error_count(self) -> int:
        """Count of error-level issues."""
        return len([issue for issue in self.issues if issue.severity == ValidationSeverity.ERROR])
    
    @property
    def warning_count(self) -> int:
        """Count of warning-level issues."""
        return len([issue for issue in self.issues if issue.severity == ValidationSeverity.WARNING])
    
    @property
    def has_errors(self) -> bool:
        """Whether there are any error-level issues."""
        return self.error_count > 0
    
    def get_issues_for_column(self, column_name: str) -> list[ValidationIssue]:
        """Get all issues for a specific column."""
        return [issue for issue in self.issues if issue.column_name == column_name]


@dataclass(frozen=True)
class ValidationResult:
    """Result of the complete validation process."""
    
    real_data: Any  # Will be pd.DataFrame in practice
    synth_data: Any  # Will be pd.DataFrame in practice
    rule_set: Optional[DomainRuleSet]
    report: ValidationReport
    is_valid: bool = True
    
    def __post_init__(self) -> None:
        if self.real_data is None:
            raise ValueError("real_data cannot be None")
        if self.synth_data is None:
            raise ValueError("synth_data cannot be None")