"""Contracts for the validation phase."""

from typing import Protocol, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    from .entities import DomainRuleSet, ValidationResult, ValidationReport


class DataValidator(Protocol):
    """Protocol for validating datasets against domain rules and coherence checks."""
    
    def validate(
        self,
        real_df: "pd.DataFrame",
        synth_df: "pd.DataFrame",
        domain_rules: Optional["DomainRuleSet"] = None
    ) -> "ValidationResult":
        """
        Validate both datasets for coherence, domain compliance, and correspondence.
        
        Validation includes:
        - Coherence: Internal consistency within each dataset
        - Domain compliance: Adherence to business rules and constraints
        - Correspondence: Compatibility between real and synthetic datasets
        
        Args:
            real_df: Original dataset
            synth_df: Synthetic dataset  
            domain_rules: Optional set of domain rules to validate against
            
        Returns:
            ValidationResult with validation status and detailed report
            
        Raises:
            ValidationError: When validation process fails
        """
        ...
    
    def can_handle(
        self,
        real_df: "pd.DataFrame",
        synth_df: "pd.DataFrame"
    ) -> bool:
        """Check if this validator can handle the given datasets."""
        ...


class DomainRuleValidator(Protocol):
    """Protocol for validating data against specific domain rules."""
    
    def validate_rules(
        self,
        data: "pd.DataFrame",
        rule_set: "DomainRuleSet"
    ) -> list["ValidationIssue"]:
        """
        Validate dataset against a set of domain rules.
        
        Args:
            data: Dataset to validate
            rule_set: Set of domain rules to check
            
        Returns:
            List of validation issues found
        """
        ...


class CoherenceValidator(Protocol):
    """Protocol for validating internal coherence of datasets."""
    
    def validate_coherence(
        self,
        data: "pd.DataFrame"
    ) -> list[str]:
        """
        Check internal coherence of a dataset.
        
        Examples of coherence checks:
        - Logical relationships between columns
        - Data type consistency
        - Value distributions that make sense
        
        Args:
            data: Dataset to check for coherence
            
        Returns:
            List of coherence issues found
        """
        ...


class CorrespondenceValidator(Protocol):
    """Protocol for validating correspondence between real and synthetic datasets."""
    
    def validate_correspondence(
        self,
        real_df: "pd.DataFrame",
        synth_df: "pd.DataFrame"
    ) -> list[str]:
        """
        Check correspondence between real and synthetic datasets.
        
        Examples of correspondence checks:
        - Schema alignment (columns, types)
        - Value ranges compatibility
        - Categorical values consistency
        - Distribution similarity (basic)
        
        Args:
            real_df: Original dataset
            synth_df: Synthetic dataset
            
        Returns:
            List of correspondence issues found
        """
        ...


class ValidationOrchestrator(Protocol):
    """Protocol for orchestrating the complete validation process."""
    
    def validate_datasets(
        self,
        real_df: "pd.DataFrame",
        synth_df: "pd.DataFrame",
        domain_rules: Optional["DomainRuleSet"] = None
    ) -> tuple["pd.DataFrame", "pd.DataFrame"]:
        """
        Orchestrate complete validation of both datasets.
        
        Returns tuple of (validated_real_df, validated_synth_df).
        May raise ValidationError if critical issues are found.
        """
        ...


class DomainRuleParser(Protocol):
    """Protocol for parsing domain rules from various formats."""
    
    def parse_rules(
        self,
        source: str,
        format_type: str
    ) -> "DomainRuleSet":
        """
        Parse domain rules from external source.
        
        Args:
            source: Path to file or string content with rules
            format_type: Format of the rules ("json", "yaml", "xml")
            
        Returns:
            Parsed DomainRuleSet
            
        Raises:
            ParseError: When rules cannot be parsed
        """
        ...
    
    def can_parse(self, format_type: str) -> bool:
        """Check if this parser can handle the given format."""
        ...