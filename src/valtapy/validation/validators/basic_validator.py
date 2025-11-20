"""Basic validator implementation for datasets and domain rules."""

import pandas as pd
import time
from typing import Optional

from ..contracts import DataValidator
from ..entities import (
    DomainRuleSet, ValidationResult, ValidationReport, ValidationIssue,
    RuleType, ValidationSeverity
)
from ..exceptions import ValidationError, InvalidRuleError


class BasicDataValidator:
    """Basic validator for coherence, domain rules, and correspondence checks."""
    
    def can_handle(self, real_df: pd.DataFrame, synth_df: pd.DataFrame) -> bool:
        """Check if this validator can handle the given datasets."""
        if not isinstance(real_df, pd.DataFrame) or not isinstance(synth_df, pd.DataFrame):
            return False
        
        if real_df.empty or synth_df.empty:
            return False
            
        return True
    
    def validate(
        self,
        real_df: pd.DataFrame,
        synth_df: pd.DataFrame,
        domain_rules: Optional[DomainRuleSet] = None
    ) -> ValidationResult:
        """
        Validate both datasets for coherence, domain compliance, and correspondence.
        
        NOTE: Many validation checks are implemented as basic examples.
        Extend as needed for specific domain requirements.
        """
        start_time = time.time()
        
        try:
            # Initialize tracking
            all_issues = []
            coherence_issues = []
            correspondence_issues = []
            total_rules_checked = 0
            
            # 1. CORRESPONDENCE CHECKS (between datasets)
            correspondence_issues = self._check_correspondence(real_df, synth_df)
            
            # 2. COHERENCE CHECKS (internal consistency)
            real_coherence = self._check_coherence(real_df, "real")
            synth_coherence = self._check_coherence(synth_df, "synthetic")
            coherence_issues = real_coherence + synth_coherence
            
            # 3. DOMAIN RULE VALIDATION (if rules provided)
            if domain_rules:
                total_rules_checked = len(domain_rules.rules)
                
                # Validate real dataset against domain rules
                real_issues = self._validate_domain_rules(real_df, domain_rules, "real")
                all_issues.extend(real_issues)
                
                # Validate synthetic dataset against domain rules
                synth_issues = self._validate_domain_rules(synth_df, domain_rules, "synthetic")
                all_issues.extend(synth_issues)
            
            # Create validation report
            processing_time = time.time() - start_time
            report = ValidationReport(
                rule_set_name=domain_rules.name if domain_rules else "no_rules",
                total_rules_checked=total_rules_checked,
                issues=all_issues,
                coherence_issues=coherence_issues,
                correspondence_issues=correspondence_issues,
                processing_time=processing_time
            )
            
            # Determine if validation passed (no critical errors)
            is_valid = not report.has_errors
            
            return ValidationResult(
                real_data=real_df,
                synth_data=synth_df,
                rule_set=domain_rules,
                report=report,
                is_valid=is_valid
            )
            
        except Exception as e:
            raise ValidationError(f"Validation failed: {str(e)}")
    
    def _check_correspondence(
        self,
        real_df: pd.DataFrame,
        synth_df: pd.DataFrame
    ) -> list[str]:
        """Check correspondence between real and synthetic datasets."""
        issues = []
        
        # Check schema alignment
        real_cols = set(real_df.columns)
        synth_cols = set(synth_df.columns)
        
        if real_cols != synth_cols:
            missing_in_synth = real_cols - synth_cols
            extra_in_synth = synth_cols - real_cols
            
            if missing_in_synth:
                issues.append(f"Columns missing in synthetic dataset: {missing_in_synth}")
            if extra_in_synth:
                issues.append(f"Extra columns in synthetic dataset: {extra_in_synth}")
        
        # Check data type compatibility for common columns
        common_cols = real_cols & synth_cols
        for col in common_cols:
            real_dtype = real_df[col].dtype
            synth_dtype = synth_df[col].dtype
            
            # Basic type family check (numeric vs object)
            if self._get_type_family(real_dtype) != self._get_type_family(synth_dtype):
                issues.append(
                    f"Type mismatch in column '{col}': "
                    f"real={real_dtype}, synthetic={synth_dtype}"
                )
        
        # TODO: Add more correspondence checks as needed:
        # - Value range compatibility
        # - Categorical value alignment
        # - Basic distribution similarity
        
        return issues
    
    def _check_coherence(self, data: pd.DataFrame, dataset_type: str) -> list[str]:
        """Check internal coherence of a dataset."""
        issues = []
        
        # Check for completely null columns
        null_cols = data.columns[data.isnull().all()].tolist()
        if null_cols:
            issues.append(f"Completely null columns in {dataset_type}: {null_cols}")
        
        # Check for duplicate column names
        if len(data.columns) != len(set(data.columns)):
            duplicates = data.columns[data.columns.duplicated()].tolist()
            issues.append(f"Duplicate column names in {dataset_type}: {duplicates}")
        
        # TODO: Add more coherence checks as needed:
        # - Logical relationships between columns
        # - Date/time consistency
        # - Value distribution sanity checks
        
        return issues
    
    def _validate_domain_rules(
        self,
        data: pd.DataFrame,
        rule_set: DomainRuleSet,
        dataset_type: str
    ) -> list[ValidationIssue]:
        """Validate dataset against domain rules."""
        issues = []
        
        for rule in rule_set.rules:
            try:
                rule_issues = self._validate_single_rule(data, rule, dataset_type)
                issues.extend(rule_issues)
            except Exception as e:
                # If rule validation fails, create an error issue
                issues.append(ValidationIssue(
                    rule_id=rule.rule_id,
                    column_name=rule.column_name,
                    severity=ValidationSeverity.ERROR,
                    message=f"Rule validation failed: {str(e)}",
                    dataset_type=dataset_type
                ))
        
        return issues
    
    def _validate_single_rule(
        self,
        data: pd.DataFrame,
        rule: "DomainRule",
        dataset_type: str
    ) -> list[ValidationIssue]:
        """Validate a single domain rule against the dataset."""
        issues = []
        
        # Check if column exists
        if rule.column_name not in data.columns:
            return [ValidationIssue(
                rule_id=rule.rule_id,
                column_name=rule.column_name,
                severity=rule.severity,
                message=f"Column '{rule.column_name}' not found in dataset",
                dataset_type=dataset_type
            )]
        
        column_data = data[rule.column_name]
        
        # Validate based on rule type
        if rule.rule_type == RuleType.RANGE:
            issues.extend(self._validate_range_rule(column_data, rule, dataset_type))
        elif rule.rule_type == RuleType.CATEGORICAL:
            issues.extend(self._validate_categorical_rule(column_data, rule, dataset_type))
        elif rule.rule_type == RuleType.NOT_NULL:
            issues.extend(self._validate_not_null_rule(column_data, rule, dataset_type))
        # TODO: Add other rule types as needed
        
        return issues
    
    def _validate_range_rule(
        self,
        column_data: pd.Series,
        rule: "DomainRule",
        dataset_type: str
    ) -> list[ValidationIssue]:
        """Validate numeric range rule."""
        issues = []
        
        # Skip non-numeric data
        if not pd.api.types.is_numeric_dtype(column_data):
            return [ValidationIssue(
                rule_id=rule.rule_id,
                column_name=rule.column_name,
                severity=ValidationSeverity.WARNING,
                message=f"Range rule applied to non-numeric column",
                dataset_type=dataset_type
            )]
        
        # Check range violations
        mask = pd.Series([True] * len(column_data))
        
        if "min_value" in rule.parameters:
            mask &= (column_data >= rule.parameters["min_value"]) | column_data.isna()
        
        if "max_value" in rule.parameters:
            mask &= (column_data <= rule.parameters["max_value"]) | column_data.isna()
        
        violations = ~mask
        if violations.any():
            violation_count = violations.sum()
            violated_indices = column_data[violations].index.tolist()
            violated_values = column_data[violations].tolist()
            
            issues.append(ValidationIssue(
                rule_id=rule.rule_id,
                column_name=rule.column_name,
                severity=rule.severity,
                message=f"Range violation: {violation_count} values outside allowed range",
                affected_rows=violated_indices,
                affected_values=violated_values,
                dataset_type=dataset_type
            ))
        
        return issues
    
    def _validate_categorical_rule(
        self,
        column_data: pd.Series,
        rule: "DomainRule",
        dataset_type: str
    ) -> list[ValidationIssue]:
        """Validate categorical values rule."""
        issues = []
        
        allowed_values = set(rule.parameters["allowed_values"])
        actual_values = set(column_data.dropna().unique())
        
        invalid_values = actual_values - allowed_values
        if invalid_values:
            # Find rows with invalid values
            mask = column_data.isin(invalid_values)
            violated_indices = column_data[mask].index.tolist()
            violated_values = column_data[mask].tolist()
            
            issues.append(ValidationIssue(
                rule_id=rule.rule_id,
                column_name=rule.column_name,
                severity=rule.severity,
                message=f"Invalid categorical values: {invalid_values}",
                affected_rows=violated_indices,
                affected_values=violated_values,
                dataset_type=dataset_type
            ))
        
        return issues
    
    def _validate_not_null_rule(
        self,
        column_data: pd.Series,
        rule: "DomainRule",
        dataset_type: str
    ) -> list[ValidationIssue]:
        """Validate not-null rule."""
        issues = []
        
        null_mask = column_data.isna()
        if null_mask.any():
            null_count = null_mask.sum()
            null_indices = column_data[null_mask].index.tolist()
            
            issues.append(ValidationIssue(
                rule_id=rule.rule_id,
                column_name=rule.column_name,
                severity=rule.severity,
                message=f"Null values found: {null_count} null values",
                affected_rows=null_indices,
                dataset_type=dataset_type
            ))
        
        return issues
    
    def _get_type_family(self, dtype) -> str:
        """Get broad type family for compatibility checking."""
        if pd.api.types.is_numeric_dtype(dtype):
            return "numeric"
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            return "datetime"
        elif pd.api.types.is_bool_dtype(dtype):
            return "boolean"
        else:
            return "object"