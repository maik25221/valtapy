"""Tests for basic validator implementation."""

import pytest
import pandas as pd
import numpy as np

from valtapy.validation.validators.basic_validator import BasicDataValidator
from valtapy.validation.entities import (
    DomainRule, DomainRuleSet, ValidationResult, RuleType, ValidationSeverity
)
from valtapy.validation.exceptions import ValidationError


class TestBasicDataValidator:
    """Test cases for BasicDataValidator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = BasicDataValidator()
    
    def test_can_handle_valid_dataframes(self):
        """Test that validator can handle valid DataFrames."""
        real_df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
        synth_df = pd.DataFrame({"col1": [3, 4], "col2": ["c", "d"]})
        
        assert self.validator.can_handle(real_df, synth_df) is True
    
    def test_cannot_handle_empty_dataframes(self):
        """Test that validator rejects empty DataFrames."""
        real_df = pd.DataFrame({"col": [1, 2]})
        empty_df = pd.DataFrame()
        
        assert self.validator.can_handle(real_df, empty_df) is False
        assert self.validator.can_handle(empty_df, real_df) is False
    
    def test_cannot_handle_non_dataframes(self):
        """Test that validator rejects non-DataFrame inputs."""
        real_df = pd.DataFrame({"col": [1, 2]})
        
        assert self.validator.can_handle(real_df, "not_a_dataframe") is False
        assert self.validator.can_handle("not_a_dataframe", real_df) is False
    
    def test_validation_with_identical_datasets_no_rules(self):
        """Test validation with identical datasets and no domain rules."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35]
        })
        real_df = df.copy()
        synth_df = df.copy()
        
        result = self.validator.validate(real_df, synth_df)
        
        # Verify result structure
        assert isinstance(result, ValidationResult)
        assert result.real_data.equals(real_df)
        assert result.synth_data.equals(synth_df)
        assert result.rule_set is None
        assert result.is_valid is True
        
        # Should have minimal issues with identical datasets
        assert result.report.error_count == 0
        assert len(result.report.correspondence_issues) == 0
    
    def test_validation_detects_column_mismatches(self):
        """Test that validation detects column mismatches."""
        real_df = pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]})
        synth_df = pd.DataFrame({"id": [1, 2], "age": [25, 30]})  # different columns
        
        result = self.validator.validate(real_df, synth_df)
        
        # Should detect correspondence issues
        assert len(result.report.correspondence_issues) > 0
        correspondence_text = " ".join(result.report.correspondence_issues).lower()
        assert "missing" in correspondence_text or "extra" in correspondence_text
    
    def test_validation_detects_type_mismatches(self):
        """Test that validation detects data type mismatches."""
        real_df = pd.DataFrame({"id": [1, 2, 3], "value": [1.0, 2.0, 3.0]})  # numeric
        synth_df = pd.DataFrame({"id": [1, 2, 3], "value": ["a", "b", "c"]})  # string
        
        result = self.validator.validate(real_df, synth_df)
        
        # Should detect type mismatch
        correspondence_issues = result.report.correspondence_issues
        assert any("type mismatch" in issue.lower() for issue in correspondence_issues)
    
    def test_validation_with_domain_rules_success(self):
        """Test validation with domain rules that pass."""
        real_df = pd.DataFrame({
            "age": [25, 30, 35],
            "gender": ["M", "F", "M"],
            "id": [1, 2, 3]
        })
        synth_df = real_df.copy()
        
        # Create domain rules
        rules = [
            DomainRule.range_rule("age_range", "age", 0, 120),
            DomainRule.categorical_rule("gender_values", "gender", ["M", "F", "O"]),
            DomainRule.not_null_rule("id_not_null", "id")
        ]
        rule_set = DomainRuleSet("test_rules", "Test rules", rules)
        
        result = self.validator.validate(real_df, synth_df, rule_set)
        
        # Should pass validation
        assert result.is_valid is True
        assert result.report.error_count == 0
        assert result.report.total_rules_checked == 3
    
    def test_validation_with_domain_rules_violations(self):
        """Test validation with domain rules that fail."""
        real_df = pd.DataFrame({
            "age": [25, 150, -5],  # 150 and -5 violate range
            "gender": ["M", "F", "X"],  # "X" violates categorical
            "id": [1, None, 3]  # None violates not_null
        })
        synth_df = real_df.copy()
        
        # Create domain rules
        rules = [
            DomainRule.range_rule("age_range", "age", 0, 120),
            DomainRule.categorical_rule("gender_values", "gender", ["M", "F", "O"]),
            DomainRule.not_null_rule("id_not_null", "id")
        ]
        rule_set = DomainRuleSet("test_rules", "Test rules", rules)
        
        result = self.validator.validate(real_df, synth_df, rule_set)
        
        # Should have validation errors
        assert result.is_valid is False
        assert result.report.error_count > 0
        
        # Check specific rule violations
        issues = result.report.issues
        age_issues = [issue for issue in issues if issue.column_name == "age"]
        gender_issues = [issue for issue in issues if issue.column_name == "gender"]
        id_issues = [issue for issue in issues if issue.column_name == "id"]
        
        assert len(age_issues) > 0  # Range violations
        assert len(gender_issues) > 0  # Categorical violations
        assert len(id_issues) > 0  # Not null violations
    
    def test_validation_with_missing_column(self):
        """Test validation when rule references missing column."""
        real_df = pd.DataFrame({"id": [1, 2, 3]})
        synth_df = real_df.copy()
        
        # Rule for non-existent column
        rules = [DomainRule.range_rule("age_range", "age", 0, 120)]
        rule_set = DomainRuleSet("test_rules", "Test rules", rules)
        
        result = self.validator.validate(real_df, synth_df, rule_set)
        
        # Should have issues about missing column
        assert len(result.report.issues) > 0
        missing_col_issues = [
            issue for issue in result.report.issues 
            if "not found" in issue.message.lower()
        ]
        assert len(missing_col_issues) > 0
    
    def test_validation_detects_coherence_issues(self):
        """Test that validation detects coherence issues."""
        # Create DataFrame with coherence issues
        real_df = pd.DataFrame({
            "id": [1, 2, 3],
            "empty_col": [None, None, None],  # Completely null column
        })
        synth_df = real_df.copy()
        
        result = self.validator.validate(real_df, synth_df)
        
        # Should detect coherence issues
        assert len(result.report.coherence_issues) > 0
        coherence_text = " ".join(result.report.coherence_issues).lower()
        assert "null" in coherence_text
    
    def test_validation_handles_non_numeric_range_rule(self):
        """Test validation gracefully handles range rule on non-numeric column."""
        real_df = pd.DataFrame({"name": ["Alice", "Bob", "Charlie"]})
        synth_df = real_df.copy()
        
        # Apply range rule to string column
        rules = [DomainRule.range_rule("name_range", "name", 0, 100)]
        rule_set = DomainRuleSet("test_rules", "Test rules", rules)
        
        result = self.validator.validate(real_df, synth_df, rule_set)
        
        # Should generate warning about non-numeric column
        warnings = [
            issue for issue in result.report.issues 
            if issue.severity == ValidationSeverity.WARNING
        ]
        assert len(warnings) > 0
        assert any("non-numeric" in warning.message.lower() for warning in warnings)
    
    def test_validation_report_contains_processing_time(self):
        """Test that validation report includes processing time."""
        real_df = pd.DataFrame({"col": [1, 2, 3]})
        synth_df = pd.DataFrame({"col": [4, 5, 6]})
        
        result = self.validator.validate(real_df, synth_df)
        
        assert result.report.processing_time > 0
    
    def test_validation_with_different_row_counts(self):
        """Test validation with datasets having different row counts."""
        real_df = pd.DataFrame({"id": [1, 2, 3, 4, 5]})  # 5 rows
        synth_df = pd.DataFrame({"id": [1, 2, 3]})       # 3 rows
        
        result = self.validator.validate(real_df, synth_df)
        
        # Should handle gracefully - different row counts are expected
        assert isinstance(result, ValidationResult)
        # No specific correspondence issue for different row counts