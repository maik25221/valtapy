"""Tests for validation entities."""

import pytest
import pandas as pd

from valtapy.validation.entities import (
    RuleType, ValidationSeverity, DomainRule, DomainRuleSet,
    ValidationIssue, ValidationReport, ValidationResult
)


class TestRuleType:
    """Test cases for RuleType enum."""
    
    def test_rule_type_values(self):
        """Test that all expected rule types exist."""
        expected_types = [
            "range", "categorical", "format", "uniqueness", "not_null", "relationship"
        ]
        
        for expected in expected_types:
            assert any(rule_type.value == expected for rule_type in RuleType)


class TestValidationSeverity:
    """Test cases for ValidationSeverity enum."""
    
    def test_severity_values(self):
        """Test that all expected severity levels exist."""
        expected_severities = ["error", "warning", "info"]
        
        for expected in expected_severities:
            assert any(severity.value == expected for severity in ValidationSeverity)


class TestDomainRule:
    """Test cases for DomainRule entity."""
    
    def test_basic_rule_creation(self):
        """Test creating a basic domain rule."""
        rule = DomainRule(
            rule_id="test_rule",
            column_name="age",
            rule_type=RuleType.RANGE,
            parameters={"min_value": 0, "max_value": 120},
            description="Age must be reasonable"
        )
        
        assert rule.rule_id == "test_rule"
        assert rule.column_name == "age"
        assert rule.rule_type == RuleType.RANGE
        assert rule.parameters == {"min_value": 0, "max_value": 120}
        assert rule.description == "Age must be reasonable"
        assert rule.severity == ValidationSeverity.ERROR  # default
    
    def test_range_rule_factory(self):
        """Test creating range rule via factory method."""
        rule = DomainRule.range_rule(
            rule_id="age_range",
            column_name="age",
            min_value=0,
            max_value=120
        )
        
        assert rule.rule_id == "age_range"
        assert rule.column_name == "age"
        assert rule.rule_type == RuleType.RANGE
        assert rule.parameters["min_value"] == 0
        assert rule.parameters["max_value"] == 120
        assert "age must be in range [0, 120]" in rule.description.lower()
    
    def test_categorical_rule_factory(self):
        """Test creating categorical rule via factory method."""
        allowed_values = ["M", "F", "O"]
        rule = DomainRule.categorical_rule(
            rule_id="gender_values",
            column_name="gender",
            allowed_values=allowed_values
        )
        
        assert rule.rule_id == "gender_values"
        assert rule.column_name == "gender"
        assert rule.rule_type == RuleType.CATEGORICAL
        assert rule.parameters["allowed_values"] == allowed_values
    
    def test_not_null_rule_factory(self):
        """Test creating not-null rule via factory method."""
        rule = DomainRule.not_null_rule(
            rule_id="id_not_null",
            column_name="id"
        )
        
        assert rule.rule_id == "id_not_null"
        assert rule.column_name == "id"
        assert rule.rule_type == RuleType.NOT_NULL
        assert rule.parameters == {}
    
    def test_empty_rule_id_raises_error(self):
        """Test that empty rule_id raises ValueError."""
        with pytest.raises(ValueError, match="rule_id cannot be empty"):
            DomainRule(
                rule_id="",
                column_name="test",
                rule_type=RuleType.RANGE,
                parameters={}
            )
    
    def test_empty_column_name_raises_error(self):
        """Test that empty column_name raises ValueError."""
        with pytest.raises(ValueError, match="column_name cannot be empty"):
            DomainRule(
                rule_id="test",
                column_name="",
                rule_type=RuleType.RANGE,
                parameters={}
            )


class TestDomainRuleSet:
    """Test cases for DomainRuleSet entity."""
    
    def test_basic_rule_set_creation(self):
        """Test creating a basic rule set."""
        rules = [
            DomainRule.range_rule("age_rule", "age", 0, 120),
            DomainRule.not_null_rule("id_rule", "id")
        ]
        
        rule_set = DomainRuleSet(
            name="user_validation",
            description="Validation rules for user data",
            rules=rules
        )
        
        assert rule_set.name == "user_validation"
        assert rule_set.description == "Validation rules for user data"
        assert len(rule_set.rules) == 2
        assert rule_set.version == "1.0"  # default
        assert rule_set.source_format == "python"  # default
    
    def test_get_rules_for_column(self):
        """Test filtering rules by column name."""
        rules = [
            DomainRule.range_rule("age_min", "age", min_value=0),
            DomainRule.range_rule("age_max", "age", max_value=120),
            DomainRule.not_null_rule("id_rule", "id")
        ]
        
        rule_set = DomainRuleSet("test", "test", rules)
        age_rules = rule_set.get_rules_for_column("age")
        
        assert len(age_rules) == 2
        assert all(rule.column_name == "age" for rule in age_rules)
    
    def test_get_rules_by_type(self):
        """Test filtering rules by type."""
        rules = [
            DomainRule.range_rule("age_rule", "age", 0, 120),
            DomainRule.categorical_rule("gender_rule", "gender", ["M", "F"]),
            DomainRule.not_null_rule("id_rule", "id")
        ]
        
        rule_set = DomainRuleSet("test", "test", rules)
        range_rules = rule_set.get_rules_by_type(RuleType.RANGE)
        
        assert len(range_rules) == 1
        assert range_rules[0].rule_type == RuleType.RANGE
    
    def test_empty_name_raises_error(self):
        """Test that empty name raises ValueError."""
        with pytest.raises(ValueError, match="name cannot be empty"):
            DomainRuleSet(name="", description="test", rules=[])


class TestValidationIssue:
    """Test cases for ValidationIssue entity."""
    
    def test_basic_issue_creation(self):
        """Test creating a basic validation issue."""
        issue = ValidationIssue(
            rule_id="test_rule",
            column_name="age",
            severity=ValidationSeverity.ERROR,
            message="Value out of range",
            affected_rows=[1, 2, 3],
            affected_values=[150, 200, -5],
            dataset_type="real"
        )
        
        assert issue.rule_id == "test_rule"
        assert issue.column_name == "age"
        assert issue.severity == ValidationSeverity.ERROR
        assert issue.message == "Value out of range"
        assert issue.affected_rows == [1, 2, 3]
        assert issue.affected_values == [150, 200, -5]
        assert issue.dataset_type == "real"
    
    def test_issue_with_defaults(self):
        """Test creating issue with default values."""
        issue = ValidationIssue(
            rule_id="test_rule",
            column_name="test_col",
            severity=ValidationSeverity.WARNING,
            message="Test message"
        )
        
        assert issue.affected_rows is None
        assert issue.affected_values is None
        assert issue.dataset_type == "real"  # default


class TestValidationReport:
    """Test cases for ValidationReport entity."""
    
    def test_basic_report_creation(self):
        """Test creating a basic validation report."""
        issues = [
            ValidationIssue("rule1", "col1", ValidationSeverity.ERROR, "Error 1"),
            ValidationIssue("rule2", "col2", ValidationSeverity.WARNING, "Warning 1")
        ]
        
        report = ValidationReport(
            rule_set_name="test_rules",
            total_rules_checked=5,
            issues=issues,
            coherence_issues=["Coherence issue 1"],
            correspondence_issues=["Correspondence issue 1"],
            processing_time=1.5
        )
        
        assert report.rule_set_name == "test_rules"
        assert report.total_rules_checked == 5
        assert len(report.issues) == 2
        assert len(report.coherence_issues) == 1
        assert len(report.correspondence_issues) == 1
        assert report.processing_time == 1.5
    
    def test_error_count_property(self):
        """Test error_count property calculation."""
        issues = [
            ValidationIssue("rule1", "col1", ValidationSeverity.ERROR, "Error 1"),
            ValidationIssue("rule2", "col2", ValidationSeverity.ERROR, "Error 2"),
            ValidationIssue("rule3", "col3", ValidationSeverity.WARNING, "Warning 1")
        ]
        
        report = ValidationReport("test", 3, issues=issues)
        
        assert report.error_count == 2
        assert report.warning_count == 1
        assert report.has_errors is True
    
    def test_get_issues_for_column(self):
        """Test filtering issues by column."""
        issues = [
            ValidationIssue("rule1", "age", ValidationSeverity.ERROR, "Error 1"),
            ValidationIssue("rule2", "age", ValidationSeverity.WARNING, "Warning 1"),
            ValidationIssue("rule3", "name", ValidationSeverity.ERROR, "Error 2")
        ]
        
        report = ValidationReport("test", 3, issues=issues)
        age_issues = report.get_issues_for_column("age")
        
        assert len(age_issues) == 2
        assert all(issue.column_name == "age" for issue in age_issues)


class TestValidationResult:
    """Test cases for ValidationResult entity."""
    
    def test_basic_result_creation(self):
        """Test creating a basic validation result."""
        real_df = pd.DataFrame({"col": [1, 2]})
        synth_df = pd.DataFrame({"col": [3, 4]})
        rule_set = DomainRuleSet("test", "test", [])
        report = ValidationReport("test", 0)
        
        result = ValidationResult(
            real_data=real_df,
            synth_data=synth_df,
            rule_set=rule_set,
            report=report,
            is_valid=True
        )
        
        assert result.real_data.equals(real_df)
        assert result.synth_data.equals(synth_df)
        assert result.rule_set == rule_set
        assert result.report == report
        assert result.is_valid is True
    
    def test_none_data_raises_error(self):
        """Test that None data raises ValueError."""
        synth_df = pd.DataFrame({"col": [1]})
        report = ValidationReport("test", 0)
        
        with pytest.raises(ValueError, match="real_data cannot be None"):
            ValidationResult(
                real_data=None,
                synth_data=synth_df,
                rule_set=None,
                report=report
            )
        
        real_df = pd.DataFrame({"col": [1]})
        with pytest.raises(ValueError, match="synth_data cannot be None"):
            ValidationResult(
                real_data=real_df,
                synth_data=None,
                rule_set=None,
                report=report
            )