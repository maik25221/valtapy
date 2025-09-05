"""Test domain contracts and protocol compliance."""

import pytest
import pandas as pd
from typing import Dict, Any

from src.valtapyV2.domain.contracts import Metric, Preprocessor, Reporter
from src.valtapyV2.domain.entities import MetricResult, DatasetSpec, RunSummary, ReportSpec
from src.valtapyV2.infrastructure.metrics.fidelity.ks_test import KSTestMetric
from src.valtapyV2.infrastructure.preprocess.transformers import TabularPreprocessor
from src.valtapyV2.infrastructure.reporting.json_reporter import JSONReporter


class TestMetricProtocol:
    """Test that metric implementations conform to the Metric protocol."""
    
    def test_ks_metric_implements_protocol(self, sample_real_data, sample_synthetic_data):
        """Test that KSTestMetric implements the Metric protocol."""
        metric = KSTestMetric()
        
        # Check required attributes
        assert hasattr(metric, 'name')
        assert hasattr(metric, 'family')
        assert hasattr(metric, 'purpose_tags')
        
        assert isinstance(metric.name, str)
        assert isinstance(metric.family, str)
        assert isinstance(metric.purpose_tags, set)
        
        # Check methods
        context = {"stats_store": None}
        fitted_metric = metric.fit(sample_real_data, sample_synthetic_data, context)
        assert fitted_metric is metric  # Should return self
        
        result = metric.compute()
        assert isinstance(result, MetricResult)
        assert result.id == "fidelity.ks"
        assert result.family == "fidelity"
        assert isinstance(result.value, float)
        assert isinstance(result.details, dict)
    
    def test_metric_result_validation(self):
        """Test MetricResult validation."""
        # Valid result
        result = MetricResult(
            id="test.metric",
            value=0.75,
            details={"test": True},
            family="fidelity",
            purpose_tags={"test"}
        )
        assert result.id == "test.metric"
        assert result.value == 0.75
        
        # Invalid family should raise error
        with pytest.raises(ValueError, match="Invalid family"):
            MetricResult(
                id="test.metric",
                value=0.5,
                details={},
                family="invalid_family",
                purpose_tags=set()
            )


class TestPreprocessorProtocol:
    """Test that preprocessor implementations conform to the Preprocessor protocol."""
    
    def test_tabular_preprocessor_implements_protocol(self, sample_real_data, sample_dataset_spec):
        """Test that TabularPreprocessor implements the Preprocessor protocol."""
        preprocessor = TabularPreprocessor()
        
        # Test fit method
        fitted = preprocessor.fit(sample_real_data, sample_dataset_spec)
        assert fitted is preprocessor  # Should return self
        
        # Test transform method
        transformed = preprocessor.transform(sample_real_data)
        assert isinstance(transformed, pd.DataFrame)
        assert len(transformed) == len(sample_real_data)
        
        # Test metadata method
        metadata = preprocessor.metadata()
        assert isinstance(metadata, dict)
        assert "fitted" in metadata
        assert metadata["fitted"] is True


class TestReporterProtocol:
    """Test that reporter implementations conform to the Reporter protocol."""
    
    def test_json_reporter_implements_protocol(self, sample_eval_plan, temp_data_files):
        """Test that JSONReporter implements the Reporter protocol."""
        import tempfile
        import os
        from pathlib import Path
        
        # Create minimal run summary
        results = [
            MetricResult(
                id="test.metric",
                value=0.85,
                details={"test": True},
                family="fidelity",
                purpose_tags={"test"}
            )
        ]
        
        run_summary = RunSummary(
            plan=sample_eval_plan,
            results=results,
            aggregates={"fidelity_score": 0.85},
            artifacts={}
        )
        
        # Create temporary output directory
        temp_dir = tempfile.mkdtemp()
        report_spec = ReportSpec(
            formats=["json"],
            output_dir=temp_dir
        )
        
        try:
            # Test reporter
            reporter = JSONReporter()
            reporter.render(run_summary, report_spec)
            
            # Check that file was created
            output_file = Path(temp_dir) / "summary.json"
            assert output_file.exists()
            
            # Check that file contains valid JSON
            import json
            with open(output_file) as f:
                data = json.load(f)
            
            assert "metadata" in data
            assert "plan" in data
            assert "results" in data
            assert "aggregates" in data
            
        finally:
            # Cleanup
            try:
                if output_file.exists():
                    os.unlink(output_file)
                os.rmdir(temp_dir)
            except:
                pass


class TestRegistryProtocols:
    """Test registry implementations."""
    
    def test_metric_registry_basic_operations(self):
        """Test basic metric registry operations."""
        from src.valtapyV2.infrastructure.metrics.registry import get_metric_registry
        
        registry = get_metric_registry()
        
        # Test list_ids
        metric_ids = registry.list_ids()
        assert isinstance(metric_ids, list)
        assert "fidelity.ks" in metric_ids
        
        # Test family filtering
        fidelity_ids = registry.list_ids(family="fidelity")
        assert isinstance(fidelity_ids, list)
        assert all(mid.startswith("fidelity.") for mid in fidelity_ids)
        
        # Test get method
        ks_metric_class = registry.get("fidelity.ks")
        assert ks_metric_class is not None
        
        # Test instantiation
        metric_instance = ks_metric_class()
        assert hasattr(metric_instance, 'name')
        assert hasattr(metric_instance, 'family')
    
    def test_reporter_registry_basic_operations(self):
        """Test basic reporter registry operations.""" 
        from src.valtapyV2.infrastructure.reporting.registry import get_reporter_registry
        
        registry = get_reporter_registry()
        
        # Test list_formats
        formats = registry.list_formats()
        assert isinstance(formats, list)
        assert "json" in formats
        assert "md" in formats
        
        # Test get method
        json_reporter_class = registry.get("json")
        assert json_reporter_class is not None
        
        # Test instantiation
        reporter_instance = json_reporter_class()
        assert hasattr(reporter_instance, 'render')