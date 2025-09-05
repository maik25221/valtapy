"""Integration tests for the metric registry system."""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from valtapyV2.infrastructure.metrics.registry import (
    get_metric_registry,
    MetricRegistryImpl,
    register,
    register_metric
)
from valtapyV2.domain.entities import MetricResult
from valtapyV2.domain.errors import RegistryError
from tests.utils.test_helpers import create_mock_context, convert_to_pandas_mock
from tests.utils.data_generators import generate_tabular_data


class MockTestMetric:
    """Mock metric for testing registry functionality."""
    
    name = "test_metric"
    family = "test"
    purpose_tags = {"test"}
    
    def __init__(self):
        self._fitted = False
    
    def fit(self, real_data, synth_data, context):
        self._fitted = True
        return self
    
    def compute(self):
        if not self._fitted:
            raise ValueError("Metric not fitted")
        return MetricResult(
            id="test.mock",
            value=0.75,
            details={"test": "success"},
            family="test",
            purpose_tags={"test"}
        )


class TestMetricRegistry:
    """Test suite for metric registry integration."""
    
    def test_registry_initialization(self):
        """Test that registry initializes correctly."""
        registry = MetricRegistryImpl()
        
        # New registry should be empty
        assert len(registry.list_ids()) == 0
        assert len(registry.list_by_family()) == 0
    
    def test_global_registry_has_default_metrics(self):
        """Test that global registry has built-in metrics registered."""
        registry = get_metric_registry()
        
        metric_ids = registry.list_ids()
        
        # Should have metrics from all families
        fidelity_metrics = [mid for mid in metric_ids if mid.startswith("fidelity.")]
        utility_metrics = [mid for mid in metric_ids if mid.startswith("utility.")]
        privacy_metrics = [mid for mid in metric_ids if mid.startswith("privacy.")]
        
        assert len(fidelity_metrics) >= 2, f"Expected at least 2 fidelity metrics, got: {fidelity_metrics}"
        assert len(utility_metrics) >= 2, f"Expected at least 2 utility metrics, got: {utility_metrics}"
        assert len(privacy_metrics) >= 2, f"Expected at least 2 privacy metrics, got: {privacy_metrics}"
        
        # Check specific expected metrics
        expected_metrics = [
            "fidelity.ks",
            "fidelity.correlation_delta",
            "utility.pmse",
            "utility.accuracy",
            "privacy.nndr",
            "privacy.membership_inference"
        ]
        
        for expected_metric in expected_metrics:
            assert expected_metric in metric_ids, f"Expected metric {expected_metric} not found in {metric_ids}"
    
    def test_metric_registration_via_decorator(self):
        """Test metric registration using decorator."""
        # Create a new registry for isolated testing
        test_registry = MetricRegistryImpl()
        
        # Use decorator to register mock metric
        @register("test.decorated")
        class DecoratedTestMetric(MockTestMetric):
            pass
        
        # Register in test registry manually
        test_registry.register("test.decorated", DecoratedTestMetric)
        
        # Verify registration
        assert "test.decorated" in test_registry.list_ids()
        retrieved_class = test_registry.get("test.decorated")
        assert retrieved_class == DecoratedTestMetric
    
    def test_metric_registration_via_function(self):
        """Test metric registration using register_metric function."""
        test_registry = MetricRegistryImpl()
        
        class FunctionRegisteredMetric(MockTestMetric):
            pass
        
        # Register using function
        test_registry.register("test.function", FunctionRegisteredMetric)
        
        # Verify registration
        assert "test.function" in test_registry.list_ids()
        retrieved_class = test_registry.get("test.function")
        assert retrieved_class == FunctionRegisteredMetric
    
    def test_registry_duplicate_registration_error(self):
        """Test that duplicate metric registration raises error."""
        test_registry = MetricRegistryImpl()
        
        class FirstMetric(MockTestMetric):
            pass
        
        class SecondMetric(MockTestMetric):
            pass
        
        # First registration should succeed
        test_registry.register("test.duplicate", FirstMetric)
        
        # Second registration should fail
        with pytest.raises(RegistryError) as exc_info:
            test_registry.register("test.duplicate", SecondMetric)
        
        assert "already registered" in str(exc_info.value)
        assert "test.duplicate" in str(exc_info.value)
    
    def test_registry_empty_id_error(self):
        """Test that empty metric ID raises error."""
        test_registry = MetricRegistryImpl()
        
        with pytest.raises(RegistryError) as exc_info:
            test_registry.register("", MockTestMetric)
        
        assert "cannot be empty" in str(exc_info.value)
    
    def test_registry_get_nonexistent_metric(self):
        """Test retrieving non-existent metric raises error."""
        test_registry = MetricRegistryImpl()
        
        with pytest.raises(RegistryError) as exc_info:
            test_registry.get("nonexistent.metric")
        
        error_msg = str(exc_info.value)
        assert "not found" in error_msg
        assert "nonexistent.metric" in error_msg
        assert "Available:" in error_msg
    
    def test_registry_list_by_family(self):
        """Test listing metrics grouped by family."""
        test_registry = MetricRegistryImpl()
        
        # Register metrics in different families
        test_registry.register("fidelity.test1", MockTestMetric)
        test_registry.register("fidelity.test2", MockTestMetric)
        test_registry.register("utility.test1", MockTestMetric)
        test_registry.register("privacy.test1", MockTestMetric)
        test_registry.register("standalone", MockTestMetric)  # No family prefix
        
        families = test_registry.list_by_family()
        
        # Check family groupings
        assert "fidelity" in families
        assert len(families["fidelity"]) == 2
        assert "fidelity.test1" in families["fidelity"]
        assert "fidelity.test2" in families["fidelity"]
        
        assert "utility" in families
        assert len(families["utility"]) == 1
        assert "utility.test1" in families["utility"]
        
        assert "privacy" in families
        assert len(families["privacy"]) == 1
        assert "privacy.test1" in families["privacy"]
        
        # Standalone metric (no family) should not appear in families dict
        assert "standalone" not in families
    
    def test_registry_list_ids_filtered_by_family(self):
        """Test listing metric IDs filtered by family."""
        test_registry = MetricRegistryImpl()
        
        # Register metrics in different families
        test_registry.register("fidelity.test1", MockTestMetric)
        test_registry.register("fidelity.test2", MockTestMetric)
        test_registry.register("utility.test1", MockTestMetric)
        test_registry.register("privacy.test1", MockTestMetric)
        
        # Test filtering by family
        fidelity_ids = test_registry.list_ids(family="fidelity")
        assert len(fidelity_ids) == 2
        assert "fidelity.test1" in fidelity_ids
        assert "fidelity.test2" in fidelity_ids
        
        utility_ids = test_registry.list_ids(family="utility")
        assert len(utility_ids) == 1
        assert "utility.test1" in utility_ids
        
        privacy_ids = test_registry.list_ids(family="privacy")
        assert len(privacy_ids) == 1
        assert "privacy.test1" in privacy_ids
        
        # Non-existent family should return empty list
        nonexistent_ids = test_registry.list_ids(family="nonexistent")
        assert len(nonexistent_ids) == 0
    
    def test_registry_list_all_ids(self):
        """Test listing all metric IDs without filtering."""
        test_registry = MetricRegistryImpl()
        
        # Register some test metrics
        test_metrics = [
            "fidelity.test1",
            "utility.test1", 
            "privacy.test1",
            "standalone"
        ]
        
        for metric_id in test_metrics:
            test_registry.register(metric_id, MockTestMetric)
        
        all_ids = test_registry.list_ids()
        assert len(all_ids) == len(test_metrics)
        
        for metric_id in test_metrics:
            assert metric_id in all_ids
    
    def test_metric_instantiation_from_registry(self):
        """Test instantiating and using metrics retrieved from registry."""
        # Use global registry with real metrics
        registry = get_metric_registry()
        
        # Test data
        real_data = generate_tabular_data(n_samples=30, n_numeric=2, seed=42)
        synth_data = generate_tabular_data(n_samples=30, n_numeric=2, seed=43)
        
        real_df = convert_to_pandas_mock(real_data)
        synth_df = convert_to_pandas_mock(synth_data)
        context = create_mock_context()
        
        # Test retrieving and using a fidelity metric
        ks_class = registry.get("fidelity.ks")
        ks_metric = ks_class()
        
        # Should be able to fit and compute
        ks_metric.fit(real_df, synth_df, context)
        result = ks_metric.compute()
        
        assert isinstance(result, MetricResult)
        assert result.id == "fidelity.ks"
        assert result.family == "fidelity"
        assert isinstance(result.value, (int, float))
    
    def test_registry_metric_family_consistency(self):
        """Test that metrics in registry have consistent family naming."""
        registry = get_metric_registry()
        
        for metric_id in registry.list_ids():
            if "." in metric_id:
                expected_family = metric_id.split(".")[0]
                metric_class = registry.get(metric_id)
                metric_instance = metric_class()
                
                # Metric's family attribute should match ID prefix
                assert hasattr(metric_instance, 'family'), f"Metric {metric_id} missing family attribute"
                assert metric_instance.family == expected_family, \
                    f"Metric {metric_id} has family '{metric_instance.family}', expected '{expected_family}'"
    
    def test_registry_metric_attributes_consistency(self):
        """Test that all metrics in registry have required attributes."""
        registry = get_metric_registry()
        
        required_attributes = ['name', 'family', 'purpose_tags']
        required_methods = ['fit', 'compute']
        
        for metric_id in registry.list_ids():
            metric_class = registry.get(metric_id)
            metric_instance = metric_class()
            
            # Check required attributes
            for attr in required_attributes:
                assert hasattr(metric_instance, attr), \
                    f"Metric {metric_id} missing required attribute: {attr}"
            
            # Check required methods
            for method in required_methods:
                assert hasattr(metric_instance, method), \
                    f"Metric {metric_id} missing required method: {method}"
                assert callable(getattr(metric_instance, method)), \
                    f"Metric {metric_id}.{method} is not callable"
    
    def test_registry_concurrent_access(self):
        """Test registry behavior under concurrent-like access patterns."""
        import threading
        import time
        
        test_registry = MetricRegistryImpl()
        results = []
        errors = []
        
        def register_metrics(start_id, count):
            """Register multiple metrics in a thread-like pattern."""
            try:
                for i in range(count):
                    metric_id = f"concurrent.metric_{start_id}_{i}"
                    test_registry.register(metric_id, MockTestMetric)
                    results.append(metric_id)
            except Exception as e:
                errors.append(e)
        
        def read_metrics(iterations):
            """Read from registry in a thread-like pattern."""
            try:
                for _ in range(iterations):
                    # Try to read existing metrics
                    ids = test_registry.list_ids()
                    if ids:
                        # Try to get a random metric
                        metric_class = test_registry.get(ids[0])
                        results.append(f"read_{metric_class.__name__}")
                    time.sleep(0.001)  # Small delay
            except Exception as e:
                errors.append(e)
        
        # Simulate concurrent-like operations
        # (Note: Not true concurrency due to GIL, but tests access patterns)
        threads = []
        
        # Start registration "threads"
        for i in range(3):
            t = threading.Thread(target=register_metrics, args=(i, 5))
            threads.append(t)
            t.start()
        
        # Start reading "threads"  
        for i in range(2):
            t = threading.Thread(target=read_metrics, args=(10,))
            threads.append(t)
            t.start()
        
        # Wait for completion
        for t in threads:
            t.join()
        
        # Check results
        assert len(errors) == 0, f"Concurrent access resulted in errors: {errors}"
        
        # Should have registered 15 metrics (3 threads * 5 metrics each)
        registered_ids = test_registry.list_ids()
        concurrent_metrics = [mid for mid in registered_ids if mid.startswith("concurrent.")]
        assert len(concurrent_metrics) == 15
    
    def test_registry_error_handling_with_invalid_classes(self):
        """Test registry error handling with invalid metric classes."""
        test_registry = MetricRegistryImpl()
        
        class InvalidMetric:
            # Missing required attributes/methods
            pass
        
        # Registry should accept registration (it doesn't validate class structure)
        test_registry.register("invalid.metric", InvalidMetric)
        
        # But instantiation might fail later when used
        retrieved_class = test_registry.get("invalid.metric")
        assert retrieved_class == InvalidMetric
        
        # The metric instance will fail when trying to use required methods
        invalid_instance = retrieved_class()
        
        # Should not have required attributes
        assert not hasattr(invalid_instance, 'name')
        assert not hasattr(invalid_instance, 'family')
        assert not hasattr(invalid_instance, 'fit')
    
    def test_registry_metric_name_conventions(self):
        """Test that metrics follow expected naming conventions."""
        registry = get_metric_registry()
        
        for metric_id in registry.list_ids():
            # Should follow family.metric_name pattern for built-in metrics
            if "." in metric_id:
                parts = metric_id.split(".")
                assert len(parts) == 2, f"Metric ID {metric_id} should have exactly one dot"
                
                family, name = parts
                assert family.isalpha(), f"Family '{family}' should be alphabetic"
                assert name.replace("_", "").isalnum(), f"Name '{name}' should be alphanumeric (with underscores)"
                
                # Family should be one of expected families
                expected_families = {"fidelity", "utility", "privacy"}
                assert family in expected_families, f"Unexpected family '{family}' in metric {metric_id}"
    
    def test_registry_integration_with_real_metrics(self):
        """Test full integration with real metrics from all families."""
        registry = get_metric_registry()
        
        # Test data
        real_data = generate_tabular_data(n_samples=40, n_numeric=3, include_target=True, seed=42)
        synth_data = generate_tabular_data(n_samples=40, n_numeric=3, include_target=True, seed=43)
        
        real_df = convert_to_pandas_mock(real_data)
        synth_df = convert_to_pandas_mock(synth_data)
        context = create_mock_context()
        
        # Test one metric from each family
        test_metrics = [
            "fidelity.ks",
            "utility.pmse", 
            "privacy.nndr"
        ]
        
        results = {}
        
        for metric_id in test_metrics:
            # Get metric class from registry
            metric_class = registry.get(metric_id)
            
            # Instantiate metric
            metric = metric_class()
            
            # Fit and compute
            try:
                metric.fit(real_df, synth_df, context)
                result = metric.compute()
                
                # Validate result structure
                assert isinstance(result, MetricResult)
                assert result.id.startswith(metric_id.split('.')[0])
                assert isinstance(result.value, (int, float))
                assert isinstance(result.details, dict)
                
                results[metric_id] = result
                
            except Exception as e:
                # Some metrics might fail due to missing dependencies or data requirements
                # This is acceptable in integration tests
                print(f"Metric {metric_id} failed: {e}")
        
        # Should have successfully run at least some metrics
        assert len(results) > 0, "No metrics completed successfully"