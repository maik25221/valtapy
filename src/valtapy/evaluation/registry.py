"""Basic implementation of metric registry."""

from typing import Dict, List, Optional

from .contracts import MetricRegistry, Metric
from .entities import MetricFamily
from .exceptions import MetricRegistrationError, MetricNotFoundError


class BasicMetricRegistry:
    """Basic implementation of metric registry."""
    
    def __init__(self):
        self._metrics: Dict[str, Metric] = {}
    
    def register(self, metric: Metric) -> None:
        """Register a metric implementation."""
        if not hasattr(metric, 'metric_id') or not metric.metric_id:
            raise MetricRegistrationError(
                "unknown", "Metric must have a non-empty metric_id attribute"
            )
        
        if not hasattr(metric, 'family') or not isinstance(metric.family, MetricFamily):
            raise MetricRegistrationError(
                metric.metric_id, "Metric must have a valid MetricFamily"
            )
        
        # Check if metric implements required methods
        required_methods = ['compute', 'can_compute', 'validate_parameters']
        for method in required_methods:
            if not hasattr(metric, method) or not callable(getattr(metric, method)):
                raise MetricRegistrationError(
                    metric.metric_id, f"Metric must implement {method} method"
                )
        
        self._metrics[metric.metric_id] = metric
    
    def get_metric(self, metric_id: str) -> Metric:
        """Get metric implementation by ID."""
        if metric_id not in self._metrics:
            available = list(self._metrics.keys())
            raise MetricNotFoundError(metric_id, available)
        
        return self._metrics[metric_id]
    
    def list_metrics(self, family: Optional[MetricFamily] = None) -> List[str]:
        """List available metric IDs, optionally filtered by family."""
        if family is None:
            return list(self._metrics.keys())
        
        return [
            metric_id for metric_id, metric in self._metrics.items()
            if metric.family == family
        ]
    
    def get_metrics_by_family(self, family: MetricFamily) -> List[Metric]:
        """Get all metrics for a specific family."""
        return [
            metric for metric in self._metrics.values()
            if metric.family == family
        ]
    
    def is_registered(self, metric_id: str) -> bool:
        """Check if a metric is registered."""
        return metric_id in self._metrics
    
    def unregister(self, metric_id: str) -> None:
        """Unregister a metric (useful for testing)."""
        if metric_id in self._metrics:
            del self._metrics[metric_id]
    
    def clear(self) -> None:
        """Clear all registered metrics (useful for testing)."""
        self._metrics.clear()
    
    def get_registry_info(self) -> dict:
        """Get information about the registry state."""
        family_counts = {}
        for metric in self._metrics.values():
            family_name = metric.family.value
            family_counts[family_name] = family_counts.get(family_name, 0) + 1
        
        return {
            "total_metrics": len(self._metrics),
            "metrics_by_family": family_counts,
            "available_metrics": list(self._metrics.keys())
        }


# Global registry instance
_global_registry = BasicMetricRegistry()


def get_metric_registry() -> BasicMetricRegistry:
    """Get the global metric registry instance."""
    return _global_registry


def register_metric(metric: Metric) -> None:
    """Convenience function to register a metric in the global registry."""
    _global_registry.register(metric)


def auto_register_builtin_metrics() -> None:
    """Auto-register all built-in metrics."""
    # Import and register Fidelity metrics
    from .metrics.fidelity.ks_test import KSTestMetric

    ks_metric = KSTestMetric()
    register_metric(ks_metric)

    # Import and register Utility metrics (ML Efficiency)
    from .metrics.utility import (
        TTRMetric,
        TTSMetric,
        TRTSMetric,
        TSTRMetric,
        TTRSMetric
    )

    # Register all ML efficiency metrics
    register_metric(TTRMetric())
    register_metric(TTSMetric())
    register_metric(TRTSMetric())
    register_metric(TSTRMetric())
    register_metric(TTRSMetric())

    # TODO: Add other built-in metrics as they are implemented
    # from .metrics.utility.pmse import PMSEMetric
    # from .metrics.privacy.nndr import NNDRMetric
    # register_metric(PMSEMetric())
    # register_metric(NNDRMetric())