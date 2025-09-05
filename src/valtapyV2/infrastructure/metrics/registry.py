"""Metric registry with decorator-based registration."""

from typing import Dict, Type, List, Optional, Callable
from ...domain.contracts import Metric
from ...domain.errors import RegistryError


class MetricRegistryImpl:
    """Concrete implementation of metric registry."""
    
    def __init__(self):
        self._metrics: Dict[str, Type[Metric]] = {}
    
    def register(self, metric_id: str, metric_class: Type[Metric]) -> None:
        """Register a metric implementation."""
        if not metric_id:
            raise RegistryError("Metric ID cannot be empty", "metric")
        
        if metric_id in self._metrics:
            raise RegistryError(f"Metric already registered: {metric_id}", "metric", metric_id)
        
        self._metrics[metric_id] = metric_class
    
    def get(self, metric_id: str) -> Type[Metric]:
        """Retrieve metric class by ID."""
        if metric_id not in self._metrics:
            available = list(self._metrics.keys())
            raise RegistryError(
                f"Metric not found: {metric_id}. Available: {available}",
                "metric",
                metric_id
            )
        return self._metrics[metric_id]
    
    def list_ids(self, family: Optional[str] = None) -> List[str]:
        """List available metric IDs, optionally filtered by family."""
        if family is None:
            return list(self._metrics.keys())
        
        # Filter by family prefix
        return [mid for mid in self._metrics.keys() if mid.startswith(f"{family}.")]
    
    def list_by_family(self) -> Dict[str, List[str]]:
        """List metrics grouped by family."""
        families = {}
        for metric_id in self._metrics.keys():
            if "." in metric_id:
                family = metric_id.split(".")[0]
                if family not in families:
                    families[family] = []
                families[family].append(metric_id)
        return families


# Global registry instance
_metric_registry = MetricRegistryImpl()


def get_metric_registry() -> MetricRegistryImpl:
    """Get global metric registry instance."""
    return _metric_registry


def register(metric_id: str) -> Callable[[Type[Metric]], Type[Metric]]:
    """
    Decorator for registering metrics in the global registry.
    
    Usage:
        @register("fidelity.ks")
        class KSTestMetric:
            ...
    """
    def decorator(metric_class: Type[Metric]) -> Type[Metric]:
        _metric_registry.register(metric_id, metric_class)
        return metric_class
    return decorator


def register_metric(metric_id: str, metric_class: Type[Metric]) -> None:
    """Register a metric in the global registry (alternative to decorator)."""
    _metric_registry.register(metric_id, metric_class)


# Import and register default metrics
def _register_default_metrics():
    """Register all built-in metrics."""
    # Import modules to trigger registration via decorators
    from .fidelity import ks_test, correlation_delta, mi_delta
    from .utility import pmse, accuracy  
    from .privacy import nndr, membership_inference


# Register defaults on import
_register_default_metrics()