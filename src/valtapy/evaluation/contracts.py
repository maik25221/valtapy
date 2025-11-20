"""Contracts for the evaluation phase."""

from typing import Protocol, TYPE_CHECKING, Self, Optional, Union

if TYPE_CHECKING:
    import pandas as pd
    from .entities import (
        MetricSpec, MetricResult, EvaluationConfig, EvaluationResult,
        MetricExecutionContext, MetricFamily
    )


class Metric(Protocol):
    """Protocol for individual metric implementations."""
    
    metric_id: str
    family: "MetricFamily"
    name: str
    description: str
    
    def compute(
        self,
        context: "MetricExecutionContext"
    ) -> "MetricResult":
        """
        Compute the metric given the execution context.
        
        Args:
            context: Contains real data, synthetic data, parameters, etc.
            
        Returns:
            MetricResult with computed value and details
            
        Raises:
            MetricComputationError: When metric computation fails
        """
        ...
    
    def can_compute(
        self,
        real_df: "pd.DataFrame",
        synth_df: "pd.DataFrame"
    ) -> bool:
        """Check if this metric can be computed for the given datasets."""
        ...
    
    def validate_parameters(self, parameters: dict) -> bool:
        """Validate that parameters are correct for this metric."""
        ...


class MetricRegistry(Protocol):
    """Protocol for managing and discovering metrics."""
    
    def register(self, metric: "Metric") -> None:
        """Register a metric implementation."""
        ...
    
    def get_metric(self, metric_id: str) -> "Metric":
        """Get metric implementation by ID."""
        ...
    
    def list_metrics(
        self,
        family: "MetricFamily" = None
    ) -> list[str]:
        """List available metric IDs, optionally filtered by family."""
        ...
    
    def get_metrics_by_family(self, family: "MetricFamily") -> list["Metric"]:
        """Get all metrics for a specific family."""
        ...
    
    def is_registered(self, metric_id: str) -> bool:
        """Check if a metric is registered."""
        ...


class MetricExecutor(Protocol):
    """Protocol for executing individual metrics."""
    
    def execute_metric(
        self,
        metric_spec: "MetricSpec",
        context: "MetricExecutionContext"
    ) -> "MetricResult":
        """
        Execute a single metric.
        
        Args:
            metric_spec: Specification of what metric to run
            context: Execution context with data and parameters
            
        Returns:
            MetricResult with computed value
            
        Raises:
            MetricNotFoundError: When metric is not registered
            MetricComputationError: When computation fails
        """
        ...
    
    def can_execute(
        self,
        metric_spec: "MetricSpec",
        context: "MetricExecutionContext"
    ) -> bool:
        """Check if metric can be executed with given context."""
        ...


class EvaluationOrchestrator(Protocol):
    """Protocol for orchestrating complete evaluation process."""
    
    def evaluate(
        self,
        real_df: "pd.DataFrame",
        synth_df: "pd.DataFrame",
        config: "EvaluationConfig"
    ) -> "EvaluationResult":
        """
        Execute complete evaluation process.
        
        Args:
            real_df: Original dataset
            synth_df: Synthetic dataset
            config: Evaluation configuration with metrics to run
            
        Returns:
            EvaluationResult with all computed metrics and summary
            
        Raises:
            EvaluationError: When evaluation process fails
        """
        ...
    
    def validate_config(
        self,
        config: "EvaluationConfig",
        real_df: "pd.DataFrame",
        synth_df: "pd.DataFrame"
    ) -> list[str]:
        """
        Validate evaluation configuration.
        
        Returns list of validation issues found.
        """
        ...


class MetricCache(Protocol):
    """Protocol for caching metric computations and intermediate results."""
    
    def get_cached_result(
        self,
        metric_id: str,
        data_hash: str,
        parameters_hash: str
    ) -> Optional["MetricResult"]:
        """Get cached metric result if available."""
        ...
    
    def cache_result(
        self,
        metric_id: str,
        data_hash: str,
        parameters_hash: str,
        result: "MetricResult"
    ) -> None:
        """Cache a metric result for future use."""
        ...
    
    def clear_cache(self, metric_id: str = None) -> None:
        """Clear cached results, optionally for specific metric."""
        ...


class MetricAggregator(Protocol):
    """Protocol for aggregating metrics into family scores."""
    
    def aggregate_family_score(
        self,
        results: list["MetricResult"],
        family: "MetricFamily"
    ) -> float:
        """
        Aggregate multiple metric results into a family score.
        
        Args:
            results: List of metric results for the family
            family: The metric family to aggregate
            
        Returns:
            Aggregated score (typically 0-1 or 0-100)
        """
        ...
    
    def aggregate_overall_score(
        self,
        family_scores: dict["MetricFamily", float]
    ) -> float:
        """
        Aggregate family scores into overall evaluation score.
        
        Args:
            family_scores: Dictionary mapping families to their scores
            
        Returns:
            Overall aggregated score
        """
        ...