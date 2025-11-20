"""Entities for the evaluation phase."""

from dataclasses import dataclass, field
from typing import Any, Optional, Literal
from enum import Enum


class MetricFamily(Enum):
    """Types of metric families for synthetic data evaluation."""
    
    FIDELITY = "fidelity"    # How well synthetic data preserves real data properties
    UTILITY = "utility"      # How useful synthetic data is for downstream tasks
    PRIVACY = "privacy"      # How well synthetic data protects original data privacy


@dataclass(frozen=True)
class MetricSpec:
    """Specification for a metric to be computed."""
    
    metric_id: str
    family: MetricFamily
    parameters: dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        if not self.metric_id:
            raise ValueError("metric_id cannot be empty")
        if not isinstance(self.family, MetricFamily):
            raise ValueError("family must be a MetricFamily enum")


@dataclass(frozen=True)
class MetricResult:
    """Result of a single metric computation."""
    
    metric_id: str
    family: MetricFamily
    value: float
    details: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    computation_time: float = 0.0
    
    def __post_init__(self) -> None:
        if not self.metric_id:
            raise ValueError("metric_id cannot be empty")
        if not isinstance(self.family, MetricFamily):
            raise ValueError("family must be a MetricFamily enum")
        if not isinstance(self.value, (int, float)):
            raise ValueError("value must be numeric")


@dataclass(frozen=True)
class EvaluationConfig:
    """Configuration for evaluation process."""
    
    metrics: list[MetricSpec]
    parallel_execution: bool = False
    cache_enabled: bool = True
    random_seed: int = 42
    
    def __post_init__(self) -> None:
        if not self.metrics:
            raise ValueError("At least one metric must be specified")
        if not isinstance(self.metrics, list):
            raise ValueError("metrics must be a list")


@dataclass(frozen=True)
class EvaluationSummary:
    """Summary statistics for evaluation results."""
    
    total_metrics: int
    successful_metrics: int
    failed_metrics: int
    total_time: float
    fidelity_score: Optional[float] = None
    utility_score: Optional[float] = None
    privacy_score: Optional[float] = None
    
    def __post_init__(self) -> None:
        if self.total_metrics < 0:
            raise ValueError("total_metrics cannot be negative")
        if self.successful_metrics < 0:
            raise ValueError("successful_metrics cannot be negative")
        if self.failed_metrics < 0:
            raise ValueError("failed_metrics cannot be negative")
    
    @property
    def success_rate(self) -> float:
        """Percentage of metrics that executed successfully."""
        if self.total_metrics == 0:
            return 0.0
        return (self.successful_metrics / self.total_metrics) * 100


@dataclass(frozen=True)
class EvaluationResult:
    """Result of the complete evaluation process."""
    
    real_data: Any  # Will be pd.DataFrame in practice
    synth_data: Any  # Will be pd.DataFrame in practice
    config: EvaluationConfig
    results: list[MetricResult]
    summary: EvaluationSummary
    errors: list[str] = field(default_factory=list)
    
    def __post_init__(self) -> None:
        if self.real_data is None:
            raise ValueError("real_data cannot be None")
        if self.synth_data is None:
            raise ValueError("synth_data cannot be None")
        if not isinstance(self.results, list):
            raise ValueError("results must be a list")
    
    def get_results_by_family(self, family: MetricFamily) -> list[MetricResult]:
        """Get all results for a specific metric family."""
        return [result for result in self.results if result.family == family]
    
    def get_result_by_id(self, metric_id: str) -> Optional[MetricResult]:
        """Get result for a specific metric ID."""
        for result in self.results:
            if result.metric_id == metric_id:
                return result
        return None
    
    def has_errors(self) -> bool:
        """Check if there were any errors during evaluation."""
        return len(self.errors) > 0 or self.summary.failed_metrics > 0


@dataclass(frozen=True)
class MetricExecutionContext:
    """Context information for metric execution."""
    
    real_data: Any  # Will be pd.DataFrame in practice
    synth_data: Any  # Will be pd.DataFrame in practice
    parameters: dict[str, Any] = field(default_factory=dict)
    cache_enabled: bool = True
    random_seed: int = 42
    
    def __post_init__(self) -> None:
        if self.real_data is None:
            raise ValueError("real_data cannot be None")
        if self.synth_data is None:
            raise ValueError("synth_data cannot be None")