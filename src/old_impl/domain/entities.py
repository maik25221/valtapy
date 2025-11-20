"""Domain entities using dataclasses for immutable data structures."""

from dataclasses import dataclass, field
from typing import Any, Literal, Optional


@dataclass(frozen=True)
class DatasetSpec:
    """Specification for dataset structure and constraints."""
    
    target: Optional[str] = None
    dtypes: dict[str, str] = field(default_factory=dict)
    constraints: dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate dataset specification after initialization."""
        if self.target and not isinstance(self.target, str):
            raise ValueError("Target must be a string column name or None")


@dataclass(frozen=True)
class EvalPlan:
    """Evaluation plan specifying metrics and execution parameters."""
    
    metric_ids: list[str]
    seed: int = 42
    cv_splits: int = 3
    models: Optional[list[str]] = None
    purpose: Optional[str] = None
    
    def __post_init__(self) -> None:
        """Validate evaluation plan after initialization."""
        if not self.metric_ids:
            raise ValueError("At least one metric_id must be specified")
        if self.seed < 0:
            raise ValueError("Seed must be non-negative")
        if self.cv_splits < 2:
            raise ValueError("CV splits must be at least 2")


@dataclass(frozen=True)
class MetricResult:
    """Result from a single metric computation."""
    
    id: str
    value: float
    details: dict[str, Any]
    family: Literal["fidelity", "utility", "privacy"]
    purpose_tags: set[str] = field(default_factory=set)
    
    def __post_init__(self) -> None:
        """Validate metric result after initialization."""
        if not self.id:
            raise ValueError("Metric ID cannot be empty")
        if self.family not in {"fidelity", "utility", "privacy"}:
            raise ValueError(f"Invalid family: {self.family}")


@dataclass(frozen=True)
class RunSummary:
    """Summary of a complete evaluation run."""
    
    plan: EvalPlan
    results: list[MetricResult]
    aggregates: dict[str, Any]
    artifacts: dict[str, Any] = field(default_factory=dict)
    
    def get_results_by_family(self, family: str) -> list[MetricResult]:
        """Get all results for a specific family."""
        return [r for r in self.results if r.family == family]
    
    def get_family_score(self, family: str) -> float:
        """Get aggregated score for a specific family."""
        return self.aggregates.get(f"{family}_score", 0.0)


@dataclass(frozen=True)
class ReportSpec:
    """Specification for report generation."""
    
    formats: list[str]
    output_dir: str
    include_details: bool = True
    include_artifacts: bool = False
    
    def __post_init__(self) -> None:
        """Validate report specification after initialization."""
        if not self.formats:
            raise ValueError("At least one format must be specified")
        if not self.output_dir:
            raise ValueError("Output directory cannot be empty")