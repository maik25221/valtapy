"""Domain contracts using typing.Protocol for stable interfaces."""

from typing import Protocol, Self, Any, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    from .entities import MetricResult, DatasetSpec, ReportSpec, RunSummary


class Metric(Protocol):
    """Protocol for metric implementations following Strategy pattern."""
    
    name: str
    family: Literal["fidelity", "utility", "privacy"]
    purpose_tags: set[str]
    
    def fit(self, real: "pd.DataFrame", synth: "pd.DataFrame", context: dict) -> Self:
        """Fit the metric with data and context. Returns self for method chaining."""
        ...
    
    def compute(self) -> "MetricResult":
        """Compute the metric value after fitting."""
        ...


class Preprocessor(Protocol):
    """Protocol for data preprocessing following Single Responsibility Principle."""
    
    def fit(self, data: "pd.DataFrame", spec: "DatasetSpec") -> Self:
        """Fit preprocessor to data and dataset specification."""
        ...
    
    def transform(self, data: "pd.DataFrame") -> "pd.DataFrame":
        """Transform data using fitted preprocessor."""
        ...
    
    def metadata(self) -> dict[str, Any]:
        """Return preprocessing metadata and transformations applied."""
        ...


class Reporter(Protocol):
    """Protocol for report generation following Interface Segregation Principle."""
    
    def render(self, run_summary: "RunSummary", report_spec: "ReportSpec") -> None:
        """Generate report from run summary according to specification."""
        ...


class MetricRegistry(Protocol):
    """Registry interface for metric management."""
    
    def register(self, metric_id: str, metric_class: type[Metric]) -> None:
        """Register a metric implementation."""
        ...
    
    def get(self, metric_id: str) -> type[Metric]:
        """Retrieve metric class by ID."""
        ...
    
    def list_ids(self, family: str | None = None) -> list[str]:
        """List available metric IDs, optionally filtered by family."""
        ...


class ReporterRegistry(Protocol):
    """Registry interface for reporter management."""
    
    def register(self, format_name: str, reporter_class: type[Reporter]) -> None:
        """Register a reporter implementation."""
        ...
    
    def get(self, format_name: str) -> type[Reporter]:
        """Retrieve reporter class by format name."""
        ...
    
    def list_formats(self) -> list[str]:
        """List available report formats."""
        ...


class PreprocessorRegistry(Protocol):
    """Registry interface for preprocessor management."""
    
    def register(self, preprocessor_name: str, preprocessor_class: type[Preprocessor]) -> None:
        """Register a preprocessor implementation."""
        ...
    
    def get(self, preprocessor_name: str) -> type[Preprocessor]:
        """Retrieve preprocessor class by name."""
        ...
    
    def list_names(self) -> list[str]:
        """List available preprocessor names."""
        ...