"""Domain layer: Core contracts, entities, and business logic."""

from .contracts import Metric, Preprocessor, Reporter, MetricRegistry, ReporterRegistry, PreprocessorRegistry
from .entities import DatasetSpec, EvalPlan, MetricResult, RunSummary, ReportSpec
from .taxonomy import FAMILIES, PURPOSES
from .errors import ConfigError, SchemaError, MetricExecutionError

__all__ = [
    "Metric", "Preprocessor", "Reporter", "MetricRegistry", "ReporterRegistry", "PreprocessorRegistry",
    "DatasetSpec", "EvalPlan", "MetricResult", "RunSummary", "ReportSpec",
    "FAMILIES", "PURPOSES",
    "ConfigError", "SchemaError", "MetricExecutionError"
]