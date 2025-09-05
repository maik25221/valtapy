"""Main orchestrator coordinating the evaluation pipeline."""

import logging
from pathlib import Path
from typing import Dict, Any, List, Callable
import pandas as pd

from ..domain.entities import RunSummary, MetricResult, ReportSpec
from ..infrastructure.runtime.config import load_config
from ..infrastructure.runtime.logging import get_logger
from ..infrastructure.runtime.cache import StatsStore
from ..infrastructure.runtime.parallel import run_in_parallel
from ..infrastructure.io.loaders import load_csv
from ..infrastructure.io.schema import align_schema
from ..infrastructure.metrics.registry import get_metric_registry
from ..infrastructure.reporting.registry import get_reporter_registry
from .plan_builder import PlanBuilder
from .aggregator import Aggregator
from .validators import validate_inputs


class Orchestrator:
    """
    Main orchestrator following the Facade pattern to coordinate evaluation pipeline.
    
    Responsibilities:
    - Load and validate configuration
    - Coordinate data loading and preprocessing  
    - Execute metrics in parallel with shared cache
    - Aggregate results and generate reports
    - Handle partial failures gracefully
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.plan_builder = PlanBuilder()
        self.aggregator = Aggregator()
    
    def run(self, config_path: str) -> RunSummary:
        """
        Execute complete evaluation pipeline from configuration.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            RunSummary containing all results and metadata
            
        Raises:
            ConfigError: If configuration is invalid
            SchemaError: If data schema validation fails
        """
        self.logger.info(f"Starting evaluation run with config: {config_path}")
        
        # Load and validate configuration
        config = load_config(config_path)
        self.logger.debug(f"Loaded configuration: {config}")
        
        # Build evaluation plan
        plan = self.plan_builder.build_from_config(config)
        plan_warnings = self.plan_builder.validate_plan(plan)
        if plan_warnings:
            for warning in plan_warnings:
                self.logger.warning(warning)
        
        # Load data
        real_data, synth_data, dataset_spec = self._load_and_prepare_data(config)
        self.logger.info(f"Loaded data: real={real_data.shape}, synth={synth_data.shape}")
        
        # Validate inputs
        validate_inputs(real_data, synth_data, dataset_spec, plan)
        
        # Initialize shared cache
        stats_store = StatsStore()
        
        # Prepare metric tasks
        metric_tasks = self._prepare_metric_tasks(plan, real_data, synth_data, stats_store)
        self.logger.info(f"Prepared {len(metric_tasks)} metric tasks")
        
        # Execute metrics in parallel
        results = run_in_parallel(metric_tasks, max_workers=0)  # 0 = auto-detect CPU count
        self.logger.info(f"Completed {len(results)} metrics")
        
        # Aggregate results
        aggregation_config = config.get("aggregation", {})
        weights = aggregation_config.get("weights")
        aggregates = self.aggregator.aggregate(results, plan, weights)
        
        # Create run summary
        artifacts = {
            "warnings": plan_warnings,
            "cache_stats": stats_store.get_stats(),
            "config_path": config_path
        }
        
        run_summary = RunSummary(
            plan=plan,
            results=results, 
            aggregates=aggregates,
            artifacts=artifacts
        )
        
        # Generate reports if specified
        if "report" in config:
            self._generate_reports(run_summary, config["report"])
        
        self.logger.info("Evaluation run completed successfully")
        return run_summary
    
    def _load_and_prepare_data(self, config: Dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, Any]:
        """Load and prepare datasets from configuration."""
        data_config = config["data"]
        
        # Load datasets
        real_data = load_csv(data_config["real"])
        synth_data = load_csv(data_config["synthetic"])
        
        # Create dataset spec
        from ..domain.entities import DatasetSpec
        dataset_spec = DatasetSpec(
            target=data_config.get("target"),
            dtypes=data_config.get("dtypes", {}),
            constraints=data_config.get("constraints", {})
        )
        
        # Align schemas (currently passthrough, TODO: implement proper alignment)
        real_aligned, synth_aligned, target_col = align_schema(real_data, synth_data, dataset_spec)
        
        return real_aligned, synth_aligned, dataset_spec
    
    def _prepare_metric_tasks(self, plan, real_data: pd.DataFrame, 
                            synth_data: pd.DataFrame, stats_store: StatsStore) -> List[Callable[[], MetricResult]]:
        """Prepare metric computation tasks for parallel execution."""
        tasks = []
        metric_registry = get_metric_registry()
        
        for metric_id in plan.metric_ids:
            task = self._create_metric_task(metric_id, real_data, synth_data, stats_store, metric_registry)
            tasks.append(task)
        
        return tasks
    
    def _create_metric_task(self, metric_id: str, real_data: pd.DataFrame, 
                          synth_data: pd.DataFrame, stats_store: StatsStore,
                          metric_registry) -> Callable[[], MetricResult]:
        """Create a single metric computation task."""
        
        def compute_metric() -> MetricResult:
            try:
                # Get metric class from registry
                metric_class = metric_registry.get(metric_id)
                
                # Create context with shared cache
                context = {"stats_store": stats_store, "seed": 42}  # TODO: use plan seed
                
                # Instantiate and compute metric
                metric = metric_class()
                metric.fit(real_data, synth_data, context)
                result = metric.compute()
                
                self.logger.debug(f"Computed metric {metric_id}: {result.value}")
                return result
                
            except Exception as e:
                self.logger.error(f"Failed to compute metric {metric_id}: {e}")
                
                # Return error result instead of failing completely
                return MetricResult(
                    id=metric_id,
                    value=0.0,
                    details={"error": str(e), "error_type": type(e).__name__},
                    family="fidelity",  # Default family for error results
                    purpose_tags=set()
                )
        
        return compute_metric
    
    def _generate_reports(self, run_summary: RunSummary, report_config: Dict[str, Any]) -> None:
        """Generate reports according to configuration."""
        try:
            report_spec = ReportSpec(
                formats=report_config["formats"],
                output_dir=report_config["output_dir"],
                include_details=report_config.get("include_details", True),
                include_artifacts=report_config.get("include_artifacts", False)
            )
            
            # Create output directory
            Path(report_spec.output_dir).mkdir(parents=True, exist_ok=True)
            
            # Generate each requested format
            reporter_registry = get_reporter_registry()
            for format_name in report_spec.formats:
                try:
                    reporter_class = reporter_registry.get(format_name)
                    reporter = reporter_class()
                    reporter.render(run_summary, report_spec)
                    self.logger.info(f"Generated {format_name} report")
                except Exception as e:
                    self.logger.error(f"Failed to generate {format_name} report: {e}")
                    
        except Exception as e:
            self.logger.error(f"Report generation failed: {e}")
            # Don't fail the entire run for report issues