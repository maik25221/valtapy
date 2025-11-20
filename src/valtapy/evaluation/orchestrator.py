"""Basic implementation of evaluation orchestrator."""

import pandas as pd
import time
from typing import List

from .contracts import EvaluationOrchestrator, MetricRegistry
from .entities import (
    EvaluationConfig, EvaluationResult, MetricExecutionContext,
    MetricResult, EvaluationSummary, MetricSpec, MetricFamily
)
from .exceptions import (
    EvaluationError, EvaluationConfigurationError, 
    MetricComputationError, UnsupportedDataError
)
from .registry import get_metric_registry


class BasicEvaluationOrchestrator:
    """Basic implementation of evaluation orchestrator."""
    
    def __init__(self, metric_registry: MetricRegistry = None):
        self._registry = metric_registry or get_metric_registry()
    
    def evaluate(
        self,
        real_df: pd.DataFrame,
        synth_df: pd.DataFrame,
        config: EvaluationConfig
    ) -> EvaluationResult:
        """Execute complete evaluation process."""
        start_time = time.time()
        
        try:
            # Validate configuration
            validation_issues = self.validate_config(config, real_df, synth_df)
            if validation_issues:
                raise EvaluationConfigurationError(
                    f"Configuration validation failed: {'; '.join(validation_issues)}"
                )
            
            # Initialize tracking
            results = []
            errors = []
            successful_count = 0
            failed_count = 0
            
            # Create execution context
            context = MetricExecutionContext(
                real_data=real_df,
                synth_data=synth_df,
                cache_enabled=config.cache_enabled,
                random_seed=config.random_seed
            )
            
            # Execute each metric
            for metric_spec in config.metrics:
                try:
                    result = self._execute_single_metric(metric_spec, context)
                    results.append(result)
                    successful_count += 1
                    
                except Exception as e:
                    error_msg = f"Metric {metric_spec.metric_id} failed: {str(e)}"
                    errors.append(error_msg)
                    failed_count += 1
            
            # Calculate family scores
            family_scores = self._calculate_family_scores(results)
            
            # Create summary
            total_time = time.time() - start_time
            summary = EvaluationSummary(
                total_metrics=len(config.metrics),
                successful_metrics=successful_count,
                failed_metrics=failed_count,
                total_time=total_time,
                fidelity_score=family_scores.get(MetricFamily.FIDELITY),
                utility_score=family_scores.get(MetricFamily.UTILITY),
                privacy_score=family_scores.get(MetricFamily.PRIVACY)
            )
            
            return EvaluationResult(
                real_data=real_df,
                synth_data=synth_df,
                config=config,
                results=results,
                summary=summary,
                errors=errors
            )
            
        except Exception as e:
            if isinstance(e, (EvaluationError, EvaluationConfigurationError)):
                raise
            else:
                raise EvaluationError(f"Evaluation failed: {str(e)}")
    
    def validate_config(
        self,
        config: EvaluationConfig,
        real_df: pd.DataFrame,
        synth_df: pd.DataFrame
    ) -> List[str]:
        """Validate evaluation configuration."""
        issues = []
        
        # Check basic data requirements
        if real_df.empty:
            issues.append("Real dataset is empty")
        
        if synth_df.empty:
            issues.append("Synthetic dataset is empty")
        
        # Check each metric specification
        for metric_spec in config.metrics:
            try:
                # Check if metric is registered
                if not self._registry.is_registered(metric_spec.metric_id):
                    available = self._registry.list_metrics()
                    issues.append(
                        f"Metric {metric_spec.metric_id} not registered. "
                        f"Available: {', '.join(available)}"
                    )
                    continue
                
                # Get metric and validate
                metric = self._registry.get_metric(metric_spec.metric_id)
                
                # Validate parameters
                if not metric.validate_parameters(metric_spec.parameters):
                    issues.append(f"Invalid parameters for metric {metric_spec.metric_id}")
                
                # Check if metric can handle the data
                if not metric.can_compute(real_df, synth_df):
                    issues.append(f"Metric {metric_spec.metric_id} cannot handle the provided data")
                    
            except Exception as e:
                issues.append(f"Error validating metric {metric_spec.metric_id}: {str(e)}")
        
        return issues
    
    def _execute_single_metric(
        self,
        metric_spec: MetricSpec,
        base_context: MetricExecutionContext
    ) -> MetricResult:
        """Execute a single metric."""
        try:
            # Get metric implementation
            metric = self._registry.get_metric(metric_spec.metric_id)
            
            # Create context with metric-specific parameters
            context = MetricExecutionContext(
                real_data=base_context.real_data,
                synth_data=base_context.synth_data,
                parameters=metric_spec.parameters,
                cache_enabled=base_context.cache_enabled,
                random_seed=base_context.random_seed
            )
            
            # Execute metric
            result = metric.compute(context)
            
            # Validate result
            if not isinstance(result, MetricResult):
                raise MetricComputationError(
                    metric_spec.metric_id,
                    "Metric did not return a MetricResult instance"
                )
            
            return result
            
        except Exception as e:
            if isinstance(e, (MetricComputationError, UnsupportedDataError)):
                raise
            else:
                raise MetricComputationError(metric_spec.metric_id, str(e))
    
    def _calculate_family_scores(self, results: List[MetricResult]) -> dict[MetricFamily, float]:
        """Calculate aggregated scores for each metric family."""
        family_scores = {}
        
        # Group results by family
        family_results = {}
        for result in results:
            if result.family not in family_results:
                family_results[result.family] = []
            family_results[result.family].append(result)
        
        # Calculate simple average for each family
        for family, family_result_list in family_results.items():
            if family_result_list:
                # Simple average of metric values
                values = [r.value for r in family_result_list if r.value is not None]
                if values:
                    family_scores[family] = sum(values) / len(values)
        
        return family_scores
    
    def get_supported_metrics(self) -> List[str]:
        """Get list of supported metric IDs."""
        return self._registry.list_metrics()
    
    def get_metrics_by_family(self, family: MetricFamily) -> List[str]:
        """Get metric IDs for a specific family."""
        return self._registry.list_metrics(family)