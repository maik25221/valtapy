"""Machine Learning Efficiency (MLE) - Unified ML utility evaluation metric.

This metric evaluates synthetic data utility by running multiple ML efficiency
techniques (TTR, TSTR, TTS, TRTS, TTRS) and aggregating their results.

It uses a strategy pattern to allow flexible model selection (DecisionTree,
CatBoost, RandomForest, etc.) and compares all techniques against the TTR
baseline to compute relative efficiency scores.
"""

from typing import Optional, Dict, Any
import time
import pandas as pd
import numpy as np

from ....contracts import Metric
from ....entities import MetricResult, MetricExecutionContext, MetricFamily
from ....exceptions import MetricComputationError
from ..ml_contracts import MLModelFactory, MetricsCalculator, DataSplitter
from ..ml_implementations import (
    DecisionTreeModelFactory,
    ClassificationMetricsCalculator,
    RegressionMetricsCalculator,
    SklearnDataSplitter,
    detect_task_type
)
from .ttr import TTRMetric
from .tstr import TSTRMetric
from .tts import TTSMetric
from .trts import TRTSMetric
from .ttrs import TTRSMetric


class MachineLearningEfficiency:
    """Unified Machine Learning Efficiency metric.

    This metric evaluates synthetic data utility by:
    1. Running multiple ML evaluation techniques (TTR, TSTR, TTS, TRTS, TTRS)
    2. Comparing each technique against the TTR baseline
    3. Computing relative efficiency scores
    4. Aggregating results into a comprehensive efficiency score

    The metric uses dependency injection for flexible model selection, allowing
    you to evaluate with different ML algorithms (DecisionTree, CatBoost, etc.)
    """

    def __init__(
        self,
        model_factory: Optional[MLModelFactory] = None,
        classification_metrics: Optional[MetricsCalculator] = None,
        regression_metrics: Optional[MetricsCalculator] = None,
        data_splitter: Optional[DataSplitter] = None,
        test_size: float = 0.2,
        random_seed: int = 42,
        techniques: Optional[list[str]] = None
    ):
        """Initialize the Machine Learning Efficiency metric.

        Args:
            model_factory: Factory for creating ML models (defaults to DecisionTree)
            classification_metrics: Calculator for classification metrics
            regression_metrics: Calculator for regression metrics
            data_splitter: Splitter for train/test data
            test_size: Proportion of data to use for testing (0.0-1.0)
            random_seed: Random seed for reproducibility
            techniques: List of techniques to evaluate. If None, uses all:
                       ['ttr', 'tstr', 'tts', 'trts', 'ttrs']
        """
        self.metric_id = "ml_efficiency"
        self.family = MetricFamily.UTILITY
        self.name = "Machine Learning Efficiency"
        self.description = (
            "Comprehensive ML utility evaluation that runs multiple techniques "
            "(TTR, TSTR, TTS, TRTS, TTRS) and aggregates their performance "
            "relative to the TTR baseline."
        )

        # Dependency injection with sensible defaults
        self.model_factory = model_factory or DecisionTreeModelFactory()
        self.classification_metrics = classification_metrics or ClassificationMetricsCalculator()
        self.regression_metrics = regression_metrics or RegressionMetricsCalculator()
        self.data_splitter = data_splitter or SklearnDataSplitter()

        self.test_size = test_size
        self.random_seed = random_seed

        # Configure which techniques to run
        self.techniques = techniques or ['ttr', 'tstr', 'tts', 'trts', 'ttrs']

        # Create technique instances (sharing injected dependencies)
        self._technique_instances = self._create_technique_instances()

    def _create_technique_instances(self) -> Dict[str, Any]:
        """Create instances of all evaluation techniques with shared dependencies.

        Returns:
            Dictionary mapping technique names to metric instances
        """
        common_kwargs = {
            'model_factory': self.model_factory,
            'classification_metrics': self.classification_metrics,
            'regression_metrics': self.regression_metrics,
            'data_splitter': self.data_splitter,
            'test_size': self.test_size,
            'random_seed': self.random_seed
        }

        instances = {}

        if 'ttr' in self.techniques:
            instances['ttr'] = TTRMetric(**common_kwargs)
        if 'tstr' in self.techniques:
            instances['tstr'] = TSTRMetric(**common_kwargs)
        if 'tts' in self.techniques:
            instances['tts'] = TTSMetric(**common_kwargs)
        if 'trts' in self.techniques:
            instances['trts'] = TRTSMetric(**common_kwargs)
        if 'ttrs' in self.techniques:
            instances['ttrs'] = TTRSMetric(**common_kwargs)

        return instances

    def compute(self, context: MetricExecutionContext) -> MetricResult:
        """Compute the comprehensive ML efficiency metric.

        This method:
        1. Runs all configured techniques
        2. Extracts their primary metric values
        3. Computes relative efficiencies compared to TTR baseline
        4. Aggregates into final efficiency score

        Args:
            context: Execution context with real data, synthetic data, and parameters

        Returns:
            MetricResult with aggregated efficiency score and detailed breakdown

        Raises:
            MetricComputationError: If computation fails
        """
        start_time = time.time()

        try:
            # Step 1: Run all techniques and collect results
            technique_results = self._run_all_techniques(context)

            # Step 2: Extract primary metric values
            technique_scores = self._extract_primary_scores(technique_results)

            # Step 3: Compute relative efficiencies vs TTR baseline
            efficiencies = self._compute_relative_efficiencies(technique_scores)

            # Step 4: Aggregate into final score
            aggregated_score = self._aggregate_efficiency_score(efficiencies)

            # Build comprehensive details
            details = {
                'techniques_evaluated': list(self.techniques),
                'baseline_technique': 'ttr',
                'technique_scores': technique_scores,
                'relative_efficiencies': efficiencies,
                'aggregation_method': 'weighted_average',
                'task_type': technique_results['ttr'].details.get('task_type') if 'ttr' in technique_results else 'unknown',
                'primary_metric': technique_results['ttr'].details.get('primary_metric') if 'ttr' in technique_results else 'unknown',
                'model_type': type(self.model_factory).__name__
            }

            # Add individual technique details
            for technique_name, result in technique_results.items():
                details[f'{technique_name}_details'] = {
                    'value': result.value,
                    'computation_time': result.computation_time,
                    'samples': {
                        'train': result.details.get('train_samples'),
                        'test': result.details.get('test_samples')
                    }
                }

            computation_time = time.time() - start_time

            return MetricResult(
                metric_id=self.metric_id,
                family=self.family,
                value=aggregated_score,
                details=details,
                metadata={
                    'test_size': self.test_size,
                    'random_seed': self.random_seed,
                    'n_techniques': len(self.techniques)
                },
                computation_time=computation_time
            )

        except Exception as e:
            raise MetricComputationError(
                metric_id=self.metric_id,
                reason=f"Failed to compute ML efficiency: {str(e)}"
            ) from e

    def _run_all_techniques(
        self,
        context: MetricExecutionContext
    ) -> Dict[str, MetricResult]:
        """Run all configured evaluation techniques.

        Args:
            context: Execution context

        Returns:
            Dictionary mapping technique names to their results
        """
        results = {}

        for technique_name, technique_instance in self._technique_instances.items():
            try:
                result = technique_instance.compute(context)
                results[technique_name] = result
            except Exception as e:
                raise MetricComputationError(
                    metric_id=self.metric_id,
                    reason=f"Failed to compute {technique_name}: {str(e)}"
                ) from e

        return results

    def _extract_primary_scores(
        self,
        technique_results: Dict[str, MetricResult]
    ) -> Dict[str, float]:
        """Extract primary metric scores from each technique result.

        Args:
            technique_results: Dictionary of technique results

        Returns:
            Dictionary mapping technique names to their primary scores
        """
        scores = {}
        for technique_name, result in technique_results.items():
            scores[technique_name] = result.value
        return scores

    def _compute_relative_efficiencies(
        self,
        technique_scores: Dict[str, float]
    ) -> Dict[str, float]:
        """Compute relative efficiency of each technique vs TTR baseline.

        The efficiency is computed as:
        - For metrics where higher is better (accuracy, r2_score):
          efficiency = technique_score / ttr_score
        - For metrics where lower is better (rmse, mae):
          efficiency = ttr_score / technique_score

        Args:
            technique_scores: Dictionary of technique scores

        Returns:
            Dictionary mapping technique names to relative efficiency (0-1+)
        """
        if 'ttr' not in technique_scores:
            raise ValueError("TTR baseline is required for efficiency computation")

        ttr_score = technique_scores['ttr']
        efficiencies = {}

        # TTR efficiency is always 1.0 (it's the baseline)
        efficiencies['ttr'] = 1.0

        # Determine if higher is better based on metric type
        # For now, assume higher is better (accuracy, r2_score)
        # This could be made configurable or detected from the metric calculator
        higher_is_better = True

        for technique_name, score in technique_scores.items():
            if technique_name == 'ttr':
                continue

            # Avoid division by zero
            if ttr_score == 0:
                efficiencies[technique_name] = 0.0
                continue

            if higher_is_better:
                # For accuracy, r2_score: technique / baseline
                efficiencies[technique_name] = score / ttr_score
            else:
                # For rmse, mae: baseline / technique
                # (lower scores are better, so we invert)
                if score == 0:
                    efficiencies[technique_name] = float('inf')
                else:
                    efficiencies[technique_name] = ttr_score / score

        return efficiencies

    def _aggregate_efficiency_score(
        self,
        efficiencies: Dict[str, float]
    ) -> float:
        """Aggregate individual efficiencies into final score.

        Uses weighted average with these weights:
        - TSTR: 0.5 (most important - can we replace training data?)
        - TRTS: 0.2 (distribution similarity)
        - TTS: 0.15 (pattern preservation)
        - TTRS: 0.15 (cross-consistency)
        - TTR: not included (it's the baseline)

        Args:
            efficiencies: Dictionary of relative efficiencies

        Returns:
            Aggregated efficiency score (0-1 range typically)
        """
        weights = {
            'tstr': 0.5,   # Most important: can synthetic replace training data?
            'trts': 0.2,   # Distribution match
            'tts': 0.15,   # Pattern preservation
            'ttrs': 0.15   # Cross-consistency
        }

        weighted_sum = 0.0
        total_weight = 0.0

        for technique_name, weight in weights.items():
            if technique_name in efficiencies:
                weighted_sum += efficiencies[technique_name] * weight
                total_weight += weight

        if total_weight == 0:
            return 0.0

        # Normalize to 0-1 range
        aggregated = weighted_sum / total_weight

        # Clip to reasonable range (efficiency can exceed 1.0 if synthetic is better)
        return min(max(aggregated, 0.0), 1.5)

    def can_compute(
        self,
        real_df: pd.DataFrame,
        synth_df: pd.DataFrame
    ) -> bool:
        """Check if this metric can be computed for the given datasets.

        Args:
            real_df: Real dataset
            synth_df: Synthetic dataset

        Returns:
            True if metric can be computed, False otherwise
        """
        # Check if any technique can compute
        for technique in self._technique_instances.values():
            if technique.can_compute(real_df, synth_df):
                return True
        return False

    def validate_parameters(self, parameters: dict) -> bool:
        """Validate that parameters are correct for this metric.

        Args:
            parameters: Dictionary of parameters

        Returns:
            True if parameters are valid, False otherwise
        """
        if 'test_size' in parameters:
            test_size = parameters['test_size']
            if not isinstance(test_size, (int, float)) or not 0 < test_size < 1:
                return False

        if 'random_seed' in parameters:
            random_seed = parameters['random_seed']
            if not isinstance(random_seed, int) or random_seed < 0:
                return False

        if 'techniques' in parameters:
            techniques = parameters['techniques']
            if not isinstance(techniques, list):
                return False
            valid_techniques = {'ttr', 'tstr', 'tts', 'trts', 'ttrs'}
            if not all(t in valid_techniques for t in techniques):
                return False

        return True
