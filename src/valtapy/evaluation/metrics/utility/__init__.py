"""Utility metrics for synthetic data evaluation.

This module contains metrics for evaluating the utility of synthetic data,
particularly focusing on ML efficiency metrics that assess how well synthetic
data can be used for machine learning tasks.

ML Efficiency Metrics (in mle subpackage):
- MachineLearningEfficiency: Comprehensive unified metric evaluating all techniques
- TTR: Train on Training (real), Test on Real - Baseline metric
- TTS: Train on Training (real), Test on Synthetic - Pattern similarity
- TRTS: Train on Real (full), Test on Synthetic - Distribution match
- TSTR: Train on Synthetic, Test on Real - Generalization capability (key metric)
- TTRS: Train on Training (real), Test on Real and Synthetic - Cross-data consistency

ML Contracts & Implementations:
- MLModel, MLModelFactory, MetricsCalculator, DataSplitter: Protocols for extensibility
- DecisionTreeModelFactory, ClassificationMetricsCalculator, etc.: Default implementations
"""

# Import ML efficiency metrics from the mle subpackage
from .mle import (
    MachineLearningEfficiency,
    BaseMLEfficiencyMetric,
    TTRMetric,
    TTSMetric,
    TRTSMetric,
    TSTRMetric,
    TTRSMetric,
)

# Import ML contracts and implementations from utility level
from .ml_contracts import (
    MLModel,
    MLModelFactory,
    MetricsCalculator,
    DataSplitter,
)

from .ml_implementations import (
    DecisionTreeModelFactory,
    ClassificationMetricsCalculator,
    RegressionMetricsCalculator,
    SklearnDataSplitter,
    detect_task_type,
)

__all__ = [
    # Unified comprehensive metric
    'MachineLearningEfficiency',

    # Individual technique metrics
    'TTRMetric',
    'TTSMetric',
    'TRTSMetric',
    'TSTRMetric',
    'TTRSMetric',

    # Base class
    'BaseMLEfficiencyMetric',

    # Contracts (for extending with new model types)
    'MLModel',
    'MLModelFactory',
    'MetricsCalculator',
    'DataSplitter',

    # Implementations (for direct use or customization)
    'DecisionTreeModelFactory',
    'ClassificationMetricsCalculator',
    'RegressionMetricsCalculator',
    'SklearnDataSplitter',
    'detect_task_type',
]
