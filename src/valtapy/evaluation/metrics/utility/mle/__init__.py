"""Machine Learning Efficiency (MLE) metrics package.

This package contains ML efficiency evaluation metrics for synthetic data,
including individual techniques (TTR, TSTR, TTS, TRTS, TTRS) and the unified
MachineLearningEfficiency metric that aggregates all techniques.

Metrics Architecture:
- Uses Strategy pattern for flexible ML model selection
- Uses Template Method pattern for consistent evaluation flow
- Dependency injection for customizable components

Main Components:
- MachineLearningEfficiency: Unified metric evaluating all techniques
- Individual technique metrics: TTR, TSTR, TTS, TRTS, TTRS
- Base class for creating custom techniques

Note: Contracts and implementations are in the parent utility package.
"""

from .machine_learning_efficiency import MachineLearningEfficiency
from .base_ml_efficiency import BaseMLEfficiencyMetric
from .ttr import TTRMetric
from .tts import TTSMetric
from .trts import TRTSMetric
from .tstr import TSTRMetric
from .ttrs import TTRSMetric

__all__ = [
    # Main unified metric
    'MachineLearningEfficiency',

    # Individual technique metrics
    'TTRMetric',
    'TTSMetric',
    'TRTSMetric',
    'TSTRMetric',
    'TTRSMetric',

    # Base class for creating custom techniques
    'BaseMLEfficiencyMetric',
]
