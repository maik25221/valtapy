# MLE Architecture Overview

## Visual Structure

```
┌─────────────────────────────────────────────────────────────┐
│                   utility/ Package                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │          Shared Contracts & Implementations          │  │
│  ├──────────────────────────────────────────────────────┤  │
│  │  ml_contracts.py                                     │  │
│  │    ├─ MLModel (Protocol)                            │  │
│  │    ├─ MLModelFactory (Protocol)                     │  │
│  │    ├─ MetricsCalculator (Protocol)                  │  │
│  │    └─ DataSplitter (Protocol)                       │  │
│  │                                                       │  │
│  │  ml_implementations.py                               │  │
│  │    ├─ DecisionTreeModelFactory                      │  │
│  │    ├─ ClassificationMetricsCalculator               │  │
│  │    ├─ RegressionMetricsCalculator                   │  │
│  │    ├─ SklearnDataSplitter                           │  │
│  │    └─ detect_task_type()                            │  │
│  └──────────────────────────────────────────────────────┘  │
│                            ▲                                │
│                            │ uses                           │
│                            │                                │
│  ┌─────────────────────────┴──────────────────────────┐   │
│  │              mle/ Subpackage                        │   │
│  ├──────────────────────────────────────────────────────┤  │
│  │  Base Classes                                       │   │
│  │    └─ base_ml_efficiency.py                        │   │
│  │         (Template Method Pattern)                   │   │
│  │                                                      │   │
│  │  Unified Metric                                     │   │
│  │    └─ machine_learning_efficiency.py                │   │
│  │         (Aggregates all techniques)                 │   │
│  │                                                      │   │
│  │  Individual Techniques                              │   │
│  │    ├─ ttr.py   (Train Training, Test Real)         │   │
│  │    ├─ tstr.py  (Train Synthetic, Test Real) ⭐     │   │
│  │    ├─ tts.py   (Train Training, Test Synthetic)    │   │
│  │    ├─ trts.py  (Train Real, Test Synthetic)        │   │
│  │    └─ ttrs.py  (Train Training, Test Real+Synth)   │   │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Dependency Flow

```
┌─────────────────────────┐
│   User Application      │
└───────────┬─────────────┘
            │ imports
            ▼
┌─────────────────────────┐
│  utility/__init__.py    │  ← Re-exports everything
│  (Public API)           │
└─────┬──────────┬────────┘
      │          │
      ▼          ▼
┌─────────┐  ┌──────────────────┐
│ Contracts│  │  mle/__init__.py │
│ & Impls  │  │  (MLE Metrics)   │
└─────────┘  └───────┬──────────┘
      ▲              │
      │              │ imports
      │              ▼
      │      ┌────────────────────────┐
      │      │  Individual Techniques │
      │      │  (ttr, tstr, etc.)     │
      │      └───────┬────────────────┘
      │              │
      └──────────────┘ uses
```

## Design Patterns Applied

### 1. Strategy Pattern (Contracts)

```python
# ml_contracts.py
class MLModelFactory(Protocol):
    def create_classifier(...) -> Model
    def create_regressor(...) -> Model

# User can inject any factory
metric = MachineLearningEfficiency(
    model_factory=CustomFactory()  # Strategy injection
)
```

### 2. Template Method Pattern (Base)

```python
# base_ml_efficiency.py
class BaseMLEfficiencyMetric:
    def compute(self, context):
        # Template method defines the flow
        self._validate_data(...)
        train_X, train_y, test_X, test_y = self._prepare_train_test_data(...)  # ← Subclass implements
        model = self._train_model(...)
        predictions = self._make_predictions(...)
        return self._calculate_metrics(...)

# Each technique implements only the specific part
class TSTRMetric(BaseMLEfficiencyMetric):
    def _prepare_train_test_data(self, ...):
        # TSTR-specific data preparation
        return synth_train, real_test
```

### 3. Dependency Injection

```python
# All dependencies can be injected
metric = MachineLearningEfficiency(
    model_factory=custom_factory,           # Strategy
    classification_metrics=custom_calc,     # Calculator
    regression_metrics=custom_reg_calc,     # Calculator
    data_splitter=custom_splitter,          # Splitter
    test_size=0.3,                          # Config
    random_seed=42                          # Config
)
```

## Component Responsibilities

### Utility Level (Shared Infrastructure)

#### `ml_contracts.py`
- **Responsibility**: Define protocols/interfaces
- **Used by**: All MLE metrics
- **Purpose**: Enable Strategy pattern and extensibility

#### `ml_implementations.py`
- **Responsibility**: Provide default implementations
- **Used by**: All MLE metrics (as defaults)
- **Purpose**: Working implementations out of the box

### MLE Package (Domain-Specific Metrics)

#### `base_ml_efficiency.py`
- **Responsibility**: Common ML evaluation logic (Template Method)
- **Used by**: All technique metrics (TTR, TSTR, etc.)
- **Purpose**: Avoid code duplication

#### `machine_learning_efficiency.py`
- **Responsibility**: Unified metric that runs all techniques
- **Uses**: All individual technique metrics
- **Purpose**: Single-call comprehensive evaluation

#### Individual Techniques (`ttr.py`, `tstr.py`, etc.)
- **Responsibility**: Implement specific data preparation strategy
- **Extends**: `BaseMLEfficiencyMetric`
- **Purpose**: Different ways to evaluate synthetic data utility

## Import Resolution Examples

### From User Code
```python
from src.valtapy.evaluation.metrics.utility import MachineLearningEfficiency
# Resolves to: utility/__init__.py → mle/__init__.py → mle/machine_learning_efficiency.py
```

### Within MLE Metrics
```python
# In mle/tstr.py
from .base_ml_efficiency import BaseMLEfficiencyMetric    # Same package
from ..ml_contracts import MLModelFactory                 # Parent package (utility)
from ....entities import MetricResult                     # 4 levels up to evaluation/
```

## Key Benefits

| Aspect | Benefit |
|--------|---------|
| **Modularity** | Contracts separate from metrics implementation |
| **Reusability** | Other metrics can use ML contracts |
| **Extensibility** | Easy to add new techniques or factories |
| **Testability** | Can mock contracts for testing |
| **Clean Imports** | Public API unchanged for users |
| **Single Responsibility** | Each component has one clear purpose |

## Future Extensibility

### Adding a New Technique (e.g., TSTTR)

1. Create `mle/tsttr.py`
2. Inherit from `BaseMLEfficiencyMetric`
3. Implement `_prepare_train_test_data()`
4. Export from `mle/__init__.py` and `utility/__init__.py`

**No changes needed** to contracts or implementations!

### Adding a New Model Strategy (e.g., CatBoost)

1. Create class implementing `MLModelFactory` protocol
2. Add to `ml_implementations.py` (optional, can be external)
3. Use: `MachineLearningEfficiency(model_factory=CatBoostFactory())`

**No changes needed** to any metrics!

### Supporting New Metric Types (e.g., AUC-based)

1. Create class implementing `MetricsCalculator` protocol
2. Add to `ml_implementations.py`
3. Use: `MachineLearningEfficiency(classification_metrics=AUCCalculator())`

**No changes needed** to core logic!

## Comparison: Before vs After

### Before Reorganization
```
utility/
├── ttr.py
├── tstr.py
├── tts.py
├── trts.py
├── ttrs.py
├── base_ml_efficiency.py
├── machine_learning_efficiency.py
├── ml_contracts.py
└── ml_implementations.py
```
❌ All files at same level (cluttered)
❌ No clear separation of concerns
❌ Contracts mixed with metrics

### After Reorganization
```
utility/
├── ml_contracts.py         ← Shared
├── ml_implementations.py   ← Shared
└── mle/                    ← MLE-specific
    ├── machine_learning_efficiency.py
    ├── base_ml_efficiency.py
    └── ttr.py, tstr.py, tts.py, trts.py, ttrs.py
```
✅ Clean separation
✅ Reusable contracts at utility level
✅ MLE metrics grouped together
✅ Easy to add more subpackages (e.g., `privacy/`, `fidelity/`)

---

This architecture follows SOLID principles and makes the codebase maintainable and extensible! 🎯
