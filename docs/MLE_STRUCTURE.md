# MLE (Machine Learning Efficiency) Package Structure

## Overview

El paquete MLE contiene todas las métricas de eficiencia de Machine Learning para evaluar la utilidad de datos sintéticos. Los contracts e implementations están en el nivel de `utility`, mientras que las métricas específicas están organizadas en el subpaquete `mle/`.

## Directory Structure

```
src/valtapy/evaluation/metrics/utility/
├── __init__.py                     # Exports from mle + contracts + implementations
├── ml_contracts.py                 # Protocols/interfaces (Strategy pattern)
├── ml_implementations.py           # Default implementations (DecisionTree, sklearn)
└── mle/                            # ML Efficiency metrics package
    ├── __init__.py                 # Metric exports
    ├── machine_learning_efficiency.py   # Unified MLE metric
    ├── base_ml_efficiency.py       # Base class for all techniques
    ├── ttr.py                      # Train on Training, Test on Real
    ├── tts.py                      # Train on Training, Test on Synthetic
    ├── trts.py                     # Train on Real, Test on Synthetic
    ├── tstr.py                     # Train on Synthetic, Test on Real
    └── ttrs.py                     # Train on Training, Test on Real+Synthetic
```

## Components

### Utility Level (Shared)
- **`ml_contracts.py`**: Protocolos/interfaces para extensibilidad (Strategy pattern)
  - `MLModel`, `MLModelFactory`, `MetricsCalculator`, `DataSplitter`
- **`ml_implementations.py`**: Implementaciones por defecto
  - `DecisionTreeModelFactory`, `ClassificationMetricsCalculator`, etc.

### MLE Package (Metrics)
- **`machine_learning_efficiency.py`**: La métrica unificada que ejecuta todas las técnicas y agrega resultados
- **`base_ml_efficiency.py`**: Clase base con Template Method pattern

### Individual Techniques (in mle/)
- **`ttr.py`**: Baseline - entrena y testea con datos reales
- **`tstr.py`**: La métrica más importante - ¿pueden los sintéticos reemplazar los reales?
- **`tts.py`**: Evalúa preservación de patrones
- **`trts.py`**: Evalúa match de distribución
- **`ttrs.py`**: Evalúa consistencia cruzada

## Import Paths

### From Outside the Package
```python
# Importar desde utility (recomendado)
from src.valtapy.evaluation.metrics.utility import (
    MachineLearningEfficiency,
    TTRMetric,
    TSTRMetric,
    # ... otros
)

# Importar directamente desde mle (también válido)
from src.valtapy.evaluation.metrics.utility.mle import (
    MachineLearningEfficiency,
    TTRMetric,
    TSTRMetric,
)
```

### Within the MLE Package
```python
# Los archivos dentro de mle/ usan imports relativos
from .base_ml_efficiency import BaseMLEfficiencyMetric
from ..ml_contracts import MLModelFactory  # Import from parent (utility)
from ....entities import MetricResult     # 4 niveles hasta evaluation/
```

## Module Hierarchy

```
valtapy/
└── evaluation/
    ├── contracts.py
    ├── entities.py
    ├── exceptions.py
    └── metrics/
        └── utility/
            ├── __init__.py              # Exports mle.* + contracts + implementations
            ├── ml_contracts.py          # Protocols at utility level
            ├── ml_implementations.py    # Implementations at utility level
            └── mle/
                ├── __init__.py          # Exports MLE metrics
                ├── machine_learning_efficiency.py
                ├── base_ml_efficiency.py
                └── ttr.py, tts.py, trts.py, tstr.py, ttrs.py
```

## Design Patterns

### 1. Template Method Pattern
`BaseMLEfficiencyMetric` define el flujo de evaluación:
- Validación de datos
- Preparación de datasets
- Entrenamiento de modelo
- Predicciones
- Cálculo de métricas

Cada técnica (TTR, TSTR, etc.) implementa solo `_prepare_train_test_data()`.

### 2. Strategy Pattern
`MLModelFactory` permite inyectar diferentes modelos:
```python
# DecisionTree (default)
metric = MachineLearningEfficiency()

# RandomForest custom
metric = MachineLearningEfficiency(model_factory=RandomForestFactory())
```

### 3. Dependency Injection
Todas las dependencias pueden ser inyectadas:
- `model_factory`: Para el modelo ML
- `classification_metrics`: Para métricas de clasificación
- `regression_metrics`: Para métricas de regresión
- `data_splitter`: Para división de datos

## Usage Examples

### Basic Usage
```python
from src.valtapy.evaluation.metrics.utility import MachineLearningEfficiency
from src.valtapy.evaluation.entities import MetricExecutionContext

metric = MachineLearningEfficiency()
context = MetricExecutionContext(real_data=real_df, synth_data=synth_df)
result = metric.compute(context)
```

### Individual Techniques
```python
from src.valtapy.evaluation.metrics.utility import TSTRMetric, TTRMetric

# Solo TSTR
tstr_metric = TSTRMetric()
tstr_result = tstr_metric.compute(context)

# Solo TTR (baseline)
ttr_metric = TTRMetric()
ttr_result = ttr_metric.compute(context)
```

### Custom Strategy
```python
from src.valtapy.evaluation.metrics.utility.mle import (
    MachineLearningEfficiency,
    MLModelFactory
)

class CatBoostFactory:
    def create_classifier(self, random_state=42, **kwargs):
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=100, random_state=random_state)

metric = MachineLearningEfficiency(model_factory=CatBoostFactory())
```

## Testing

Todos los tests están en `tests/unit/evaluation/test_ml_efficiency_metrics.py`:

```bash
# Ejecutar todos los tests de MLE
pytest tests/unit/evaluation/test_ml_efficiency_metrics.py -v

# Solo MachineLearningEfficiency
pytest tests/unit/evaluation/test_ml_efficiency_metrics.py::TestMachineLearningEfficiency -v

# Solo técnicas individuales
pytest tests/unit/evaluation/test_ml_efficiency_metrics.py::TestTSTRMetric -v
```

## Migration Guide

### Si usabas imports antiguos (antes de la reorganización):
```python
# ANTES (ya no funciona)
from src.valtapy.evaluation.metrics.utility.tstr import TSTRMetric

# AHORA (funciona igual)
from src.valtapy.evaluation.metrics.utility import TSTRMetric

# O directamente desde mle
from src.valtapy.evaluation.metrics.utility.mle import TSTRMetric
```

**Nota**: Los imports desde `utility` siguen funcionando igual, la API pública no cambió.

## Benefits of This Structure

### 1. **Separation of Concerns**
- **Contracts & Implementations** (utility level): Compartidos y reutilizables por otras métricas
- **MLE Metrics** (mle/ subfolder): Solo las métricas específicas de ML efficiency

### 2. **Reusability**
Los contracts (`MLModelFactory`, etc.) pueden ser usados por:
- Métricas de MLE
- Futuras métricas de utility que necesiten ML
- Otras familias de métricas (privacy, fidelity) si lo necesitan

### 3. **Clean Organization**
```
utility/
├── ml_contracts.py          ← Shared contracts
├── ml_implementations.py    ← Shared implementations
└── mle/                     ← Specific MLE metrics
    ├── machine_learning_efficiency.py
    └── ttr.py, tstr.py, etc.
```

### 4. **Extensibility**
- Agregar nuevas técnicas MLE: solo agregar en `mle/`
- Agregar nuevas factories: solo modificar `ml_implementations.py`
- Otros tipos de métricas pueden usar los mismos contracts

### 5. **Backward Compatibility**
```python
# Sigue funcionando igual desde fuera
from src.valtapy.evaluation.metrics.utility import (
    MachineLearningEfficiency,
    TSTRMetric,
    MLModelFactory  # ← Contracts también disponibles
)
```

## Future Extensions

Posibles adiciones al paquete MLE:

1. **Nuevas técnicas**: TSTTR, TTRTS, etc.
2. **Factories adicionales**: XGBoostFactory, LightGBMFactory
3. **Métricas especializadas**: Para series temporales, imágenes, etc.
4. **Benchmark suite**: Comparación automática entre modelos
5. **Visualización**: Gráficos comparativos de eficiencias

## Related Documentation

- `ML_EFFICIENCY_METRIC.md`: Guía detallada de uso de MachineLearningEfficiency
- `examples/ml_efficiency_example.py`: Ejemplos de uso completos
- `tests/unit/evaluation/test_ml_efficiency_metrics.py`: Suite de tests
