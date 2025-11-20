# Machine Learning Efficiency Metric

## Overview

La métrica `MachineLearningEfficiency` es una métrica unificada y completa para evaluar la utilidad de datos sintéticos mediante técnicas de evaluación de eficiencia de Machine Learning.

## Características Principales

### 1. **Evaluación Integral**
La métrica ejecuta múltiples técnicas de evaluación de forma automática:

- **TTR** (Train on Training, Test on Real): Baseline que entrena y testea con datos reales
- **TSTR** (Train on Synthetic, Test on Real): Evalúa si los datos sintéticos pueden reemplazar los reales para entrenamiento
- **TTS** (Train on Training, Test on Synthetic): Evalúa la preservación de patrones
- **TRTS** (Train on Real, Test on Synthetic): Evalúa el match de distribución
- **TTRS** (Train on Training, Test on Real and Synthetic): Evalúa la consistencia cruzada

### 2. **Patrón Strategy para Modelos ML**
Utiliza el patrón de diseño Strategy mediante `MLModelFactory` para permitir flexibilidad en el modelo de evaluación:

- **DecisionTree** (por defecto)
- **RandomForest**
- **CatBoost**
- **XGBoost**
- Cualquier modelo compatible con sklearn

### 3. **Cálculo de Eficiencias Relativas**
Compara todas las técnicas contra el baseline (TTR) para calcular eficiencias relativas:

```
efficiency = technique_score / ttr_score
```

### 4. **Agregación Ponderada**
Colapsa todas las medidas en un score final usando pesos:

- TSTR: 50% (más importante - ¿puede reemplazar los datos de entrenamiento?)
- TRTS: 20% (match de distribución)
- TTS: 15% (preservación de patrones)
- TTRS: 15% (consistencia cruzada)

## Uso Básico

```python
from src.valtapy.evaluation.entities import MetricExecutionContext
from src.valtapy.evaluation.metrics.utility import MachineLearningEfficiency

# Crear la métrica con configuración por defecto (DecisionTree)
metric = MachineLearningEfficiency()

# Crear el contexto de ejecución
context = MetricExecutionContext(
    real_data=real_df,
    synth_data=synth_df,
    parameters={}
)

# Ejecutar la evaluación
result = metric.compute(context)

# Acceder a los resultados
print(f"Overall Efficiency: {result.value:.4f}")
print(f"Technique Scores: {result.details['technique_scores']}")
print(f"Relative Efficiencies: {result.details['relative_efficiencies']}")
```

## Uso Avanzado

### Estrategia de Modelo Personalizado

```python
from sklearn.ensemble import RandomForestClassifier

class RandomForestFactory:
    def create_classifier(self, random_state=42, **kwargs):
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=random_state
        )

    def create_regressor(self, random_state=42, **kwargs):
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=random_state
        )

# Usar la estrategia personalizada
metric = MachineLearningEfficiency(
    model_factory=RandomForestFactory()
)
```

### Técnicas Selectivas

```python
# Evaluar solo TTR y TSTR (más rápido)
metric = MachineLearningEfficiency(
    techniques=['ttr', 'tstr']
)
```

### Parámetros Personalizados

```python
metric = MachineLearningEfficiency(
    test_size=0.3,  # 30% para test
    random_seed=99,
    techniques=['ttr', 'tstr', 'tts']
)
```

## Estructura de Resultados

El `MetricResult` devuelto contiene:

### Valor Principal
```python
result.value  # Score de eficiencia agregado (0-1.5)
```

### Detalles Completos
```python
result.details = {
    'techniques_evaluated': ['ttr', 'tstr', 'tts', 'trts', 'ttrs'],
    'baseline_technique': 'ttr',
    'technique_scores': {
        'ttr': 0.85,
        'tstr': 0.75,
        'tts': 0.80,
        'trts': 0.78,
        'ttrs': 0.82
    },
    'relative_efficiencies': {
        'ttr': 1.0,      # Baseline
        'tstr': 0.882,   # 88.2% de la baseline
        'tts': 0.941,
        'trts': 0.918,
        'ttrs': 0.965
    },
    'task_type': 'classification',
    'primary_metric': 'accuracy',
    'aggregation_method': 'weighted_average',

    # Detalles por técnica
    'ttr_details': {
        'value': 0.85,
        'computation_time': 0.012,
        'samples': {'train': 800, 'test': 200}
    },
    # ... otros detalles
}
```

### Metadata
```python
result.metadata = {
    'test_size': 0.2,
    'random_seed': 42,
    'n_techniques': 5
}
```

## Interpretación de Resultados

### Eficiencia TSTR (más importante)
- **≥ 0.9**: Excelente - Los datos sintéticos son altamente efectivos
- **0.75-0.9**: Bueno - Capturan la mayoría de patrones importantes
- **0.6-0.75**: Moderado - Tienen utilidad pero hay margen de mejora
- **< 0.6**: Pobre - Pueden no ser adecuados para entrenamiento

### Score Agregado
- **≥ 0.85**: Datos sintéticos de muy alta calidad
- **0.70-0.85**: Alta calidad, útiles para la mayoría de propósitos
- **0.55-0.70**: Calidad moderada, útiles con limitaciones
- **< 0.55**: Calidad baja, considerar mejorar el generador

## Arquitectura

### Patrón Template Method
`BaseMLEfficiencyMetric` define el flujo común:
1. Validar datos
2. Preparar datasets
3. Entrenar modelo
4. Hacer predicciones
5. Calcular métricas

Cada técnica (TTR, TSTR, etc.) implementa `_prepare_train_test_data()` con su estrategia específica.

### Patrón Strategy
`MLModelFactory` permite inyectar diferentes modelos sin modificar el código de las métricas.

### Inyección de Dependencias
Todas las dependencias pueden ser inyectadas:
- `model_factory`: Para el modelo ML
- `classification_metrics`: Para métricas de clasificación
- `regression_metrics`: Para métricas de regresión
- `data_splitter`: Para división de datos

## Ejemplos

Ver `examples/ml_efficiency_example.py` para ejemplos completos de:
1. Uso básico con DecisionTree
2. Estrategia de modelo personalizado (RandomForest)
3. Evaluación selectiva de técnicas
4. Tareas de regresión
5. Inspección detallada de resultados

## Tests

Los tests completos están en `tests/unit/evaluation/test_ml_efficiency_metrics.py`:

```bash
# Ejecutar tests de MachineLearningEfficiency
pytest tests/unit/evaluation/test_ml_efficiency_metrics.py::TestMachineLearningEfficiency -v
```

## Ventajas sobre Métricas Individuales

1. **Evaluación Integral**: Una sola llamada evalúa múltiples aspectos
2. **Comparación Automática**: Calcula eficiencias relativas automáticamente
3. **Score Unificado**: Proporciona un indicador único de calidad
4. **Consistencia**: Usa el mismo modelo para todas las técnicas
5. **Flexibilidad**: Permite elegir técnicas y estrategias de modelo
6. **Interpretación**: Proporciona análisis comprehensivo en los detalles

## Relación con Métricas Individuales

Las métricas individuales (`TTRMetric`, `TSTRMetric`, etc.) siguen disponibles para:
- Evaluación específica de una técnica
- Casos donde solo se necesita una medida
- Debugging y análisis detallado

`MachineLearningEfficiency` internamente usa estas métricas y las combina de forma inteligente.
