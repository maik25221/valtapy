# ValtaPyV2

Prototipo básico para evaluación de datos sintéticos tabulares.

## ¿Qué funciona ahora?

**✅ Lo que está:**
- Cargar archivos CSV
- 6 métricas básicas que funcionan
- Tests para verificar que no se rompe

**❌ Lo que NO está:**
- CLI completo
- Configuración por archivos
- Reportes bonitos
- La mayoría de cosas que uno esperaría

## Uso Real (Lo Único que Funciona)

### 1. Cargar datos

```python
from valtapyV2.infrastructure.io.loaders import load_csv

real_df = load_csv("mis_datos_reales.csv")
synth_df = load_csv("mis_datos_sinteticos.csv")
```

### 2. Ejecutar una métrica

```python
from valtapyV2.infrastructure.metrics.registry import get_metric_registry
from valtapyV2.domain.entities import DatasetSpec

# Configurar contexto (necesario para métricas de utilidad)
context = {}
context["dataset_spec"] = DatasetSpec(target="columna_target")

# Elegir métrica
registry = get_metric_registry()
metric_class = registry.get("fidelity.ks")  # o "utility.accuracy", etc.
metric = metric_class()

# Ejecutar
metric.fit(real_df, synth_df, context)
resultado = metric.compute()

print(f"Resultado: {resultado.value}")
print(f"Detalles: {resultado.details}")
```

## Las 6 métricas que funcionan

- `fidelity.ks` - Test Kolmogorov-Smirnov
- `fidelity.correlation_delta` - Diferencias en correlaciones
- `utility.pmse` - Error predictivo (TSTR)
- `utility.accuracy` - Precisión en clasificación
- `privacy.nndr` - Ratios de distancia a vecinos
- `privacy.membership_inference` - Ataques de inferencia (básico)

## Probar que funciona

```bash
# Test básico - debería funcionar sin errores
python tests/basic_functionality_test.py

# Si quieres ver todos los tests
python -m pytest tests/unit/ -v
```

## ¿Cómo agregar una métrica?

```python
from valtapyV2.infrastructure.metrics.registry import register
from valtapyV2.infrastructure.metrics.base import MetricBase

@register("fidelity.mi_metrica")
class MiMetrica(MetricBase):
    name = "mi_metrica"
    family = "fidelity"
    purpose_tags = {"test"}
    
    def fit(self, real_data, synth_data, context):
        self._setup(real_data, synth_data, context)
        return self
    
    def compute(self):
        # Tu código aquí
        return MetricResult(
            id="fidelity.mi_metrica",
            value=0.5,  # tu cálculo
            details={},
            family="fidelity",
            purpose_tags={"test"}
        )
```

## Lo que falta (mucho)

- CLI que se pueda usar
- Leer configuración de archivos YAML
- Reportes en JSON/Markdown
- Paralelización
- Manejo de errores decente
- Documentación de las métricas
- Interfaz gráfica
- Muchas más métricas
- Optimizaciones de rendimiento