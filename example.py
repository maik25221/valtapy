#!/usr/bin/env python3
"""
ValtaPyV2 - Ejemplo de uso del framework de evaluación de datos sintéticos

Este ejemplo demuestra el workflow completo usando la arquitectura del proyecto:
1. Uso de las entidades del dominio (DatasetSpec, EvalPlan, MetricResult)
2. Carga de datos a través de la capa de infraestructura
3. Ejecución usando las clases de aplicación (PlanBuilder, Aggregator)
4. Presentación de resultados estructurados
"""

import sys
import tempfile
import os
from pathlib import Path

# Add src to path for imports  
sys.path.append(str(Path(__file__).parent / "src"))

# Domain layer - Entidades y contratos
from valtapyV2.domain.entities import DatasetSpec, EvalPlan, MetricResult, RunSummary
from valtapyV2.domain.contracts import Metric

# Infrastructure layer - Implementaciones concretas probadas
from valtapyV2.infrastructure.io.loaders import load_csv
from valtapyV2.infrastructure.metrics.registry import get_metric_registry
from valtapyV2.infrastructure.runtime.cache import StatsStore


def create_example_datasets():
    """Crear datasets de ejemplo para la demostración."""
    
    # Datos reales simulados - dataset tabular típico
    real_data_content = """age,income,score,education,risk_category
25,45000,0.73,bachelor,low
32,62000,0.81,master,medium  
45,78000,0.69,bachelor,high
28,51000,0.85,phd,low
38,67000,0.74,master,medium
52,89000,0.62,bachelor,high
29,48000,0.88,bachelor,low
41,71000,0.71,master,medium
35,59000,0.79,phd,low
47,82000,0.65,master,high
"""

    # Datos sintéticos (similares con variación realista)
    synth_data_content = """age,income,score,education,risk_category
26,46200,0.75,bachelor,low
31,63500,0.79,master,medium
44,76800,0.71,bachelor,high  
27,52100,0.83,phd,low
39,68200,0.72,master,medium
51,87500,0.64,bachelor,high
30,49300,0.86,bachelor,low
42,72800,0.69,master,medium
36,60500,0.77,phd,low
48,84200,0.63,master,high
"""
    
    # Crear archivos temporales
    real_fd, real_path = tempfile.mkstemp(suffix='.csv', prefix='example_real_')
    synth_fd, synth_path = tempfile.mkstemp(suffix='.csv', prefix='example_synth_')
    
    try:
        with os.fdopen(real_fd, 'w') as f:
            f.write(real_data_content)
        
        with os.fdopen(synth_fd, 'w') as f:
            f.write(synth_data_content)
        
        return real_path, synth_path
        
    except:
        # Limpiar en caso de error
        try:
            os.unlink(real_path)
            os.unlink(synth_path)
        except:
            pass
        raise


class ValtaPyV2Example:
    """
    Ejemplo que demuestra el uso correcto de la arquitectura ValtaPyV2.
    
    Esta clase encapsula el workflow usando las capas apropiadas:
    - Domain: Entidades inmutables (DatasetSpec, EvalPlan, MetricResult)
    - Application: Lógica de negocio (PlanBuilder, Aggregator)
    - Infrastructure: Implementaciones concretas (loaders, registry, cache)
    """
    
    def __init__(self):
        """Inicializar el ejemplo con componentes funcionales."""
        self.stats_store = StatsStore()
        
    def create_dataset_spec(self) -> DatasetSpec:
        """Crear especificación del dataset usando entidades del dominio."""
        return DatasetSpec(
            target="risk_category",
            dtypes={
                "age": "int64",
                "income": "int64",
                "score": "float64", 
                "education": "object",
                "risk_category": "object"
            },
            constraints={
                "age": {"min": 18, "max": 100},
                "income": {"min": 0},
                "score": {"min": 0.0, "max": 1.0}
            }
        )
    
    def create_evaluation_plan(self) -> EvalPlan:
        """Crear plan de evaluación usando entidades del dominio."""
        return EvalPlan(
            metric_ids=[
                "fidelity.ks",
                "fidelity.correlation_delta", 
                "utility.pmse",
                "privacy.nndr"
            ],
            seed=42,
            cv_splits=3,
            purpose="synthetic_data_evaluation"
        )
    
    def load_datasets(self, real_path: str, synth_path: str):
        """Cargar datasets usando la capa de infraestructura."""
        print("2. CARGA DE DATASETS (Infrastructure Layer)")
        print("-" * 50)
        
        # Usar loader de infraestructura
        real_df = load_csv(real_path)
        synth_df = load_csv(synth_path)
        
        print(f"[OK] Dataset real cargado: {len(real_df)} filas, {len(real_df.columns)} columnas")
        print(f"[OK] Dataset sintético cargado: {len(synth_df)} filas, {len(synth_df.columns)} columnas")
        print(f"[INFO] Columnas detectadas: {list(real_df.columns)}")
        
        return real_df, synth_df
    
    def execute_metrics(self, real_df, synth_df, dataset_spec: DatasetSpec, eval_plan: EvalPlan) -> list[MetricResult]:
        """Ejecutar métricas usando el registry de infraestructura."""
        print("\n4. EJECUCIÓN DE MÉTRICAS (Infrastructure + Domain)")
        print("-" * 50)
        
        # Obtener registry de métricas 
        registry = get_metric_registry()
        
        # Crear contexto para métricas
        context = {
            "dataset_spec": dataset_spec,
            "stats_store": self.stats_store,
            "seed": eval_plan.seed,
            "cv_splits": eval_plan.cv_splits
        }
        
        results = []
        
        for metric_id in eval_plan.metric_ids:
            print(f"\n[EXEC] Ejecutando métrica: {metric_id}")
            
            try:
                # Obtener clase de métrica del registry
                metric_class = registry.get(metric_id)
                metric: Metric = metric_class()
                
                print(f"       [INFO] Familia: {metric.family}")
                print(f"       [INFO] Propósitos: {', '.join(metric.purpose_tags)}")
                
                # Ejecutar métrica usando el protocolo
                metric.fit(real_df, synth_df, context)
                result: MetricResult = metric.compute()
                
                # Validar que es una MetricResult válida
                assert isinstance(result, MetricResult)
                assert result.id == metric_id
                assert result.family in {"fidelity", "utility", "privacy"}
                
                results.append(result)
                print(f"       [OK] Resultado: {result.value:.4f}")
                
                if "error" in result.details:
                    print(f"       [WARNING] {result.details['error']}")
                
            except Exception as e:
                print(f"       [ERROR] Falló ejecución de {metric_id}: {str(e)}")
                # Crear resultado de error usando entidad del dominio
                error_result = MetricResult(
                    id=metric_id,
                    value=0.0,
                    details={"error": str(e)},
                    family="fidelity",  # Default family
                    purpose_tags=set()
                )
                results.append(error_result)
        
        return results
    
    def aggregate_results(self, eval_plan: EvalPlan, results: list[MetricResult]) -> RunSummary:
        """Agregar resultados de forma básica."""
        print("\n5. AGREGACIÓN DE RESULTADOS")
        print("-" * 50)
        
        # Agregación básica por familia
        aggregates = {}
        for family in ["fidelity", "utility", "privacy"]:
            family_results = [r for r in results if r.family == family and "error" not in r.details]
            if family_results:
                family_score = sum(r.value for r in family_results) / len(family_results)
                aggregates[f"{family}_score"] = family_score
                aggregates[f"{family}_count"] = len(family_results)
        
        # Score general
        all_successful = [r for r in results if "error" not in r.details]
        if all_successful:
            aggregates["overall_score"] = sum(r.value for r in all_successful) / len(all_successful)
        
        # Crear resumen usando entidad del dominio
        run_summary = RunSummary(
            plan=eval_plan,
            results=results,
            aggregates=aggregates,
            artifacts={
                "execution_timestamp": "2025-01-12",
                "framework_version": "ValtaPyV2-alpha",
                "total_metrics": len(results),
                "successful_metrics": len(all_successful)
            }
        )
        
        print(f"[OK] Agregación completada")
        print(f"[INFO] Resultados por familia:")
        for family in ["fidelity", "utility", "privacy"]:
            family_results = run_summary.get_results_by_family(family)
            family_score = run_summary.get_family_score(family)
            print(f"       - {family.upper()}: {len(family_results)} métricas, score: {family_score:.4f}")
        
        return run_summary
    
    def run(self) -> RunSummary:
        """Ejecutar el ejemplo completo usando la arquitectura del proyecto."""
        
        print("=" * 70)
        print("ValtaPyV2 - Ejemplo de Arquitectura en Capas")
        print("=" * 70)
        print()
        
        # 1. Crear datasets temporales
        print("1. PREPARACIÓN DE DATOS")
        print("-" * 50)
        
        real_path, synth_path = create_example_datasets()
        print(f"[OK] Archivos temporales creados")
        print(f"     - Real: {os.path.basename(real_path)}")
        print(f"     - Sintético: {os.path.basename(synth_path)}")
        
        try:
            # 2. Cargar datasets (Infrastructure Layer)
            real_df, synth_df = self.load_datasets(real_path, synth_path)
            
            # 3. Crear entidades del dominio
            print("\n3. ENTIDADES DEL DOMINIO (Domain Layer)")
            print("-" * 50)
            
            dataset_spec = self.create_dataset_spec()
            eval_plan = self.create_evaluation_plan()
            
            print(f"[OK] DatasetSpec creado - Target: {dataset_spec.target}")
            print(f"[OK] EvalPlan creado - {len(eval_plan.metric_ids)} métricas planificadas")
            print(f"[INFO] Métricas: {', '.join(eval_plan.metric_ids)}")
            
            # 4. Ejecutar métricas (Infrastructure + Domain)
            results = self.execute_metrics(real_df, synth_df, dataset_spec, eval_plan)
            
            # 5. Agregar resultados (Application Layer)
            run_summary = self.aggregate_results(eval_plan, results)
            
            # 6. Mostrar resumen final
            print("\n6. RESUMEN FINAL")
            print("=" * 70)
            
            successful_results = [r for r in results if "error" not in r.details]
            failed_results = [r for r in results if "error" in r.details]
            
            print(f"\n[RESULTADOS] {len(successful_results)}/{len(results)} métricas exitosas")
            print("-" * 50)
            
            for result in successful_results:
                family_label = f"[{result.family.upper()}]"
                print(f"{family_label:12} {result.id}: {result.value:.4f}")
                
                # Interpretación básica
                if result.family == "fidelity" and result.value > 0.8:
                    print(f"             Alta similitud con datos reales")
                elif result.family == "utility" and result.value < 0.2:
                    print(f"             Buena preservación de utilidad")  
                elif result.family == "privacy" and result.value > 0.1:
                    print(f"             Nivel adecuado de protección")
            
            if failed_results:
                print(f"\n[ERRORES] {len(failed_results)} métricas fallidas:")
                for result in failed_results:
                    print(f"  - {result.id}: {result.details.get('error', 'Error desconocido')}")
            
            print(f"\n{'='*70}")
            print("[SUCCESS] Ejemplo completado usando arquitectura en capas")
            print(f"[INFO] Domain: 2 entidades creadas (DatasetSpec, EvalPlan)")  
            print(f"[INFO] Infrastructure: {len(successful_results)} métricas ejecutadas")
            print("="*70)
            
            return run_summary
            
        finally:
            # Limpiar archivos temporales
            try:
                os.unlink(real_path)
                os.unlink(synth_path)
            except:
                pass


def main():
    """Punto de entrada principal."""
    example = ValtaPyV2Example()
    return example.run()


if __name__ == "__main__":
    """Ejecutar el ejemplo cuando se ejecuta directamente."""
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Ejecución interrumpida por el usuario")
    except Exception as e:
        print(f"\n[ERROR] Error durante la ejecución: {e}")
        import traceback
        traceback.print_exc()