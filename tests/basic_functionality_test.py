"""Test básico de funcionalidad completa: cargar CSV y ejecutar métricas."""

import sys
import tempfile
import os
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from valtapyV2.infrastructure.io.loaders import load_csv, save_csv
from valtapyV2.infrastructure.metrics.registry import get_metric_registry
from valtapyV2.domain.entities import DatasetSpec
from utils.test_helpers import create_mock_context


def test_basic_workflow():
    """Test completo del workflow básico: cargar datos reales/sintéticos y ejecutar métricas."""
    
    print("Iniciando test basico de funcionalidad...")
    
    # 1. Crear datos de prueba reales
    real_data_content = """feature1,feature2,feature3,target
1.0,10.0,0.5,high
2.0,20.0,1.0,medium
3.0,30.0,1.5,low
4.0,40.0,2.0,high
5.0,50.0,2.5,medium
6.0,60.0,3.0,low
7.0,70.0,3.5,high
8.0,80.0,4.0,medium
9.0,90.0,4.5,low
10.0,100.0,5.0,high
"""
    
    # 2. Crear datos sintéticos (similares pero con ruido)
    synth_data_content = """feature1,feature2,feature3,target
1.1,10.2,0.6,high
2.2,19.8,0.9,medium
3.1,29.7,1.4,low
4.2,40.3,2.1,high
5.1,49.9,2.6,medium
6.2,60.1,2.9,low
7.1,69.8,3.4,high
8.2,80.2,4.1,medium
9.1,89.9,4.6,low
10.2,100.1,5.1,high
"""
    
    # 3. Guardar en archivos temporales
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as real_f:
        real_f.write(real_data_content)
        real_path = real_f.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as synth_f:
        synth_f.write(synth_data_content)
        synth_path = synth_f.name
    
    try:
        print(f"[FILES] Datos reales guardados en: {real_path}")
        print(f"[FILES] Datos sintéticos guardados en: {synth_path}")
        
        # 4. Cargar los datasets
        print("\n[DATA] Cargando datasets...")
        real_df = load_csv(real_path)
        synth_df = load_csv(synth_path)
        
        print(f"[OK] Datos reales cargados: {len(real_df)} filas, {len(real_df.columns)} columnas")
        print(f"[OK] Datos sintéticos cargados: {len(synth_df)} filas, {len(synth_df.columns)} columnas")
        print(f"[INFO] Columnas: {list(real_df.columns)}")
        
        # 5. Configurar contexto
        dataset_spec = DatasetSpec(
            target="target",
            dtypes={
                "feature1": "float64",
                "feature2": "float64", 
                "feature3": "float64",
                "target": "object"
            }
        )
        context = create_mock_context()
        context["dataset_spec"] = dataset_spec
        
        print(f"\n[CONFIG] Contexto configurado con target: {dataset_spec.target}")
        
        # 6. Obtener el registry de métricas
        print("\n[TOOLS] Obteniendo registro de métricas...")
        registry = get_metric_registry()
        available_metrics = registry.list_ids()
        
        print(f"[METRICS] Métricas disponibles: {len(available_metrics)}")
        for family, metrics in registry.list_by_family().items():
            print(f"  - {family}: {len(metrics)} métricas ({', '.join(metrics)})")
        
        # 7. Ejecutar algunas métricas básicas
        print("\n[TEST] Ejecutando métricas...")
        
        test_metrics = [
            "fidelity.ks",
            "utility.accuracy", 
            "privacy.nndr"
        ]
        
        results = {}
        
        for metric_id in test_metrics:
            try:
                print(f"\n  [EXEC] Ejecutando {metric_id}...")
                
                # Obtener clase de métrica
                metric_class = registry.get(metric_id)
                metric = metric_class()
                
                print(f"    - Familia: {metric.family}")
                print(f"    - Propósito: {metric.purpose_tags}")
                
                # Ejecutar métrica
                metric.fit(real_df, synth_df, context)
                result = metric.compute()
                
                results[metric_id] = result
                
                print(f"    [OK] Resultado: {result.value:.4f}")
                if "error" in result.details:
                    print(f"    [WARNING] Error: {result.details['error']}")
                else:
                    print(f"    [DATA] Detalles: {len(result.details)} campos")
                
            except Exception as e:
                print(f"    [ERROR] Error ejecutando {metric_id}: {e}")
                results[metric_id] = None
        
        # 8. Resumen de resultados
        print(f"\n[INFO] RESUMEN DE RESULTADOS:")
        print(f"{'='*50}")
        
        successful_metrics = [m for m, r in results.items() if r is not None]
        failed_metrics = [m for m, r in results.items() if r is None]
        
        print(f"[OK] Métricas exitosas: {len(successful_metrics)}")
        for metric_id in successful_metrics:
            result = results[metric_id]
            print(f"  - {metric_id}: {result.value:.4f} ({result.family})")
        
        if failed_metrics:
            print(f"\n[ERROR] Métricas fallidas: {len(failed_metrics)}")
            for metric_id in failed_metrics:
                print(f"  - {metric_id}")
        
        # 9. Verificación básica
        print(f"\n[EXEC] VERIFICACIÓN:")
        assert len(real_df) > 0, "Datos reales no cargados"
        assert len(synth_df) > 0, "Datos sintéticos no cargados" 
        assert len(real_df.columns) == len(synth_df.columns), "Columnas no coinciden"
        assert len(successful_metrics) > 0, "Ninguna métrica ejecutada exitosamente"
        
        print("[OK] Todas las verificaciones pasaron")
        print(f"\n[SUCCESS] Test básico COMPLETADO exitosamente!")
        print(f"   - Carga de CSV: [OK]")
        print(f"   - Registro de métricas: [OK]") 
        print(f"   - Ejecución de métricas: [OK] ({len(successful_metrics)}/{len(test_metrics)})")
        
        return True
        
    finally:
        # Limpiar archivos temporales
        try:
            os.unlink(real_path)
            os.unlink(synth_path)
        except:
            pass


if __name__ == "__main__":
    """Ejecutar el test básico directamente."""
    try:
        success = test_basic_workflow()
        if success:
            print("\n[PASS] TEST BÁSICO EXITOSO")
        else:
            print("\n[FAIL] TEST BÁSICO FALLIDO")
    except Exception as e:
        print(f"\n[FAIL] ERROR EN TEST BÁSICO: {e}")
        import traceback
        traceback.print_exc()