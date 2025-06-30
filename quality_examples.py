"""
Ejemplo de uso directo de los métodos de calidad de ValtaPy
Este archivo muestra cómo usar los diferentes métodos de calidad individualmente
"""

import numpy as np
import pandas as pd

from valtapy.validators.quality import QualityValidator
from valtapy.validators.quality.detection import QualityDetectionOrchestrator
from valtapy.validators.quality.statistics import QualityStatisticsOrchestrator


def main():
    # Crear datos de ejemplo simples para demostración
    print("Creando datos de ejemplo...")
    np.random.seed(42)

    # Datos reales simulados
    real_data = pd.DataFrame(
        {
            "feature1": np.random.normal(0, 1, 1000),
            "feature2": np.random.exponential(2, 1000),
            "feature3": np.random.uniform(-1, 1, 1000),
            "feature4": np.random.gamma(2, 2, 1000),
        }
    )

    # Datos sintéticos simulados (con algunas diferencias introducidas)
    synthetic_data = pd.DataFrame(
        {
            "feature1": np.random.normal(
                0.1, 1.1, 1000
            ),  # Ligero cambio en media y varianza
            "feature2": np.random.exponential(2.1, 1000),  # Ligero cambio en parámetro
            "feature3": np.random.uniform(-0.9, 1.1, 1000),  # Ligero cambio en rango
            "feature4": np.random.gamma(2.2, 1.9, 1000),  # Ligero cambio en parámetros
        }
    )

    print(f"Datos reales: {real_data.shape}")
    print(f"Datos sintéticos: {synthetic_data.shape}")
    print()

    # Ejemplo 1: Usar el validador completo de calidad
    print("=" * 60)
    print("EJEMPLO 1: VALIDADOR COMPLETO DE CALIDAD")
    print("=" * 60)

    quality_validator = QualityValidator(output_path="quality_example_results", seed=42)

    full_results = quality_validator.validate(real_data, synthetic_data)

    print(f"Puntuación general de calidad: {full_results['overall_quality_score']:.4f}")
    print(f"Completitud: {full_results['completeness']:.4f}")
    print(f"Consistencia: {full_results['consistency']:.4f}")
    print()

    # Ejemplo 2: Solo métodos de detección
    print("=" * 60)
    print("EJEMPLO 2: SOLO MÉTODOS DE DETECCIÓN")
    print("=" * 60)

    detection_orchestrator = QualityDetectionOrchestrator(
        real_data, synthetic_data, "detection_example_results", seed=42
    )

    detection_results = detection_orchestrator.run_all_detections(contamination=0.05)

    print("Resultados de detección:")
    if "isolation_forest" in detection_results:
        if_results = detection_results["isolation_forest"]
        print(
            f"  Isolation Forest - Quality Score: {if_results.get('quality_score', 'N/A'):.4f}"
        )
        print(
            f"  Isolation Forest - Anomaly Rate: {if_results.get('anomaly_rate', 'N/A'):.4f}"
        )

    if "overall_detection_quality_score" in detection_results:
        print(
            f"  Puntuación general de detección: {detection_results['overall_detection_quality_score']:.4f}"
        )
    print()

    # Ejemplo 3: Solo métodos estadísticos
    print("=" * 60)
    print("EJEMPLO 3: SOLO MÉTODOS ESTADÍSTICOS")
    print("=" * 60)

    statistics_orchestrator = QualityStatisticsOrchestrator(
        real_data, synthetic_data, "statistics_example_results", seed=42
    )

    statistics_results = statistics_orchestrator.run_all_statistics()

    print("Resultados estadísticos:")
    if "statistical_tests" in statistics_results:
        tests_results = statistics_results["statistical_tests"]
        print(f"  KS Test p-value: {tests_results.get('ks_p_value', 'N/A'):.6f}")
        print(f"  KL Divergence: {tests_results.get('kl_divergence', 'N/A'):.6f}")

    if "correlation_analysis" in statistics_results:
        corr_results = statistics_results["correlation_analysis"]
        print(
            f"  Correlation Similarity: {corr_results.get('correlation_similarity', 'N/A'):.4f}"
        )

    if "overall_statistics_quality_score" in statistics_results:
        print(
            f"  Puntuación general estadística: {statistics_results['overall_statistics_quality_score']:.4f}"
        )
    print()

    # Ejemplo 4: Métodos individuales
    print("=" * 60)
    print("EJEMPLO 4: MÉTODOS INDIVIDUALES")
    print("=" * 60)

    # Isolation Forest individual
    from valtapy.validators.quality.detection.methods import IsolationForestDetection

    if_detector = IsolationForestDetection(
        real_data, synthetic_data, seed=42, contamination=0.1
    )
    if_result = if_detector.execute()
    print(
        f"Isolation Forest individual - Quality Score: {if_result['quality_score']:.4f}"
    )

    # LOF individual
    from valtapy.validators.quality.detection.methods import LOFDetection

    lof_detector = LOFDetection(real_data, synthetic_data, seed=42, n_neighbors=20)
    lof_result = lof_detector.execute()
    print(f"LOF individual - Quality Score: {lof_result['quality_score']:.4f}")

    # Tests estadísticos individuales
    from valtapy.validators.quality.statistics.methods import Tests

    tests = Tests(real_data, synthetic_data, seed=42, all_data=True)
    tests_result = tests.execute()
    print(f"Tests estadísticos - KS p-value: {tests_result['ks_p_value']:.6f}")
    print(f"Tests estadísticos - KL Divergence: {tests_result['kl_divergence']:.6f}")

    # Análisis de correlación individual
    from valtapy.validators.quality.statistics.methods import Correlation

    correlation = Correlation(real_data, synthetic_data, seed=42)
    correlation_result = correlation.execute()
    print(
        f"Correlación - Similarity Score: {correlation_result['correlation_similarity']:.4f}"
    )

    print()
    print("=" * 60)
    print("TODOS LOS EJEMPLOS DE CALIDAD COMPLETADOS EXITOSAMENTE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
