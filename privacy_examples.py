"""
Ejemplo de uso directo de los métodos de privacidad de ValtaPy
Este archivo muestra cómo usar los diferentes métodos de privacidad individualmente
"""

import numpy as np
import pandas as pd

from valtapy.validators.privacy import PrivacyValidator
from valtapy.validators.privacy.extraction import PrivacyExtractionOrchestrator
from valtapy.validators.privacy.inference import PrivacyInferenceOrchestrator


def main():
    # Crear datos de ejemplo simples para demostración
    print("Creando datos de ejemplo...")
    np.random.seed(42)

    # Datos reales simulados
    real_data = pd.DataFrame(
        {
            "age": np.random.randint(18, 80, 1000),
            "income": np.random.normal(50000, 15000, 1000),
            "score": np.random.uniform(0, 100, 1000),
        }
    )

    # Datos sintéticos simulados (con algo de ruido añadido)
    synthetic_data = real_data.copy()
    synthetic_data["age"] += np.random.randint(-2, 3, 1000)
    synthetic_data["income"] *= np.random.uniform(0.95, 1.05, 1000)
    synthetic_data["score"] += np.random.normal(0, 5, 1000)

    print(f"Datos reales: {real_data.shape}")
    print(f"Datos sintéticos: {synthetic_data.shape}")
    print()

    # Ejemplo 1: Usar el validador completo de privacidad
    print("=" * 60)
    print("EJEMPLO 1: VALIDADOR COMPLETO DE PRIVACIDAD")
    print("=" * 60)

    privacy_validator = PrivacyValidator(
        output_path="privacy_example_results", epsilon=1.0, delta=1e-5
    )

    full_results = privacy_validator.validate(real_data, synthetic_data)

    print(
        f"Puntuación general de privacidad: {full_results['overall_privacy_score']:.4f}"
    )
    print(f"K-anonimato estimado: {full_results['k_anonymity']}")
    print(
        f"Puntuación de differential privacy: {full_results['differential_privacy']:.4f}"
    )
    print()

    # Ejemplo 2: Solo métodos de extracción
    print("=" * 60)
    print("EJEMPLO 2: SOLO MÉTODOS DE EXTRACCIÓN")
    print("=" * 60)

    extraction_orchestrator = PrivacyExtractionOrchestrator(
        real_data, synthetic_data, "extraction_example_results"
    )

    extraction_results = extraction_orchestrator.run_all_extractions(
        epsilon=0.5, delta=1e-6
    )

    print("Resultados de extracción:")
    if "dcr" in extraction_results and "dcr" in extraction_results["dcr"]:
        print(f"  DCR: {extraction_results['dcr']['dcr']:.4f}")

    if "overall_extraction_privacy_score" in extraction_results:
        print(
            f"  Puntuación general de extracción: {extraction_results['overall_extraction_privacy_score']:.4f}"
        )
    print()

    # Ejemplo 3: Solo métodos de inferencia
    print("=" * 60)
    print("EJEMPLO 3: SOLO MÉTODOS DE INFERENCIA")
    print("=" * 60)

    inference_orchestrator = PrivacyInferenceOrchestrator(
        real_data, synthetic_data, "inference_example_results"
    )

    inference_results = inference_orchestrator.run_all_inferences()

    print("Resultados de inferencia:")
    if "membership_inference_attack" in inference_results:
        mia_acc = inference_results["membership_inference_attack"].get(
            "membership_inference_accuracy", "N/A"
        )
        print(f"  Precisión del ataque de inferencia de membresía: {mia_acc:.4f}")
        print(f"  (Nota: menor precisión = mejor privacidad)")

    if "overall_inference_privacy_score" in inference_results:
        print(
            f"  Puntuación general de inferencia: {inference_results['overall_inference_privacy_score']:.4f}"
        )
    print()

    # Ejemplo 4: Métodos individuales
    print("=" * 60)
    print("EJEMPLO 4: MÉTODOS INDIVIDUALES")
    print("=" * 60)

    # DCR individual
    from valtapy.validators.privacy.extraction.methods import DCR

    dcr = DCR(real_data, synthetic_data)
    dcr_result = dcr.execute()
    print(f"DCR individual: {dcr_result['dcr']:.4f}")

    # Differential Privacy individual
    from valtapy.validators.privacy.extraction.methods import DifferentialPrivacy

    dp = DifferentialPrivacy(real_data, synthetic_data, epsilon=1.0, delta=1e-5)
    dp_result = dp.execute()
    print(
        f"Differential Privacy Score: {dp_result['differential_privacy']['privacy_score']:.4f}"
    )

    # Membership Inference Attack individual
    from valtapy.validators.privacy.inference.methods import MembershipInferenceAttack

    mia = MembershipInferenceAttack(real_data, synthetic_data)
    mia_result = mia.execute()
    print(f"MIA Accuracy: {mia_result['membership_inference_accuracy']:.4f}")

    print()
    print("=" * 60)
    print("TODOS LOS EJEMPLOS COMPLETADOS EXITOSAMENTE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
