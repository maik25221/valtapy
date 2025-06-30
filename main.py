import pandas as pd

from valtapy import Valtapy

# Crear datos de ejemplo
real_data = pd.read_csv(r"data\real\cardio_train.csv", sep=",")
synthetic_data = pd.read_csv(
    r"data\synt\Cardiovascular_Synthetic_Tabular_Data.csv", sep=","
)

# Crear instancia de ValtaPy con directorio personalizado para resultados
validator = Valtapy(results_directory="my_validation_results")

# 1. Validar y guardar automáticamente en todos los formatos
# results = validator.validate(
#     real_data, synthetic_data, save_results=True, experiment_name="cardio_experiment_1"
# )

# 2. Validar solo efficiency y guardar solo en JSON
# results = validator.validate(
#     real_data,
#     synthetic_data,
#     branches=["Efficiency"],
#     save_results=True,
#     experiment_name="efficiency_only",
#     save_format="json",
# )

# 2. Validar solo privacidad con todos los métodos integrados
results = validator.validate(
    real_data,
    synthetic_data,
    branches=["Privacy"],
    save_results=True,
    experiment_name="privacy_comprehensive",
    save_format="json",
)

# 3. Validar sin guardar, luego guardar manualmente
# results = validator.validate(real_data, synthetic_data, branches=["Utility"])

# Guardar manualmente en formato específico
# filepath = validator.save_results(results, "manual_save", "csv")
# print(f"Results saved to: {filepath}")

# Guardar en todos los formatos
saved_files = validator.save_results(results, "complete_analysis", "all")
print("Files saved:")
for format_type, path in saved_files.items():
    print(f"  {format_type}: {path}")

# Mostrar resultados de privacidad
print("\n" + "=" * 50)
print("RESULTADOS DE VALIDACIÓN DE PRIVACIDAD")
print("=" * 50)

if "Privacy" in results:
    privacy_results = results["Privacy"]

    print(
        f"\nPuntuación general de privacidad: {privacy_results.get('overall_privacy_score', 'N/A'):.4f}"
    )
    print(f"K-anonimato estimado: {privacy_results.get('k_anonymity', 'N/A')}")

    # Mostrar resultados de métodos de extracción
    if "extraction_methods" in privacy_results:
        print("\n--- MÉTODOS DE EXTRACCIÓN ---")
        extraction = privacy_results["extraction_methods"]

        if "dcr" in extraction and "dcr" in extraction["dcr"]:
            print(f"DCR (Disclosure Control Ratio): {extraction['dcr']['dcr']:.4f}")

        if (
            "differential_privacy" in extraction
            and "differential_privacy" in extraction["differential_privacy"]
        ):
            dp = extraction["differential_privacy"]["differential_privacy"]
            print(f"Differential Privacy Score: {dp.get('privacy_score', 'N/A'):.4f}")
            print(f"Epsilon: {dp.get('epsilon', 'N/A')}")
            print(f"Delta: {dp.get('delta', 'N/A')}")

        if "identity_attribute_disclosure" in extraction:
            iad = extraction["identity_attribute_disclosure"]
            print(
                f"Identity Disclosure Rate: {iad.get('identity_disclosure_rate', 'N/A'):.4f}"
            )
            print(
                f"Attribute Disclosure Rate: {iad.get('attribute_disclosure_rate', 'N/A'):.4f}"
            )

    # Mostrar resultados de métodos de inferencia
    if "inference_methods" in privacy_results:
        print("\n--- MÉTODOS DE INFERENCIA ---")
        inference = privacy_results["inference_methods"]

        if "membership_inference_attack" in inference:
            mia = inference["membership_inference_attack"]
            print(
                f"Membership Inference Attack Accuracy: {mia.get('membership_inference_accuracy', 'N/A'):.4f}"
            )

        if "identity_attribute_disclosure_inference" in inference:
            iad_inf = inference["identity_attribute_disclosure_inference"]
            print(
                f"Identity Disclosure Rate (Inference): {iad_inf.get('identity_disclosure_rate', 'N/A'):.4f}"
            )
            print(
                f"Attribute Disclosure Rate (Inference): {iad_inf.get('attribute_disclosure_rate', 'N/A'):.4f}"
            )

print("\n" + "=" * 50)

# ============================================================================
# EJEMPLOS ADICIONALES DE VALIDACIÓN DE PRIVACIDAD
# ============================================================================

# Ejemplo 1: Validación completa (todas las ramas incluyendo privacidad)
# results_complete = validator.validate(
#     real_data,
#     synthetic_data,
#     save_results=True,
#     experiment_name="complete_with_privacy",
#     save_format="all"
# )

# Ejemplo 2: Solo métodos de extracción de privacidad
# from valtapy.validators.privacy.extraction import PrivacyExtractionOrchestrator
#
# extraction_orchestrator = PrivacyExtractionOrchestrator(
#     real_data, synthetic_data, "privacy_extraction_results"
# )
# extraction_results = extraction_orchestrator.run_all_extractions(epsilon=0.5, delta=1e-6)
# print("Extraction Privacy Results:", extraction_results)

# Ejemplo 3: Solo métodos de inferencia de privacidad
# from valtapy.validators.privacy.inference import PrivacyInferenceOrchestrator
#
# inference_orchestrator = PrivacyInferenceOrchestrator(
#     real_data, synthetic_data, "privacy_inference_results"
# )
# inference_results = inference_orchestrator.run_all_inferences()
# print("Inference Privacy Results:", inference_results)

# Ejemplo 4: Validación de privacidad con parámetros personalizados
# from valtapy.validators.privacy import PrivacyValidator
#
# privacy_validator = PrivacyValidator(
#     output_path="custom_privacy_results",
#     epsilon=0.1,  # Más estricto (mejor privacidad)
#     delta=1e-7    # Más estricto (mejor privacidad)
# )
# privacy_results = privacy_validator.validate(real_data, synthetic_data)
# print("Custom Privacy Results:", privacy_results)

# Ejemplo 5: Ejecutar métodos individuales de privacidad
# from valtapy.validators.privacy.extraction.methods import DCR, DifferentialPrivacy
# from valtapy.validators.privacy.inference.methods import MembershipInferenceAttack
#
# # DCR individual
# dcr = DCR(real_data, synthetic_data)
# dcr_result = dcr.execute()
# print("DCR Result:", dcr_result)
#
# # Differential Privacy individual
# dp = DifferentialPrivacy(real_data, synthetic_data, epsilon=1.0, delta=1e-5)
# dp_result = dp.execute()
# print("Differential Privacy Result:", dp_result)
#
# # Membership Inference Attack individual
# mia = MembershipInferenceAttack(real_data, synthetic_data)
# mia_result = mia.execute()
# print("Membership Inference Attack Result:", mia_result)
