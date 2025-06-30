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
# results = validator.validate(
#     real_data,
#     synthetic_data,
#     branches=["Privacy"],
#     save_results=True,
#     experiment_name="privacy_comprehensive",
#     save_format="json",
# )

# 2. Validar solo calidad con todos los métodos integrados
# results = validator.validate(
#     real_data,
#     synthetic_data,
#     branches=["Quality"],
#     save_results=True,
#     experiment_name="quality_comprehensive",
#     save_format="json",
# )

# 3. Validar sin guardar, luego guardar manualmente
# results = validator.validate(real_data, synthetic_data, branches=["Utility"])

# Guardar manualmente en formato específico
# filepath = validator.save_results(results, "manual_save", "csv")
# print(f"Results saved to: {filepath}")


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

# ============================================================================
# EJEMPLOS ADICIONALES DE VALIDACIÓN DE CALIDAD
# ============================================================================

# Ejemplo 1: Solo métodos de detección de calidad
# from valtapy.validators.quality.detection import QualityDetectionOrchestrator
#
# detection_orchestrator = QualityDetectionOrchestrator(
#     real_data, synthetic_data, "quality_detection_results"
# )
# detection_results = detection_orchestrator.run_all_detections(contamination=0.05)
# print("Detection Quality Results:", detection_results)

# Ejemplo 2: Solo métodos estadísticos de calidad
# from valtapy.validators.quality.statistics import QualityStatisticsOrchestrator
#
# statistics_orchestrator = QualityStatisticsOrchestrator(
#     real_data, synthetic_data, "quality_statistics_results"
# )
# statistics_results = statistics_orchestrator.run_all_statistics()
# print("Statistics Quality Results:", statistics_results)

# Ejemplo 3: Validación de calidad con parámetros personalizados
# from valtapy.validators.quality import QualityValidator
#
# quality_validator = QualityValidator(
#     output_path="custom_quality_results",
#     seed=123
# )
# quality_results = quality_validator.validate(real_data, synthetic_data)
# print("Custom Quality Results:", quality_results)

# Ejemplo 4: Ejecutar métodos individuales de calidad
# from valtapy.validators.quality.detection.methods import IsolationForestDetection, LOFDetection
# from valtapy.validators.quality.statistics.methods import Tests, Correlation
#
# # Isolation Forest individual
# if_detector = IsolationForestDetection(real_data, synthetic_data, contamination=0.1)
# if_result = if_detector.execute()
# print("Isolation Forest Result:", if_result)
#
# # LOF individual
# lof_detector = LOFDetection(real_data, synthetic_data, n_neighbors=15)
# lof_result = lof_detector.execute()
# print("LOF Result:", lof_result)
#
# # Statistical Tests individual
# tests = Tests(real_data, synthetic_data, all_data=True)
# tests_result = tests.execute()
# print("Statistical Tests Result:", tests_result)
#
# # Correlation Analysis individual
# correlation = Correlation(real_data, synthetic_data)
# correlation_result = correlation.execute()
# print("Correlation Analysis Result:", correlation_result)

# Ejemplo 5: Validación completa (privacidad + calidad + utilidad)
results = validator.validate(
    real_data,
    synthetic_data,
    branches=["Privacy", "Quality", "Utility"],
    save_results=True,
    experiment_name="complete_validation",
    save_format="all"
)
print("Complete Validation Results:", results)

# Guardar en todos los formatos
saved_files = validator.save_results(results, "complete_analysis", "all")
print("Files saved:")
for format_type, path in saved_files.items():
    print(f"  {format_type}: {path}")

# Mostrar resultados de privacidad
print("\n" + "=" * 50)
print("RESULTADOS DE VALIDACIÓN DE PRIVACIDAD")
print("=" * 50)