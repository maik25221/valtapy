import pandas as pd

from valtapy import Valtapy

# Crear datos de ejemplo
real_data = pd.read_csv(r"data\real\cardio_train.csv", sep=";")
synthetic_data = pd.read_csv(
    r"data\synt\Cardiovascular_Synthetic_Tabular_Data.csv", sep=";"
)

# Crear instancia de ValtaPy con directorio personalizado para resultados
validator = Valtapy(results_directory="my_validation_results")

# 1. Validar y guardar automáticamente en todos los formatos
results = validator.validate(
    real_data, synthetic_data, save_results=True, experiment_name="cardio_experiment_1"
)

# 2. Validar solo efficiency y guardar solo en JSON
results = validator.validate(
    real_data,
    synthetic_data,
    branches=["Efficiency"],
    save_results=True,
    experiment_name="efficiency_only",
    save_format="json",
)

# 3. Validar sin guardar, luego guardar manualmente
results = validator.validate(real_data, synthetic_data, branches=["Utility"])

# Guardar manualmente en formato específico
filepath = validator.save_results(results, "manual_save", "csv")
print(f"Results saved to: {filepath}")

# Guardar en todos los formatos
saved_files = validator.save_results(results, "complete_analysis", "all")
print("Files saved:")
for format_type, path in saved_files.items():
    print(f"  {format_type}: {path}")
