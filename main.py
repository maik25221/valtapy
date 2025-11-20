"""
Main script to evaluate synthetic data using ValtaPy pipeline.

This script demonstrates the complete flow:
1. Ingestion: Read real and synthetic data from CSV files
2. Preprocessing: Clean and prepare data for evaluation
3. Evaluation: Compute metrics (KS Test) to assess data quality
"""

import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

import pandas as pd
import numpy as np
import json
from datetime import datetime

# Ingestion phase
from valtapy.ingestion.entities import DataSource
from valtapy.ingestion.readers.csv_reader import CSVReader

# Preprocessing phase
from valtapy.preprocessing.entities import PreprocessingConfig
from valtapy.preprocessing.processors.basic_preprocessor import BasicDataPreprocessor

# Evaluation phase
from valtapy.evaluation.entities import MetricExecutionContext, MetricFamily, MetricSpec, EvaluationConfig
from valtapy.evaluation.metrics.fidelity.ks_test import KSTestMetric


def print_separator(title: str = ""):
    """Print a formatted separator."""
    print("\n" + "=" * 80)
    if title:
        print(f"  {title}")
        print("=" * 80)
    print()


def main():
    """Run the complete ValtaPy evaluation pipeline."""

    print_separator("ValtaPy - Synthetic Data Evaluation Pipeline")

    # ============================================================================
    # 1. INGESTION PHASE - Load data from CSV files
    # ============================================================================
    print_separator("Phase 1: Data Ingestion")

    # Define data sources
    real_data_path = "data/real/a_hepatitis_sampling_baseline.csv"
    synth_data_path = "data/synt/f_hepatitis_CTGAN_syn_best.csv"

    print(f"Real data source:      {real_data_path}")
    print(f"Synthetic data source: {synth_data_path}")

    # Create data sources
    real_source = DataSource(path=real_data_path, format="csv")
    synth_source = DataSource(path=synth_data_path, format="csv")

    # Initialize CSV reader
    reader = CSVReader()

    # Read data
    print("\nReading real data...")
    real_result = reader.read(real_source)
    print(f"  [OK] Loaded {real_result.data.shape[0]} rows, {real_result.data.shape[1]} columns")
    print(f"  [OK] File size: {real_result.metadata['file_size_mb']} MB")
    print(f"  [OK] Memory usage: {real_result.metadata['memory_usage_mb']} MB")

    print("\nReading synthetic data...")
    synth_result = reader.read(synth_source)
    print(f"  [OK] Loaded {synth_result.data.shape[0]} rows, {synth_result.data.shape[1]} columns")
    print(f"  [OK] File size: {synth_result.metadata['file_size_mb']} MB")
    print(f"  [OK] Memory usage: {synth_result.metadata['memory_usage_mb']} MB")

    # Get DataFrames
    real_df = real_result.data
    synth_df = synth_result.data

    # Show column names
    print(f"\nColumns in real data: {list(real_df.columns)[:5]}... ({len(real_df.columns)} total)")
    print(f"Columns in synth data: {list(synth_df.columns)[:5]}... ({len(synth_df.columns)} total)")

    # ============================================================================
    # 2. PREPROCESSING PHASE - Clean and prepare data
    # ============================================================================
    print_separator("Phase 2: Data Preprocessing")

    # Create preprocessing configuration
    preproc_config = PreprocessingConfig(
        drop_empty_rows=True,
        drop_empty_columns=True,
        na_strategy="keep",  # Keep NaN values for now
        enforce_consistent_types=False,  # Not implemented yet
        normalize_numeric=False,  # Not implemented yet
        encode_categorical=False  # Not implemented yet
    )

    print("Preprocessing configuration:")
    print(f"  - Drop empty rows: {preproc_config.drop_empty_rows}")
    print(f"  - Drop empty columns: {preproc_config.drop_empty_columns}")
    print(f"  - NA strategy: {preproc_config.na_strategy}")

    # Initialize preprocessor
    preprocessor = BasicDataPreprocessor()

    # Check if preprocessor can handle the data
    if not preprocessor.can_handle(real_df, synth_df):
        print("ERROR: Preprocessor cannot handle the provided data")
        return

    # Preprocess data
    print("\nPreprocessing data...")
    preproc_result = preprocessor.preprocess(real_df, synth_df, preproc_config)

    print(f"  [OK] Processing time: {preproc_result.processing_time:.4f} seconds")
    print(f"  [OK] Operations applied: {preproc_result.report.operations_applied}")

    # Show preprocessing changes
    if preproc_result.report.real_changes:
        print(f"\n  Real data changes:")
        for key, value in preproc_result.report.real_changes.items():
            print(f"    - {key}: {value}")

    if preproc_result.report.synth_changes:
        print(f"\n  Synthetic data changes:")
        for key, value in preproc_result.report.synth_changes.items():
            print(f"    - {key}: {value}")

    # Show compatibility issues
    if preproc_result.report.compatibility_issues:
        print(f"\n  [WARNING] Compatibility issues found:")
        for issue in preproc_result.report.compatibility_issues:
            print(f"    - {issue}")
    else:
        print(f"\n  [OK] No compatibility issues found")

    # Get preprocessed DataFrames
    real_df_clean = preproc_result.real_data
    synth_df_clean = preproc_result.synth_data

    print(f"\nPreprocessed data shapes:")
    print(f"  Real:      {real_df_clean.shape}")
    print(f"  Synthetic: {synth_df_clean.shape}")

    # ============================================================================
    # 3. EVALUATION PHASE - Compute metrics
    # ============================================================================
    print_separator("Phase 3: Metric Evaluation")

    # Initialize KS Test metric
    ks_metric = KSTestMetric()

    print(f"Metric: {ks_metric.name}")
    print(f"  ID:          {ks_metric.metric_id}")
    print(f"  Family:      {ks_metric.family.value}")
    print(f"  Description: {ks_metric.description}")

    # Check if metric can be computed
    if not ks_metric.can_compute(real_df_clean, synth_df_clean):
        print("\nERROR: KS Test metric cannot be computed for this data")
        print("This might be because there are no numeric columns in common.")
        return

    print("\n  [OK] Metric can be computed")

    # Create execution context
    context = MetricExecutionContext(
        real_data=real_df_clean,
        synth_data=synth_df_clean,
        parameters={},
        cache_enabled=False,
        random_seed=42
    )

    # Compute metric
    print("\nComputing KS Test metric...")
    metric_result = ks_metric.compute(context)

    print(f"  [OK] Computation completed in {metric_result.computation_time:.4f} seconds")

    # ============================================================================
    # 4. RESULTS - Display and interpret results
    # ============================================================================
    print_separator("Evaluation Results")

    print(f"Overall Score (p-value): {metric_result.value:.6f}")
    print(f"Overall KS Statistic:    {metric_result.details['overall_ks_statistic']:.6f}")
    print(f"\nColumns tested:  {metric_result.details['num_columns_tested']}")
    print(f"Columns failed:  {metric_result.details['num_columns_failed']}")

    # Interpretation
    print("\nInterpretation:")
    if metric_result.value >= 0.05:
        print("  [GOOD] GOOD FIDELITY: Distributions are not significantly different")
        print("         The synthetic data preserves the statistical properties of the real data well.")
    elif metric_result.value >= 0.01:
        print("  [MODERATE] MODERATE FIDELITY: Distributions are somewhat different")
        print("             The synthetic data shows some differences from the real data.")
    else:
        print("  [POOR] POOR FIDELITY: Distributions are significantly different")
        print("         The synthetic data does not preserve the statistical properties well.")

    # Show per-column results
    print("\n" + "-" * 80)
    print("Per-Column Results:")
    print("-" * 80)
    print(f"{'Column Name':<30} {'KS Statistic':<15} {'P-Value':<12} {'Significant':<12}")
    print("-" * 80)

    for col_name, col_result in metric_result.details['column_results'].items():
        if "error" in col_result:
            print(f"{col_name:<30} ERROR: {col_result['error']}")
        else:
            ks_stat = col_result['ks_statistic']
            p_val = col_result['p_value']
            sig = "Yes" if col_result['significant'] else "No"
            print(f"{col_name:<30} {ks_stat:<15.6f} {p_val:<12.6f} {sig:<12}")

    # Show columns with significant differences
    significant_cols = [
        col for col, res in metric_result.details['column_results'].items()
        if "significant" in res and res["significant"]
    ]

    if significant_cols:
        print(f"\n[WARNING] Columns with significant differences (p < 0.05):")
        for col in significant_cols:
            res = metric_result.details['column_results'][col]
            print(f"  - {col}: {res['interpretation']}")
    else:
        print(f"\n[OK] No columns show significant differences")

    # ============================================================================
    # 5. SAVE RESULTS - Export to JSON
    # ============================================================================
    print_separator("Saving Results")

    # Helper function to convert numpy types to Python native types
    def convert_numpy_types(obj):
        """Convert numpy types to Python native types for JSON serialization."""
        if isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        return obj

    # Prepare results for JSON export
    results_dict = {
        "timestamp": datetime.now().isoformat(),
        "data_sources": {
            "real": real_data_path,
            "synthetic": synth_data_path
        },
        "data_shapes": {
            "real": {
                "original": list(real_df.shape),
                "preprocessed": list(real_df_clean.shape)
            },
            "synthetic": {
                "original": list(synth_df.shape),
                "preprocessed": list(synth_df_clean.shape)
            }
        },
        "preprocessing": {
            "config": {
                "drop_empty_rows": preproc_config.drop_empty_rows,
                "drop_empty_columns": preproc_config.drop_empty_columns,
                "na_strategy": preproc_config.na_strategy
            },
            "processing_time": preproc_result.processing_time,
            "operations_applied": preproc_result.report.operations_applied,
            "compatibility_issues": preproc_result.report.compatibility_issues
        },
        "evaluation": {
            "metric_id": metric_result.metric_id,
            "metric_family": metric_result.family.value,
            "overall_score": metric_result.value,
            "overall_ks_statistic": metric_result.details['overall_ks_statistic'],
            "overall_p_value": metric_result.details['overall_p_value'],
            "computation_time": metric_result.computation_time,
            "columns_tested": metric_result.details['num_columns_tested'],
            "columns_failed": metric_result.details['num_columns_failed'],
            "column_results": convert_numpy_types(metric_result.details['column_results']),
            "metadata": metric_result.metadata
        }
    }

    # Save to JSON
    output_file = "evaluation_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)

    print(f"Results saved to: {output_file}")

    print_separator("Pipeline Completed Successfully")
    print(f"Total columns evaluated: {metric_result.details['num_columns_tested']}")
    print(f"Overall fidelity score: {metric_result.value:.6f}")
    print(f"\nFor detailed results, see: {output_file}")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExecution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
