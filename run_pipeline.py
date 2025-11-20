"""
ValtaPy Pipeline Runner - Configuration-based execution.

This script runs the ValtaPy evaluation pipeline based on a JSON configuration file.
It supports flexible phase execution allowing you to:
- Skip preprocessing if data is already clean
- Skip validation if you trust your data
- Run only specific metrics
- Configure all aspects via JSON

Usage:
    python run_pipeline.py config.json
    python run_pipeline.py config_evaluation_only.json
"""

import sys
import json
from pathlib import Path
from typing import Any, Optional
from datetime import datetime

# Add src to path for imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

import pandas as pd
import numpy as np

# Ingestion phase
from valtapy.ingestion.entities import DataSource
from valtapy.ingestion.readers.csv_reader import CSVReader

# Preprocessing phase
from valtapy.preprocessing.entities import PreprocessingConfig
from valtapy.preprocessing.processors.basic_preprocessor import BasicDataPreprocessor

# Evaluation phase
from valtapy.evaluation.entities import MetricExecutionContext
from valtapy.evaluation.metrics.fidelity.ks_test import KSTestMetric
from valtapy.evaluation.metrics.privacy.membership_inference import MembershipInferenceMetric
from valtapy.evaluation.metrics.utility import (
    TTRMetric,
    TTSMetric,
    TRTSMetric,
    TSTRMetric,
    TTRSMetric
)


class PipelineRunner:
    """Orchestrates the ValtaPy pipeline based on configuration."""

    def __init__(self, config_path: str):
        """Initialize runner with configuration file."""
        self.config_path = config_path
        self.config = self._load_config()
        self.results = {}

    def _load_config(self) -> dict:
        """Load and validate configuration file."""
        config_file = Path(self.config_path)

        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # Validate required sections
        required_sections = ['pipeline', 'data_sources']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required section in config: {section}")

        return config

    def _print_separator(self, title: str = "", char: str = "="):
        """Print a formatted separator."""
        print("\n" + char * 80)
        if title:
            print(f"  {title}")
            print(char * 80)
        print()

    def _convert_numpy_types(self, obj):
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
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        return obj

    def run(self):
        """Execute the pipeline according to configuration."""
        self._print_separator(f"ValtaPy Pipeline: {self.config['pipeline']['name']}")

        print(f"Description: {self.config['pipeline'].get('description', 'N/A')}")
        print(f"Enabled phases: {', '.join(self.config['pipeline']['enabled_phases'])}")
        print(f"Configuration file: {self.config_path}")

        # Phase 1: Ingestion (always required)
        real_df, synth_df = self._run_ingestion()

        # Phase 2: Validation (optional)
        if "validation" in self.config['pipeline']['enabled_phases']:
            real_df, synth_df = self._run_validation(real_df, synth_df)

        # Phase 3: Preprocessing (optional)
        if "preprocessing" in self.config['pipeline']['enabled_phases']:
            real_df, synth_df = self._run_preprocessing(real_df, synth_df)
        else:
            print("\n[SKIP] Preprocessing phase disabled in configuration")
            self.results['preprocessing'] = {
                'enabled': False,
                'skipped': True
            }

        # Phase 4: Evaluation (optional but typically required)
        if "evaluation" in self.config['pipeline']['enabled_phases']:
            self._run_evaluation(real_df, synth_df)
        else:
            print("\n[SKIP] Evaluation phase disabled in configuration")
            self.results['evaluation'] = {
                'enabled': False,
                'skipped': True
            }

        # Save results
        self._save_results()

        self._print_separator("Pipeline Completed Successfully")

    def _run_ingestion(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Run data ingestion phase."""
        self._print_separator("Phase 1: Data Ingestion")

        real_config = self.config['data_sources']['real']
        synth_config = self.config['data_sources']['synthetic']

        print(f"Real data source:      {real_config['path']}")
        print(f"Synthetic data source: {synth_config['path']}")

        # Create data sources
        real_source = DataSource(
            path=real_config['path'],
            format=real_config['format'],
            read_params=real_config.get('read_params', {})
        )

        synth_source = DataSource(
            path=synth_config['path'],
            format=synth_config['format'],
            read_params=synth_config.get('read_params', {})
        )

        # Initialize reader
        reader = CSVReader()

        # Read data
        print("\nReading real data...")
        real_result = reader.read(real_source)
        print(f"  [OK] Loaded {real_result.data.shape[0]} rows, {real_result.data.shape[1]} columns")
        print(f"  [OK] File size: {real_result.metadata['file_size_mb']} MB")

        print("\nReading synthetic data...")
        synth_result = reader.read(synth_source)
        print(f"  [OK] Loaded {synth_result.data.shape[0]} rows, {synth_result.data.shape[1]} columns")
        print(f"  [OK] File size: {synth_result.metadata['file_size_mb']} MB")

        # Store results
        self.results['ingestion'] = {
            'real': {
                'path': real_config['path'],
                'shape': list(real_result.data.shape),
                'file_size_mb': real_result.metadata['file_size_mb'],
                'memory_usage_mb': real_result.metadata['memory_usage_mb'],
                'columns': real_result.metadata['columns']
            },
            'synthetic': {
                'path': synth_config['path'],
                'shape': list(synth_result.data.shape),
                'file_size_mb': synth_result.metadata['file_size_mb'],
                'memory_usage_mb': synth_result.metadata['memory_usage_mb'],
                'columns': synth_result.metadata['columns']
            }
        }

        return real_result.data, synth_result.data

    def _run_validation(self, real_df: pd.DataFrame, synth_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Run validation phase (placeholder for future implementation)."""
        self._print_separator("Phase 2: Validation")

        validation_config = self.config.get('validation', {})

        if not validation_config.get('enabled', True):
            print("[SKIP] Validation disabled in configuration")
            self.results['validation'] = {'enabled': False, 'skipped': True}
            return real_df, synth_df

        print("[INFO] Validation phase not yet implemented")
        print("       Skipping validation checks")

        self.results['validation'] = {
            'enabled': True,
            'implemented': False,
            'message': 'Validation phase placeholder - to be implemented'
        }

        return real_df, synth_df

    def _run_preprocessing(self, real_df: pd.DataFrame, synth_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Run preprocessing phase."""
        self._print_separator("Phase 3: Data Preprocessing")

        preproc_config_dict = self.config.get('preprocessing', {})

        if not preproc_config_dict.get('enabled', True):
            print("[SKIP] Preprocessing disabled in configuration")
            self.results['preprocessing'] = {'enabled': False, 'skipped': True}
            return real_df, synth_df

        # Create preprocessing configuration
        config_params = preproc_config_dict.get('config', {})
        preproc_config = PreprocessingConfig(
            drop_empty_rows=config_params.get('drop_empty_rows', True),
            drop_empty_columns=config_params.get('drop_empty_columns', True),
            na_strategy=config_params.get('na_strategy', 'keep'),
            enforce_consistent_types=config_params.get('enforce_consistent_types', False),
            normalize_numeric=config_params.get('normalize_numeric', False),
            encode_categorical=config_params.get('encode_categorical', False)
        )

        print("Preprocessing configuration:")
        print(f"  - Drop empty rows: {preproc_config.drop_empty_rows}")
        print(f"  - Drop empty columns: {preproc_config.drop_empty_columns}")
        print(f"  - NA strategy: {preproc_config.na_strategy}")

        # Initialize preprocessor
        preprocessor = BasicDataPreprocessor()

        if not preprocessor.can_handle(real_df, synth_df):
            raise ValueError("Preprocessor cannot handle the provided data")

        # Preprocess data
        print("\nPreprocessing data...")
        preproc_result = preprocessor.preprocess(real_df, synth_df, preproc_config)

        print(f"  [OK] Processing time: {preproc_result.processing_time:.4f} seconds")
        print(f"  [OK] Operations applied: {preproc_result.report.operations_applied}")

        # Show changes
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

        print(f"\nPreprocessed shapes: Real={preproc_result.real_data.shape}, Synth={preproc_result.synth_data.shape}")

        # Store results
        self.results['preprocessing'] = {
            'enabled': True,
            'config': {
                'drop_empty_rows': preproc_config.drop_empty_rows,
                'drop_empty_columns': preproc_config.drop_empty_columns,
                'na_strategy': preproc_config.na_strategy
            },
            'processing_time': preproc_result.processing_time,
            'operations_applied': preproc_result.report.operations_applied,
            'real_changes': preproc_result.report.real_changes,
            'synth_changes': preproc_result.report.synth_changes,
            'compatibility_issues': preproc_result.report.compatibility_issues,
            'output_shapes': {
                'real': list(preproc_result.real_data.shape),
                'synthetic': list(preproc_result.synth_data.shape)
            }
        }

        return preproc_result.real_data, preproc_result.synth_data

    def _run_evaluation(self, real_df: pd.DataFrame, synth_df: pd.DataFrame):
        """Run evaluation phase."""
        self._print_separator("Phase 4: Metric Evaluation")

        eval_config = self.config.get('evaluation', {})

        if not eval_config.get('enabled', True):
            print("[SKIP] Evaluation disabled in configuration")
            self.results['evaluation'] = {'enabled': False, 'skipped': True}
            return

        metrics_config = eval_config.get('metrics', [])
        random_seed = eval_config.get('random_seed', 42)
        cache_enabled = eval_config.get('cache_enabled', False)

        print(f"Configured metrics: {len(metrics_config)}")
        print(f"Random seed: {random_seed}")
        print(f"Cache enabled: {cache_enabled}")

        # Available metrics registry
        available_metrics = {
            'fidelity.ks': KSTestMetric(),
            'privacy.mia': MembershipInferenceMetric(),
            'ml_efficiency_ttr': TTRMetric(),
            'ml_efficiency_tts': TTSMetric(),
            'ml_efficiency_trts': TRTSMetric(),
            'ml_efficiency_tstr': TSTRMetric(),
            'ml_efficiency_ttrs': TTRSMetric()
        }

        metric_results = []

        for metric_config in metrics_config:
            metric_id = metric_config['metric_id']

            if not metric_config.get('enabled', True):
                print(f"\n[SKIP] Metric '{metric_id}' disabled in configuration")
                continue

            if metric_id not in available_metrics:
                print(f"\n[ERROR] Metric '{metric_id}' not found in registry")
                print(f"        Available metrics: {list(available_metrics.keys())}")
                continue

            # Get metric instance
            metric = available_metrics[metric_id]

            print(f"\n{'-' * 80}")
            print(f"Computing metric: {metric.name} ({metric_id})")
            print(f"{'-' * 80}")
            print(f"  Family: {metric.family.value}")
            print(f"  Description: {metric.description}")

            # Check if metric can be computed
            if not metric.can_compute(real_df, synth_df):
                print(f"  [ERROR] Metric cannot be computed for this data")
                continue

            print(f"  [OK] Metric can be computed")

            # Create execution context
            context = MetricExecutionContext(
                real_data=real_df,
                synth_data=synth_df,
                parameters=metric_config.get('parameters', {}),
                cache_enabled=cache_enabled,
                random_seed=random_seed
            )

            # Compute metric
            print(f"\n  Computing...")
            result = metric.compute(context)

            print(f"  [OK] Computation completed in {result.computation_time:.4f} seconds")
            print(f"  Overall Score: {result.value:.6f}")

            # Interpretation for KS Test
            if metric_id == 'fidelity.ks':
                if result.value >= 0.05:
                    interpretation = "GOOD FIDELITY"
                elif result.value >= 0.01:
                    interpretation = "MODERATE FIDELITY"
                else:
                    interpretation = "POOR FIDELITY"
                print(f"  Interpretation: {interpretation}")

            # Interpretation for MIA
            elif metric_id == 'privacy.mia':
                if result.value >= 0.4:
                    interpretation = "GOOD PRIVACY"
                elif result.value >= 0.2:
                    interpretation = "MODERATE PRIVACY"
                else:
                    interpretation = "POOR PRIVACY (HIGH RISK)"
                print(f"  Interpretation: {interpretation}")
                print(f"  Attack Accuracy: {result.details['attack_accuracy']:.4f}")

            # Interpretation for ML Efficiency metrics
            elif metric_id.startswith('ml_efficiency_'):
                task_type = result.details.get('task_type', 'unknown')
                primary_metric = result.details.get('primary_metric', 'unknown')

                print(f"  Task Type: {task_type}")
                print(f"  Primary Metric: {primary_metric}")
                print(f"  Primary Value: {result.value:.4f}")

                if task_type == 'classification':
                    print(f"\n  Classification Metrics:")
                    print(f"    Accuracy:  {result.details.get('accuracy', 0):.4f}")
                    print(f"    F1-Score:  {result.details.get('f1_score', 0):.4f}")
                    print(f"    Precision: {result.details.get('precision', 0):.4f}")
                    print(f"    Recall:    {result.details.get('recall', 0):.4f}")
                    if result.details.get('auc_roc') is not None:
                        print(f"    AUC-ROC:   {result.details.get('auc_roc'):.4f}")
                elif task_type == 'regression':
                    print(f"\n  Regression Metrics:")
                    print(f"    RMSE:     {result.details.get('rmse', 0):.4f}")
                    print(f"    MAE:      {result.details.get('mae', 0):.4f}")
                    print(f"    R²-Score: {result.details.get('r2_score', 0):.4f}")

                print(f"\n  Data Configuration:")
                data_src = result.details.get('data_source', {})
                print(f"    Train: {data_src.get('train', 'unknown')}")
                print(f"    Test:  {data_src.get('test', 'unknown')}")
                print(f"    Train samples: {result.details.get('train_samples', 0)}")
                print(f"    Test samples:  {result.details.get('test_samples', 0)}")

            metric_results.append({
                'metric_id': result.metric_id,
                'metric_family': result.family.value,
                'metric_name': metric.name,
                'value': result.value,
                'computation_time': result.computation_time,
                'details': self._convert_numpy_types(result.details),
                'metadata': result.metadata
            })

        # Store evaluation results
        self.results['evaluation'] = {
            'enabled': True,
            'random_seed': random_seed,
            'cache_enabled': cache_enabled,
            'metrics_configured': len(metrics_config),
            'metrics_computed': len(metric_results),
            'results': metric_results
        }

        # Print summary
        print(f"\n{'-' * 80}")
        print(f"Evaluation Summary:")
        print(f"{'-' * 80}")
        print(f"  Metrics configured: {len(metrics_config)}")
        print(f"  Metrics computed:   {len(metric_results)}")

        for result in metric_results:
            print(f"\n  {result['metric_name']} ({result['metric_id']}):")
            print(f"    Score: {result['value']:.6f}")
            print(f"    Time:  {result['computation_time']:.4f}s")

    def _save_results(self):
        """Save pipeline results to JSON file."""
        self._print_separator("Saving Results")

        output_file = self.config['pipeline'].get('output_file', 'pipeline_results.json')

        # Prepare final results
        final_results = {
            'timestamp': datetime.now().isoformat(),
            'config_file': self.config_path,
            'pipeline': {
                'name': self.config['pipeline']['name'],
                'description': self.config['pipeline'].get('description', ''),
                'enabled_phases': self.config['pipeline']['enabled_phases']
            },
            'results': self.results
        }

        # Save to JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)

        print(f"Results saved to: {output_file}")
        print(f"Configuration used: {self.config_path}")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python run_pipeline.py <config_file.json>")
        print("\nAvailable configurations (in configs/ folder):")
        print("  python run_pipeline.py configs/config_utility_ml_efficiency.json  # ML efficiency (TTR, TTS, TRTS, TSTR, TTRS)")
        print("  python run_pipeline.py configs/config_evaluation_only.json        # KS Test only")
        print("  python run_pipeline.py configs/config_with_mia.json               # KS Test + MIA (Privacy)")
        print("  python run_pipeline.py configs/config_full_pipeline.json          # Full pipeline (all phases)")
        print("\nResults are saved to: results/")
        sys.exit(1)

    config_file = sys.argv[1]

    try:
        runner = PipelineRunner(config_file)
        runner.run()
    except KeyboardInterrupt:
        print("\n\nExecution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
