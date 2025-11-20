# ValtaPy Pipeline - Main Script

## Overview

This `main.py` script demonstrates the complete ValtaPy evaluation pipeline for assessing synthetic data quality. It orchestrates three main phases:

1. **Ingestion**: Reading real and synthetic data from CSV files
2. **Preprocessing**: Cleaning and preparing data for evaluation
3. **Evaluation**: Computing metrics (KS Test) to assess fidelity

## Usage

### Running the Pipeline

Simply execute the main script from the project root:

```bash
python main.py
```

### Default Configuration

The script is pre-configured to evaluate:
- **Real data**: `data/real/a_hepatitis_sampling_baseline.csv`
- **Synthetic data**: `data/synt/f_hepatitis_CTGAN_syn_best.csv`
- **Metric**: Kolmogorov-Smirnov Test (KS Test) for distribution similarity

## Pipeline Phases

### 1. Ingestion Phase

The ingestion phase reads both datasets using the `CSVReader`:

```python
from valtapy.ingestion.entities import DataSource
from valtapy.ingestion.readers.csv_reader import CSVReader

reader = CSVReader()
real_result = reader.read(DataSource(path="data/real/...", format="csv"))
synth_result = reader.read(DataSource(path="data/synt/...", format="csv"))
```

**Output**:
- DataFrames with loaded data
- Metadata (file size, memory usage, shapes, etc.)

### 2. Preprocessing Phase

The preprocessing phase cleans and prepares data using `BasicDataPreprocessor`:

```python
from valtapy.preprocessing.entities import PreprocessingConfig
from valtapy.preprocessing.processors.basic_preprocessor import BasicDataPreprocessor

config = PreprocessingConfig(
    drop_empty_rows=True,
    drop_empty_columns=True,
    na_strategy="keep"
)

preprocessor = BasicDataPreprocessor()
result = preprocessor.preprocess(real_df, synth_df, config)
```

**Output**:
- Cleaned DataFrames
- Preprocessing report with changes and compatibility issues

### 3. Evaluation Phase

The evaluation phase computes the KS Test metric:

```python
from valtapy.evaluation.metrics.fidelity.ks_test import KSTestMetric
from valtapy.evaluation.entities import MetricExecutionContext

metric = KSTestMetric()
context = MetricExecutionContext(
    real_data=real_df,
    synth_data=synth_df,
    parameters={},
    cache_enabled=False,
    random_seed=42
)

result = metric.compute(context)
```

**Output**:
- Overall fidelity score (p-value)
- Per-column KS statistics and p-values
- Interpretation of results

## Output

### Console Output

The script provides detailed console output for each phase:

```
================================================================================
  ValtaPy - Synthetic Data Evaluation Pipeline
================================================================================

================================================================================
  Phase 1: Data Ingestion
================================================================================

Real data source:      data/real/a_hepatitis_sampling_baseline.csv
Synthetic data source: data/synt/f_hepatitis_CTGAN_syn_best.csv

Reading real data...
  [OK] Loaded 1000 rows, 29 columns
  [OK] File size: 0.17 MB
  [OK] Memory usage: 0.63 MB

...
```

### JSON Output

Results are saved to `evaluation_results.json` with the following structure:

```json
{
  "timestamp": "2025-10-17T16:43:19.687449",
  "data_sources": {
    "real": "data/real/a_hepatitis_sampling_baseline.csv",
    "synthetic": "data/synt/f_hepatitis_CTGAN_syn_best.csv"
  },
  "data_shapes": {
    "real": { "original": [1000, 29], "preprocessed": [1000, 29] },
    "synthetic": { "original": [1000, 29], "preprocessed": [1000, 29] }
  },
  "preprocessing": {
    "config": { ... },
    "processing_time": 0.0028,
    "operations_applied": ["empty_data_cleaning"],
    "compatibility_issues": []
  },
  "evaluation": {
    "metric_id": "fidelity.ks",
    "metric_family": "fidelity",
    "overall_score": 0.003702,
    "overall_ks_statistic": 0.064810,
    "columns_tested": 21,
    "columns_failed": 0,
    "column_results": { ... }
  }
}
```

## Understanding Results

### KS Test Metric

The Kolmogorov-Smirnov test compares the distributions of real and synthetic data:

- **P-value (Overall Score)**:
  - `>= 0.05`: Good fidelity (distributions are similar)
  - `0.01 - 0.05`: Moderate fidelity (some differences)
  - `< 0.01`: Poor fidelity (significant differences)

- **KS Statistic**:
  - Lower values indicate more similar distributions
  - Ranges from 0 (identical) to 1 (completely different)

### Per-Column Results

Each numeric column is tested individually:

```
Column Name                    KS Statistic    P-Value      Significant
--------------------------------------------------------------------------------
Age                            0.078000        0.004544     Yes
BMI                            0.165000        0.000000     Yes
Fever                          0.008000        1.000000     No
```

- **Significant = Yes**: Distributions are significantly different (p < 0.05)
- **Significant = No**: Distributions are similar (p >= 0.05)

## Customization

### Using Different Data Files

Modify the file paths in `main.py`:

```python
real_data_path = "path/to/your/real_data.csv"
synth_data_path = "path/to/your/synthetic_data.csv"
```

### Adjusting Preprocessing

Modify the `PreprocessingConfig`:

```python
preproc_config = PreprocessingConfig(
    drop_empty_rows=True,
    drop_empty_columns=True,
    na_strategy="keep",  # or "drop", "fill"
    # More options available but not yet implemented
)
```

### Adding More Metrics

The architecture supports multiple metrics. To add more:

```python
# Example for future extension
from valtapy.evaluation.metrics.privacy.membership_inference import MembershipInferenceMetric

privacy_metric = MembershipInferenceMetric()
privacy_result = privacy_metric.compute(context)
```

## Architecture

The pipeline follows clean architecture principles:

```
┌─────────────────┐
│   main.py       │  ← Orchestration Layer
└────────┬────────┘
         │
    ┌────▼─────────────────────────────┐
    │                                   │
┌───▼────────┐  ┌──────────────┐  ┌───▼─────────┐
│ Ingestion  │  │Preprocessing │  │ Evaluation  │
│   Phase    │─►│    Phase     │─►│    Phase    │
└────────────┘  └──────────────┘  └─────────────┘
    │                │                  │
┌───▼─────┐    ┌────▼────┐        ┌────▼────┐
│Readers  │    │Processor│        │ Metrics │
│(CSV)    │    │(Basic)  │        │(KS Test)│
└─────────┘    └─────────┘        └─────────┘
```

Each phase:
- Uses contracts (protocols) for dependency inversion
- Has dedicated entities for data modeling
- Implements proper error handling
- Provides detailed reporting

## Requirements

- Python 3.13+
- pandas
- numpy
- scipy (for KS test computation)

## Example Output

```
================================================================================
  Evaluation Results
================================================================================

Overall Score (p-value): 0.003702
Overall KS Statistic:    0.064810

Columns tested:  21
Columns failed:  0

Interpretation:
  [POOR] POOR FIDELITY: Distributions are significantly different
         The synthetic data does not preserve the statistical properties well.

[WARNING] Columns with significant differences (p < 0.05):
  - ALT 36: Distributions are significantly different (poor fidelity)
  - Age: Distributions are significantly different (poor fidelity)
  - BMI: Distributions are significantly different (poor fidelity)
  ...
```

## Next Steps

This main script provides a foundation for:
1. Adding more evaluation metrics
2. Implementing metric orchestration
3. Creating batch evaluation pipelines
4. Building visualization dashboards
5. Integrating with CI/CD workflows

For more details on individual components, see the respective module documentation.
