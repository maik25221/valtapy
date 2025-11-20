# ValtaPy Configuration-Based Pipeline

## Overview

The `run_pipeline.py` script provides a flexible, configuration-based way to run ValtaPy evaluations. Instead of hardcoding your pipeline, you define everything in a JSON configuration file.

This allows you to:
- **Skip phases you don't need** (e.g., skip preprocessing if data is already clean)
- **Run multiple metrics** with different parameters
- **Configure all aspects** of the pipeline without touching code
- **Create reusable configurations** for different datasets or evaluation scenarios

## Quick Start

### 1. Choose or Create a Configuration File

ValtaPy includes several example configurations:

```bash
# Full pipeline (ingestion + preprocessing + evaluation)
python run_pipeline.py config.json

# Evaluation only (skip preprocessing)
python run_pipeline.py config_evaluation_only.json

# Full pipeline with validation
python run_pipeline.py config_full_pipeline.json
```

### 2. Run the Pipeline

```bash
python run_pipeline.py <your_config.json>
```

### 3. Check Results

Results are saved to the file specified in your config's `pipeline.output_file`:

```json
{
  "timestamp": "2025-10-17T16:55:24.204755",
  "config_file": "config.json",
  "pipeline": { ... },
  "results": {
    "ingestion": { ... },
    "preprocessing": { ... },
    "evaluation": { ... }
  }
}
```

## Configuration File Structure

### Minimal Configuration

The simplest possible configuration:

```json
{
  "pipeline": {
    "name": "My Evaluation",
    "enabled_phases": ["ingestion", "evaluation"],
    "output_file": "results.json"
  },
  "data_sources": {
    "real": {
      "path": "data/real/my_real_data.csv",
      "format": "csv"
    },
    "synthetic": {
      "path": "data/synthetic/my_synth_data.csv",
      "format": "csv"
    }
  },
  "evaluation": {
    "enabled": true,
    "metrics": [
      {
        "metric_id": "fidelity.ks",
        "enabled": true
      }
    ]
  }
}
```

### Full Configuration

A complete configuration with all options:

```json
{
  "pipeline": {
    "name": "Full Pipeline Example",
    "description": "Comprehensive evaluation with all phases",
    "enabled_phases": ["ingestion", "validation", "preprocessing", "evaluation"],
    "output_file": "full_results.json"
  },
  "data_sources": {
    "real": {
      "path": "data/real/dataset.csv",
      "format": "csv",
      "read_params": {
        "encoding": "utf-8",
        "na_values": ["", "NA", "null"],
        "sep": ","
      }
    },
    "synthetic": {
      "path": "data/synth/dataset.csv",
      "format": "csv",
      "read_params": {
        "encoding": "utf-8",
        "na_values": ["", "NA", "null"],
        "sep": ","
      }
    }
  },
  "validation": {
    "enabled": true,
    "config": {
      "check_schema": true,
      "check_data_types": true,
      "check_column_alignment": true,
      "strict_mode": false
    }
  },
  "preprocessing": {
    "enabled": true,
    "config": {
      "drop_empty_rows": true,
      "drop_empty_columns": true,
      "na_strategy": "keep",
      "enforce_consistent_types": false,
      "normalize_numeric": false,
      "encode_categorical": false
    }
  },
  "evaluation": {
    "enabled": true,
    "metrics": [
      {
        "metric_id": "fidelity.ks",
        "enabled": true,
        "parameters": {
          "significance_level": 0.05
        }
      }
    ],
    "random_seed": 42,
    "cache_enabled": false
  }
}
```

## Configuration Sections

### 1. Pipeline Section

Defines overall pipeline settings.

```json
{
  "pipeline": {
    "name": "Evaluation Name",              // Required: Human-readable name
    "description": "What this does",        // Optional: Description
    "enabled_phases": [                     // Required: Which phases to run
      "ingestion",                          // Always required
      "validation",                         // Optional
      "preprocessing",                      // Optional
      "evaluation"                          // Typically required
    ],
    "output_file": "results.json"           // Optional: Default "pipeline_results.json"
  }
}
```

**Available Phases:**
- `ingestion` - Load data from files (required)
- `validation` - Validate data structure and compatibility (optional, placeholder)
- `preprocessing` - Clean and prepare data (optional)
- `evaluation` - Compute metrics (optional but typically needed)

### 2. Data Sources Section

Specifies where to load real and synthetic data.

```json
{
  "data_sources": {
    "real": {
      "path": "path/to/real_data.csv",     // Required: File path
      "format": "csv",                      // Required: File format (currently only "csv")
      "read_params": {                      // Optional: pandas read_csv parameters
        "encoding": "utf-8",
        "sep": ",",
        "na_values": ["NA", "null"],
        "header": 0
      }
    },
    "synthetic": {
      "path": "path/to/synth_data.csv",
      "format": "csv",
      "read_params": { ... }
    }
  }
}
```

**Supported Formats:**
- `csv` - Comma-separated values (default)

**Common Read Parameters:**
- `encoding` - File encoding (default: "utf-8")
- `sep` - Column separator (default: ",")
- `na_values` - Values to treat as NaN
- `header` - Row number for column names
- Any pandas `read_csv` parameter

### 3. Validation Section (Optional)

Configure data validation (currently placeholder).

```json
{
  "validation": {
    "enabled": true,                       // Enable/disable validation
    "config": {
      "check_schema": true,                // Check column schemas match
      "check_data_types": true,            // Check data type consistency
      "check_column_alignment": true,      // Check columns align
      "strict_mode": false                 // Fail on any validation error
    }
  }
}
```

**Note:** Validation is currently a placeholder. Set `enabled: false` or omit from `enabled_phases`.

### 4. Preprocessing Section (Optional)

Configure data preprocessing.

```json
{
  "preprocessing": {
    "enabled": true,                       // Enable/disable preprocessing
    "config": {
      "drop_empty_rows": true,             // Drop completely empty rows
      "drop_empty_columns": true,          // Drop completely empty columns
      "na_strategy": "keep",               // How to handle NaN: "keep", "drop", "fill"
      "enforce_consistent_types": false,   // Align dtypes (not implemented)
      "normalize_numeric": false,          // Normalize numbers (not implemented)
      "encode_categorical": false          // Encode categories (not implemented)
    }
  }
}
```

**Preprocessing Options:**

| Option | Values | Description |
|--------|--------|-------------|
| `drop_empty_rows` | `true`/`false` | Remove rows with all NaN values |
| `drop_empty_columns` | `true`/`false` | Remove columns with all NaN values |
| `na_strategy` | `"keep"`, `"drop"`, `"fill"` | How to handle missing values |
| Others | - | Not yet implemented |

### 5. Evaluation Section

Configure metrics to compute.

```json
{
  "evaluation": {
    "enabled": true,                       // Enable/disable evaluation
    "metrics": [                            // List of metrics to compute
      {
        "metric_id": "fidelity.ks",        // Metric identifier
        "enabled": true,                   // Enable/disable this metric
        "parameters": {                    // Metric-specific parameters
          "significance_level": 0.05
        }
      }
    ],
    "random_seed": 42,                     // Random seed for reproducibility
    "cache_enabled": false                 // Enable metric caching
  }
}
```

**Available Metrics:**

| Metric ID | Family | Description | Parameters |
|-----------|--------|-------------|------------|
| `fidelity.ks` | Fidelity | Kolmogorov-Smirnov test for distribution similarity | `significance_level` (default: 0.05) |

**Evaluation Options:**
- `random_seed` - Random seed for reproducible results
- `cache_enabled` - Cache metric computations (useful for expensive metrics)

## Use Cases

### Use Case 1: Quick Evaluation (No Preprocessing)

When your data is already clean and you just want to compute metrics:

```json
{
  "pipeline": {
    "name": "Quick Evaluation",
    "enabled_phases": ["ingestion", "evaluation"],
    "output_file": "quick_eval.json"
  },
  "data_sources": { ... },
  "preprocessing": {
    "enabled": false
  },
  "evaluation": {
    "enabled": true,
    "metrics": [
      { "metric_id": "fidelity.ks", "enabled": true }
    ]
  }
}
```

### Use Case 2: Full Pipeline

When you want complete processing:

```json
{
  "pipeline": {
    "name": "Full Evaluation",
    "enabled_phases": ["ingestion", "preprocessing", "evaluation"]
  },
  "data_sources": { ... },
  "preprocessing": {
    "enabled": true,
    "config": {
      "drop_empty_rows": true,
      "drop_empty_columns": true,
      "na_strategy": "keep"
    }
  },
  "evaluation": {
    "enabled": true,
    "metrics": [
      { "metric_id": "fidelity.ks", "enabled": true }
    ]
  }
}
```

### Use Case 3: Multiple Metrics (Future)

When more metrics are implemented:

```json
{
  "evaluation": {
    "enabled": true,
    "metrics": [
      {
        "metric_id": "fidelity.ks",
        "enabled": true,
        "parameters": { "significance_level": 0.05 }
      },
      {
        "metric_id": "privacy.membership_inference",
        "enabled": true,
        "parameters": { "attack_type": "shadow_model" }
      },
      {
        "metric_id": "utility.ml_efficacy",
        "enabled": true,
        "parameters": { "model": "random_forest" }
      }
    ]
  }
}
```

### Use Case 4: Batch Evaluation

Evaluate multiple datasets with different configurations:

```bash
# Evaluate CTGAN
python run_pipeline.py config_ctgan.json

# Evaluate TVAE
python run_pipeline.py config_tvae.json

# Evaluate Gaussian Copula
python run_pipeline.py config_gaussiancopula.json
```

## Output Format

### Console Output

The pipeline provides structured console output:

```
================================================================================
  ValtaPy Pipeline: Hepatitis Data Evaluation
================================================================================

Description: Evaluate CTGAN synthetic hepatitis data against baseline
Enabled phases: ingestion, preprocessing, evaluation
Configuration file: config.json

================================================================================
  Phase 1: Data Ingestion
================================================================================

Real data source:      data/real/a_hepatitis_sampling_baseline.csv
Synthetic data source: data/synt/f_hepatitis_CTGAN_syn_best.csv

Reading real data...
  [OK] Loaded 1000 rows, 29 columns
  [OK] File size: 0.17 MB
...
```

### JSON Output

Results are saved to a structured JSON file:

```json
{
  "timestamp": "2025-10-17T16:55:24.204755",
  "config_file": "config.json",
  "pipeline": {
    "name": "Hepatitis Data Evaluation",
    "enabled_phases": ["ingestion", "preprocessing", "evaluation"]
  },
  "results": {
    "ingestion": {
      "real": {
        "path": "data/real/...",
        "shape": [1000, 29],
        "columns": [...]
      },
      "synthetic": { ... }
    },
    "preprocessing": {
      "enabled": true,
      "processing_time": 0.0025,
      "operations_applied": ["empty_data_cleaning"],
      "compatibility_issues": []
    },
    "evaluation": {
      "enabled": true,
      "metrics_computed": 1,
      "results": [
        {
          "metric_id": "fidelity.ks",
          "value": 0.003702,
          "computation_time": 0.6322,
          "details": { ... }
        }
      ]
    }
  }
}
```

## Advanced Topics

### Creating Custom Configurations

1. **Copy an example configuration:**
   ```bash
   cp config.json my_config.json
   ```

2. **Edit your configuration:**
   - Update data paths
   - Enable/disable phases
   - Configure metrics
   - Set parameters

3. **Run your configuration:**
   ```bash
   python run_pipeline.py my_config.json
   ```

### Comparing Multiple Generators

Create a config for each generator:

```bash
# config_ctgan.json
{
  "pipeline": { "name": "CTGAN Evaluation", ... },
  "data_sources": {
    "synthetic": { "path": "data/synth/ctgan_output.csv" }
  }
}

# config_tvae.json
{
  "pipeline": { "name": "TVAE Evaluation", ... },
  "data_sources": {
    "synthetic": { "path": "data/synth/tvae_output.csv" }
  }
}
```

Then run and compare:

```bash
python run_pipeline.py config_ctgan.json
python run_pipeline.py config_tvae.json
# Compare results_ctgan.json vs results_tvae.json
```

### Reproducible Experiments

Always set `random_seed` for reproducible results:

```json
{
  "evaluation": {
    "random_seed": 42,  // Same seed = same results
    "cache_enabled": false
  }
}
```

## Troubleshooting

### "Configuration file not found"

Make sure the config file exists:
```bash
ls config.json
```

### "Missing required section"

Your config must have `pipeline` and `data_sources` sections:
```json
{
  "pipeline": { ... },      // Required
  "data_sources": { ... }   // Required
}
```

### "Metric 'X' not found"

Check the metric ID is correct:
- Available: `fidelity.ks`
- Coming soon: `privacy.*`, `utility.*`

### Phase not running

Check `enabled_phases` includes the phase:
```json
{
  "pipeline": {
    "enabled_phases": ["ingestion", "preprocessing", "evaluation"]
  }
}
```

## Examples

See the included example configurations:

- **`config.json`** - Standard full pipeline
- **`config_evaluation_only.json`** - Skip preprocessing
- **`config_full_pipeline.json`** - All phases including validation

## Next Steps

1. **Add more metrics** as they are implemented
2. **Create custom configs** for your datasets
3. **Automate evaluations** with scripts
4. **Compare generators** using multiple configs
5. **Build dashboards** from JSON results

For more details on individual phases, see:
- Ingestion: `src/valtapy/ingestion/`
- Preprocessing: `src/valtapy/preprocessing/`
- Evaluation: `src/valtapy/evaluation/`
