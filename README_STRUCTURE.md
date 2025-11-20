# ValtaPy Project Structure

## 📁 Directory Organization

```
valtapy/
├── configs/                          # Configuration files
│   ├── config_utility_ml_efficiency.json   # ML efficiency metrics (TTR, TTS, TRTS, TSTR, TTRS)
│   ├── config_evaluation_only.json         # Evaluation only (no preprocessing)
│   ├── config_with_mia.json               # Fidelity + Privacy (MIA)
│   ├── config_full_pipeline.json          # Full pipeline (all phases)
│   └── config.json                        # Basic configuration
│
├── results/                          # Evaluation results
│   ├── evaluation_results_ml_utility.json   # ML efficiency results
│   ├── evaluation_results_no_preproc.json   # Evaluation without preprocessing
│   ├── evaluation_results_with_mia.json     # Results with privacy metrics
│   └── evaluation_results_full.json         # Full pipeline results
│
├── data/                             # Data files
│   ├── real/                         # Real datasets
│   │   └── a_hepatitis_sampling_baseline.csv
│   └── synt/                         # Synthetic datasets
│       └── f_hepatitis_CTGAN_syn_best.csv
│
├── src/valtapy/                      # Source code
│   ├── ingestion/                    # Data ingestion phase
│   ├── preprocessing/                # Data preprocessing phase
│   ├── validation/                   # Data validation phase
│   └── evaluation/                   # Evaluation phase
│       ├── metrics/
│       │   ├── fidelity/            # Fidelity metrics (KS test)
│       │   ├── privacy/             # Privacy metrics (MIA)
│       │   └── utility/             # Utility metrics (ML efficiency)
│       │       ├── ttr.py           # Train on Training, Test on Real
│       │       ├── tts.py           # Train on Training, Test on Synthetic
│       │       ├── trts.py          # Train on Real, Test on Synthetic
│       │       ├── tstr.py          # Train on Synthetic, Test on Real
│       │       └── ttrs.py          # Train on Training, Test on Both
│       ├── contracts.py             # Metric protocols
│       ├── entities.py              # Domain entities
│       └── registry.py              # Metric registry
│
├── tests/                            # Test suite
│   ├── unit/
│   │   └── evaluation/
│   │       ├── test_ml_efficiency_metrics.py
│   │       └── test_ml_metrics_output.py
│   └── integration/
│
├── run_pipeline.py                   # Main pipeline runner
├── main.py                          # Alternative entry point
└── pyproject.toml                   # Project configuration

```

## 🚀 Usage

### Running Evaluations

All configurations are in the `configs/` folder. Results are automatically saved to `results/`.

```bash
# ML Efficiency Utility Evaluation (TTR, TTS, TRTS, TSTR, TTRS)
python run_pipeline.py configs/config_utility_ml_efficiency.json

# Evaluation Only (KS Test)
python run_pipeline.py configs/config_evaluation_only.json

# Fidelity + Privacy (KS Test + MIA)
python run_pipeline.py configs/config_with_mia.json

# Full Pipeline (All Phases)
python run_pipeline.py configs/config_full_pipeline.json
```

### Configuration Structure

Each config file specifies:
- **Pipeline phases**: Which phases to run (ingestion, preprocessing, evaluation)
- **Data sources**: Paths to real and synthetic data
- **Metrics**: Which metrics to compute and their parameters
- **Output**: Where to save results (automatically in `results/`)

### Results

All evaluation results are saved as JSON files in `results/` with:
- Timestamp
- Configuration used
- Detailed metric results
- Computation times
- Metadata

## 📊 Available Metrics

### Fidelity Metrics
- **KS Test** (`fidelity.ks`): Kolmogorov-Smirnov test for distribution similarity

### Privacy Metrics
- **MIA** (`privacy.mia`): Membership Inference Attack for privacy risk assessment

### Utility Metrics (ML Efficiency)
- **TTR** (`ml_efficiency_ttr`): Train on Training, Test on Real - Baseline
- **TTS** (`ml_efficiency_tts`): Train on Training, Test on Synthetic - Pattern similarity
- **TRTS** (`ml_efficiency_trts`): Train on Real, Test on Synthetic - Distribution match
- **TSTR** (`ml_efficiency_tstr`): Train on Synthetic, Test on Real - Generalization (KEY)
- **TTRS** (`ml_efficiency_ttrs`): Train on Training, Test on Both - Consistency

## 🔧 Development

### Adding New Metrics

1. Create metric implementation in `src/valtapy/evaluation/metrics/<family>/`
2. Register in `src/valtapy/evaluation/registry.py`
3. Add to `run_pipeline.py` available metrics dictionary
4. Create configuration file in `configs/`

### Running Tests

```bash
# All tests
pytest

# Specific test file
pytest tests/unit/evaluation/test_ml_efficiency_metrics.py

# With verbose output
pytest -v -s
```

## 📝 Notes

- All paths in configs are relative to the project root
- Results include complete reproducibility information
- Metrics automatically handle categorical variables via label encoding
- DecisionTree is the default model for ML efficiency metrics
