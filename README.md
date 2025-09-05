# ValtaPyV2

A modular framework for evaluating synthetic tabular data quality, utility, and privacy preservation.

## Overview

ValtaPyV2 provides a clean, extensible architecture for comprehensive synthetic data evaluation across three key dimensions:

- **Fidelity**: How well synthetic data preserves statistical properties of real data
- **Utility**: How useful synthetic data is for downstream machine learning tasks  
- **Privacy**: How well synthetic data protects individual privacy

## Architecture

The framework follows SOLID principles with clear separation of concerns:

```
├── domain/          # Core contracts, entities, and business logic
├── application/     # Orchestration and use case implementations
├── infrastructure/ # Technical implementations (I/O, metrics, reporting)
└── interface/      # CLI and SDK for user interaction
```

### Key Design Patterns

- **Factory Pattern**: Metric and reporter creation
- **Strategy Pattern**: Pluggable metric implementations
- **Orchestrator Pattern**: Coordinated execution with shared caching
- **Protocol-Based Design**: Type-safe interfaces using `typing.Protocol`

## Quick Start

### Installation

```bash
pip install -e .
```

### Basic Usage

1. **Generate configuration template:**
```bash
python -m valtapyV2.interface.cli init-config --output my_config.yaml
```

2. **Edit configuration** to specify your data paths and preferences

3. **Run evaluation:**
```bash
python -m valtapyV2.interface.cli evaluate --config my_config.yaml
```

### Programmatic Usage

```python
from valtapyV2 import Evaluator

# Quick evaluation
evaluator = Evaluator()
results = evaluator.quick_eval(
    real_data_path="data/real.csv",
    synthetic_data_path="data/synthetic.csv",
    purpose="privacy_hardening",
    output_dir="reports"
)

# Get summary
summary = evaluator.get_result_summary(results)
print(f"Composite Score: {summary['scores']['composite']:.3f}")
```

## Configuration

ValtaPyV2 uses YAML configuration files with the following structure:

```yaml
data:
  real: "path/to/real.csv"
  synthetic: "path/to/synthetic.csv"
  target: "target_column"  # Optional, required for utility metrics

evaluation:
  purpose: "privacy_hardening"  # Or specify metric_ids explicitly
  seed: 42
  cv_splits: 3

aggregation:
  weights:
    fidelity: 0.4
    utility: 0.3
    privacy: 0.3
  composite_index: true

report:
  formats: ["json", "md"]
  output_dir: "reports/evaluation"
```

### Evaluation Purposes

Choose from predefined purposes that automatically select appropriate metrics:

- `privacy_hardening`: Focus on privacy and fidelity metrics
- `model_selection`: Focus on utility and fidelity metrics  
- `data_release`: Balanced evaluation across all dimensions
- `research_validation`: Comprehensive fidelity assessment

Or specify explicit `metric_ids` for custom evaluation profiles.

## Available Metrics

### Fidelity Metrics

- `fidelity.ks`: Kolmogorov-Smirnov test for distribution similarity
- `fidelity.correlation_delta`: Correlation matrix comparison
- `fidelity.mi_delta`: Mutual information comparison (stub)

### Utility Metrics

- `utility.pmse`: Predictive Mean Squared Error (TSTR evaluation)
- `utility.accuracy`: Classification accuracy preservation

### Privacy Metrics

- `privacy.nndr`: Nearest Neighbor Distance Ratio
- `privacy.membership_inference`: Membership inference attack simulation

## Extending ValtaPyV2

### Adding New Metrics

1. **Implement the Metric protocol:**

```python
from valtapyV2.infrastructure.metrics.registry import register
from valtapyV2.infrastructure.metrics.base import MetricBase
from valtapyV2.domain.entities import MetricResult

@register("family.metric_name")
class MyMetric(MetricBase):
    name = "my_metric"
    family = "fidelity"  # or "utility", "privacy"
    purpose_tags = {"tag1", "tag2"}
    
    def fit(self, real_data, synth_data, context):
        self._setup(real_data, synth_data, context)
        return self
    
    def compute(self):
        # Your computation logic here
        value = self._compute_metric_value()
        
        return MetricResult(
            id="family.metric_name",
            value=value,
            details={"computation_info": "details"},
            family=self.family,
            purpose_tags=self.purpose_tags
        )
```

2. **The metric is automatically registered** via the `@register` decorator

### Using Cached Computations

Metrics can leverage the shared `StatsStore` for expensive computations:

```python
def compute(self):
    # Use cached correlation matrix
    corr_matrix = self._get_correlation_matrix("real")
    
    # Use cached train/test splits
    splits = self._get_train_test_splits(n_splits=3)
    
    # Use cached KNN distances for privacy metrics
    knn_data = self._get_knn_distances(k=5)
```

## Development

### Running Tests

```bash
pytest -q
```

### Project Commands

```bash
# List available metrics
python -m valtapyV2.interface.cli list-metrics

# Generate default config
python -m valtapyV2.interface.cli init-config --output config.yaml

# Run evaluation
python -m valtapyV2.interface.cli evaluate --config config.yaml

# Run with verbose logging
python -m valtapyV2.interface.cli --verbose evaluate --config config.yaml
```

## Technical Details

### Dependencies

- **Core**: pandas, numpy, pyyaml  
- **ML**: scikit-learn (for utility metrics)
- **Statistics**: scipy (for statistical tests)
- **Development**: pytest

### Performance Features

- **Parallel Execution**: CPU-bound metrics run in parallel using multiprocessing
- **Shared Caching**: Expensive computations (correlations, train/test splits, KNN) are cached and reused
- **Graceful Degradation**: Individual metric failures don't stop the entire evaluation

### Output Formats

- **JSON**: Structured data for programmatic consumption
- **Markdown**: Human-readable reports with tables and summaries
- **Extensible**: Add custom reporters by implementing the `Reporter` protocol

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]