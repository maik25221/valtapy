"""Configuration loading and validation."""

import yaml
from pathlib import Path
from typing import Dict, Any
from ...domain.errors import ConfigError


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load and validate YAML configuration file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Parsed configuration dictionary
        
    Raises:
        ConfigError: If config file cannot be loaded or is invalid
    """
    try:
        path = Path(config_path)
        if not path.exists():
            raise ConfigError(f"Configuration file not found: {config_path}", config_path)
        
        with open(path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        if config is None:
            raise ConfigError("Configuration file is empty", config_path)
        
        # Basic validation
        _validate_config_structure(config, config_path)
        
        return config
        
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML syntax: {e}", config_path)
    except Exception as e:
        raise ConfigError(f"Failed to load configuration: {e}", config_path)


def _validate_config_structure(config: Dict[str, Any], config_path: str) -> None:
    """Validate basic configuration structure."""
    
    # Required top-level sections
    required_sections = ["data"]
    for section in required_sections:
        if section not in config:
            raise ConfigError(f"Missing required section: {section}", config_path)
    
    # Validate data section
    data_config = config["data"]
    if not isinstance(data_config, dict):
        raise ConfigError("'data' section must be a dictionary", config_path)
    
    required_data_keys = ["real", "synthetic"]
    for key in required_data_keys:
        if key not in data_config:
            raise ConfigError(f"Missing required key in data section: {key}", config_path)
    
    # Validate evaluation section (if present)
    if "evaluation" in config:
        eval_config = config["evaluation"]
        if not isinstance(eval_config, dict):
            raise ConfigError("'evaluation' section must be a dictionary", config_path)
        
        # Check that either metric_ids or purpose is specified
        has_metrics = "metric_ids" in eval_config
        has_purpose = "purpose" in eval_config
        
        if not has_metrics and not has_purpose:
            raise ConfigError("Either 'metric_ids' or 'purpose' must be specified in evaluation", config_path)
    
    # Validate report section (if present)
    if "report" in config:
        report_config = config["report"]
        if not isinstance(report_config, dict):
            raise ConfigError("'report' section must be a dictionary", config_path)
        
        if "formats" not in report_config:
            raise ConfigError("'formats' must be specified in report section", config_path)
        
        if "output_dir" not in report_config:
            raise ConfigError("'output_dir' must be specified in report section", config_path)


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary to save
        config_path: Output file path
    """
    try:
        path = Path(config_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2, sort_keys=True)
            
    except Exception as e:
        raise ConfigError(f"Failed to save configuration: {e}", config_path)


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries with override taking precedence.
    
    Args:
        base_config: Base configuration
        override_config: Override configuration
        
    Returns:
        Merged configuration dictionary
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            merged[key] = merge_configs(merged[key], value)
        else:
            # Override value
            merged[key] = value
    
    return merged


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration template.
    
    Returns:
        Default configuration dictionary
    """
    return {
        "data": {
            "real": "path/to/real.csv",
            "synthetic": "path/to/synthetic.csv",
            "target": None,
            "dtypes": {},
            "constraints": {}
        },
        "evaluation": {
            "purpose": "privacy_hardening",
            "metric_ids": [
                "fidelity.ks",
                "utility.pmse", 
                "privacy.nndr"
            ],
            "seed": 42,
            "cv_splits": 3,
            "models": None
        },
        "aggregation": {
            "weights": {
                "fidelity": 0.4,
                "utility": 0.3, 
                "privacy": 0.3
            },
            "composite_index": True
        },
        "report": {
            "formats": ["json", "md"],
            "output_dir": "reports/evaluation_run",
            "include_details": True,
            "include_artifacts": False
        },
        "preprocessing": {
            "preprocessor": "tabular",
            "missing_values": "pandas",
            "scaling": False
        },
        "runtime": {
            "max_workers": 0,
            "cache_size": 1000,
            "log_level": "INFO"
        }
    }


def validate_file_paths(config: Dict[str, Any]) -> list[str]:
    """
    Validate that file paths in configuration exist.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        List of validation warnings
    """
    warnings = []
    
    # Check data file paths
    data_config = config.get("data", {})
    
    for key in ["real", "synthetic"]:
        if key in data_config:
            path = Path(data_config[key])
            if not path.exists():
                warnings.append(f"Data file not found: {data_config[key]}")
            elif not path.is_file():
                warnings.append(f"Data path is not a file: {data_config[key]}")
    
    return warnings