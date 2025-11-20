"""High-level pipeline orchestration."""

from typing import Dict, Any
from ..domain.entities import RunSummary
from .orchestrator import Orchestrator


def run_from_config(config_path: str) -> RunSummary:
    """
    Run complete evaluation pipeline from configuration file.
    
    This is the main entry point for the evaluation system, providing
    a simple interface that delegates to the Orchestrator.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        RunSummary containing complete evaluation results
        
    Raises:
        ConfigError: If configuration is invalid
        SchemaError: If data validation fails
        MetricExecutionError: If critical metrics fail
    """
    orchestrator = Orchestrator()
    return orchestrator.run(config_path)