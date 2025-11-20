"""Python SDK for programmatic access to ValtaPyV2."""

from typing import Dict, Any, Optional
from pathlib import Path

from ..application.pipeline import run_from_config
from ..domain.entities import RunSummary
from ..infrastructure.runtime.config import get_default_config, save_config, merge_configs


class Evaluator:
    """
    Main SDK class for programmatic evaluation of synthetic data.
    
    This class provides a high-level interface for running evaluations,
    delegating to the application pipeline while providing convenience methods.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize evaluator with optional configuration override.
        
        Args:
            config: Optional configuration dictionary to override defaults
        """
        self._config_override = config or {}
    
    def run(self, config_path: str) -> RunSummary:
        """
        Run evaluation from configuration file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            RunSummary with complete evaluation results
        """
        return run_from_config(config_path)
    
    def run_with_override(self, config_path: str, config_override: Dict[str, Any]) -> RunSummary:
        """
        Run evaluation with configuration overrides.
        
        Args:
            config_path: Base configuration file path
            config_override: Configuration values to override
            
        Returns:
            RunSummary with complete evaluation results
        """
        from ..infrastructure.runtime.config import load_config
        import tempfile
        import os
        
        # Load base config and merge with overrides
        base_config = load_config(config_path)
        merged_config = merge_configs(base_config, config_override)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name
        
        try:
            save_config(merged_config, temp_path)
            return run_from_config(temp_path)
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_path)
            except:
                pass
    
    def quick_eval(self, real_data_path: str, synthetic_data_path: str, 
                   purpose: str = "privacy_hardening", 
                   output_dir: Optional[str] = None) -> RunSummary:
        """
        Quick evaluation with minimal configuration.
        
        Args:
            real_data_path: Path to real data CSV
            synthetic_data_path: Path to synthetic data CSV
            purpose: Evaluation purpose (privacy_hardening, model_selection, etc.)
            output_dir: Optional output directory for reports
            
        Returns:
            RunSummary with evaluation results
        """
        import tempfile
        import os
        
        # Create minimal configuration
        config = {
            "data": {
                "real": real_data_path,
                "synthetic": synthetic_data_path
            },
            "evaluation": {
                "purpose": purpose,
                "seed": 42
            }
        }
        
        if output_dir:
            config["report"] = {
                "formats": ["json", "md"],
                "output_dir": output_dir
            }
        
        # Merge with any instance overrides
        if self._config_override:
            config = merge_configs(config, self._config_override)
        
        # Save to temporary file and run
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name
        
        try:
            save_config(config, temp_path)
            return run_from_config(temp_path)
        finally:
            try:
                os.unlink(temp_path)
            except:
                pass
    
    def create_config_template(self, output_path: str, 
                              real_data_path: Optional[str] = None,
                              synthetic_data_path: Optional[str] = None,
                              purpose: Optional[str] = None) -> None:
        """
        Create a configuration template file.
        
        Args:
            output_path: Where to save the configuration template
            real_data_path: Optional real data path to include
            synthetic_data_path: Optional synthetic data path to include
            purpose: Optional evaluation purpose to set
        """
        config = get_default_config()
        
        # Override with provided values
        if real_data_path:
            config["data"]["real"] = real_data_path
        if synthetic_data_path:
            config["data"]["synthetic"] = synthetic_data_path
        if purpose:
            config["evaluation"]["purpose"] = purpose
        
        # Merge with instance overrides
        if self._config_override:
            config = merge_configs(config, self._config_override)
        
        save_config(config, output_path)
    
    def list_available_metrics(self, family: Optional[str] = None) -> Dict[str, list]:
        """
        List available evaluation metrics.
        
        Args:
            family: Optional family filter (fidelity, utility, privacy)
            
        Returns:
            Dictionary mapping families to metric lists
        """
        from ..infrastructure.metrics.registry import get_metric_registry
        
        registry = get_metric_registry()
        
        if family:
            return {family: registry.list_ids(family)}
        else:
            return registry.list_by_family()
    
    def validate_config(self, config_path: str) -> Dict[str, Any]:
        """
        Validate configuration file and return status.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Validation status dictionary
        """
        try:
            from ..infrastructure.runtime.config import load_config, validate_file_paths
            
            # Try to load config
            config = load_config(config_path)
            
            # Check file paths
            path_warnings = validate_file_paths(config)
            
            return {
                "valid": True,
                "warnings": path_warnings,
                "config_sections": list(config.keys())
            }
            
        except Exception as e:
            return {
                "valid": False,
                "error": str(e),
                "warnings": []
            }
    
    @staticmethod
    def get_purpose_info() -> Dict[str, Dict[str, Any]]:
        """
        Get information about available evaluation purposes.
        
        Returns:
            Dictionary with purpose information
        """
        from ..domain.taxonomy import PURPOSES, PURPOSE_METRIC_DEFAULTS
        
        purpose_info = {}
        for purpose, families in PURPOSES.items():
            purpose_info[purpose] = {
                "families": list(families),
                "default_metrics": list(PURPOSE_METRIC_DEFAULTS.get(purpose, []))
            }
        
        return purpose_info
    
    def get_result_summary(self, run_summary: RunSummary) -> Dict[str, Any]:
        """
        Extract key information from RunSummary for easy consumption.
        
        Args:
            run_summary: RunSummary from evaluation
            
        Returns:
            Dictionary with key metrics and scores
        """
        summary = {
            "overall": {
                "total_metrics": len(run_summary.results),
                "successful": len([r for r in run_summary.results if "error" not in r.details]),
                "failed": len([r for r in run_summary.results if "error" in r.details])
            },
            "scores": {},
            "failed_metrics": []
        }
        
        # Extract family scores
        for family in ["fidelity", "utility", "privacy"]:
            if f"{family}_score" in run_summary.aggregates:
                summary["scores"][family] = run_summary.aggregates[f"{family}_score"]
        
        # Add composite score if available
        if "composite_score" in run_summary.aggregates:
            summary["scores"]["composite"] = run_summary.aggregates["composite_score"]
        
        # List failed metrics
        for result in run_summary.results:
            if "error" in result.details:
                summary["failed_metrics"].append({
                    "id": result.id,
                    "family": result.family,
                    "error": result.details["error"]
                })
        
        return summary