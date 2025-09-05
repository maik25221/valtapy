"""Plan builder for constructing evaluation plans from configuration."""

from typing import Dict, Any, List, Optional
from ..domain.entities import EvalPlan
from ..domain.taxonomy import get_default_metrics_for_purpose, PURPOSES
from ..domain.errors import ConfigError


class PlanBuilder:
    """Builds EvalPlan from configuration following Builder pattern."""
    
    def build_from_config(self, config: Dict[str, Any]) -> EvalPlan:
        """
        Build evaluation plan from configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            EvalPlan configured according to the specification
            
        Raises:
            ConfigError: If configuration is invalid
        """
        try:
            eval_config = config.get("evaluation", {})
            
            # Extract basic parameters
            seed = eval_config.get("seed", 42)
            cv_splits = eval_config.get("cv_splits", 3)
            models = eval_config.get("models")
            purpose = eval_config.get("purpose")
            
            # Determine metric IDs
            metric_ids = self._resolve_metric_ids(eval_config, purpose)
            
            return EvalPlan(
                metric_ids=metric_ids,
                seed=seed,
                cv_splits=cv_splits,
                models=models,
                purpose=purpose
            )
            
        except KeyError as e:
            raise ConfigError(f"Missing required configuration key: {e}")
        except Exception as e:
            raise ConfigError(f"Failed to build evaluation plan: {e}")
    
    def _resolve_metric_ids(self, eval_config: Dict[str, Any], purpose: Optional[str]) -> List[str]:
        """
        Resolve metric IDs from configuration.
        
        Priority:
        1. Explicit metric_ids in config
        2. Default metrics for specified purpose
        3. Raise error if neither provided
        
        Args:
            eval_config: Evaluation section of configuration
            purpose: Evaluation purpose (optional)
            
        Returns:
            List of metric IDs to execute
            
        Raises:
            ConfigError: If no metrics can be resolved
        """
        # Check for explicit metric IDs first
        explicit_metrics = eval_config.get("metric_ids")
        if explicit_metrics:
            if not isinstance(explicit_metrics, list):
                raise ConfigError("metric_ids must be a list")
            return explicit_metrics
        
        # Fall back to purpose-based defaults
        if purpose:
            if purpose not in PURPOSES:
                raise ConfigError(f"Unknown purpose: {purpose}. Available: {list(PURPOSES.keys())}")
            
            default_metrics = get_default_metrics_for_purpose(purpose)
            if not default_metrics:
                raise ConfigError(f"No default metrics defined for purpose: {purpose}")
            
            return list(default_metrics)
        
        # No metrics specified
        raise ConfigError("Either 'metric_ids' or 'purpose' must be specified in evaluation config")
    
    def validate_plan(self, plan: EvalPlan) -> List[str]:
        """
        Validate evaluation plan and return list of warnings.
        
        Args:
            plan: The evaluation plan to validate
            
        Returns:
            List of validation warnings (empty if all good)
        """
        warnings = []
        
        # Check for reasonable number of metrics
        if len(plan.metric_ids) > 20:
            warnings.append(f"Large number of metrics ({len(plan.metric_ids)}) may impact performance")
        
        # Check CV splits
        if plan.cv_splits > 10:
            warnings.append(f"High number of CV splits ({plan.cv_splits}) may be slow")
        
        # Check for metric family coverage if purpose is specified
        if plan.purpose and plan.purpose in PURPOSES:
            expected_families = PURPOSES[plan.purpose]
            actual_families = set()
            
            for metric_id in plan.metric_ids:
                if "." in metric_id:
                    family = metric_id.split(".")[0]
                    actual_families.add(family)
            
            missing_families = expected_families - actual_families
            if missing_families:
                warnings.append(f"Purpose '{plan.purpose}' expects families {expected_families}, "
                               f"but missing: {missing_families}")
        
        return warnings