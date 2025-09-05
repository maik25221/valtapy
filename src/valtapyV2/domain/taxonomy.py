"""Domain taxonomy defining metric families and evaluation purposes."""

from typing import Dict, Set

# Core metric families
FAMILIES: Set[str] = {"fidelity", "utility", "privacy"}

# Evaluation purposes mapping to relevant families
PURPOSES: Dict[str, Set[str]] = {
    "privacy_hardening": {"privacy", "fidelity"},
    "model_selection": {"utility", "fidelity"},
    "data_release": {"fidelity", "privacy", "utility"},
    "research_validation": {"fidelity", "utility"}
}

# Default metric IDs for each purpose (curated sets)
PURPOSE_METRIC_DEFAULTS: Dict[str, Set[str]] = {
    "privacy_hardening": {
        "fidelity.ks",
        "fidelity.correlation_delta", 
        "privacy.nndr",
        "privacy.membership_inference"
    },
    "model_selection": {
        "utility.pmse",
        "utility.accuracy",
        "fidelity.ks",
        "fidelity.mi_delta"
    },
    "data_release": {
        "fidelity.ks",
        "fidelity.correlation_delta",
        "utility.pmse",
        "privacy.nndr",
        "privacy.membership_inference"
    },
    "research_validation": {
        "fidelity.ks",
        "fidelity.correlation_delta",
        "fidelity.mi_delta",
        "utility.pmse"
    }
}


def get_default_metrics_for_purpose(purpose: str) -> Set[str]:
    """
    Get default metric IDs for a given evaluation purpose.
    
    The plan builder uses this function to translate purposes to metric_ids
    when no explicit metric_ids are provided in the configuration.
    
    Args:
        purpose: The evaluation purpose name
        
    Returns:
        Set of metric IDs suitable for the purpose
        
    Raises:
        ValueError: If purpose is not recognized
    """
    if purpose not in PURPOSES:
        raise ValueError(f"Unknown purpose: {purpose}. Available: {list(PURPOSES.keys())}")
    
    return PURPOSE_METRIC_DEFAULTS.get(purpose, set())


def get_families_for_purpose(purpose: str) -> Set[str]:
    """
    Get metric families relevant for a given evaluation purpose.
    
    Args:
        purpose: The evaluation purpose name
        
    Returns:
        Set of family names relevant for the purpose
        
    Raises:
        ValueError: If purpose is not recognized
    """
    if purpose not in PURPOSES:
        raise ValueError(f"Unknown purpose: {purpose}. Available: {list(PURPOSES.keys())}")
    
    return PURPOSES[purpose]