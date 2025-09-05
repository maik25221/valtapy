"""Result aggregation and scoring logic."""

from typing import List, Dict, Any, Optional
import statistics
from ..domain.entities import MetricResult, EvalPlan
from ..domain.taxonomy import FAMILIES


class Aggregator:
    """Aggregates metric results into family scores and composite indices."""
    
    def aggregate(self, results: List[MetricResult], plan: EvalPlan, 
                 weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Aggregate metric results into family scores and overall metrics.
        
        Args:
            results: List of computed metric results
            plan: Evaluation plan used for the run
            weights: Optional weights for family scoring (defaults to equal weighting)
            
        Returns:
            Dictionary containing family scores and aggregated metrics
        """
        aggregates = {}
        
        # Default equal weights if not provided
        if weights is None:
            active_families = self._get_active_families(results)
            weights = {family: 1.0 / len(active_families) for family in active_families}
        
        # Normalize weights to sum to 1.0
        weight_sum = sum(weights.values())
        if weight_sum > 0:
            weights = {k: v / weight_sum for k, v in weights.items()}
        
        # Calculate family scores
        family_scores = {}
        for family in FAMILIES:
            family_results = [r for r in results if r.family == family]
            if family_results:
                family_score = self._calculate_family_score(family_results, family)
                family_scores[family] = family_score
                aggregates[f"{family}_score"] = family_score
                aggregates[f"{family}_count"] = len(family_results)
        
        # Calculate composite index if requested
        composite_score = self._calculate_composite_score(family_scores, weights)
        if composite_score is not None:
            aggregates["composite_score"] = composite_score
        
        # Add summary statistics
        all_values = [r.value for r in results]
        if all_values:
            aggregates["mean_metric_value"] = statistics.mean(all_values)
            aggregates["median_metric_value"] = statistics.median(all_values)
            aggregates["std_metric_value"] = statistics.stdev(all_values) if len(all_values) > 1 else 0.0
        
        # Add execution metadata
        aggregates["total_metrics"] = len(results)
        aggregates["successful_metrics"] = len([r for r in results if "error" not in r.details])
        aggregates["failed_metrics"] = len([r for r in results if "error" in r.details])
        
        return aggregates
    
    def _get_active_families(self, results: List[MetricResult]) -> List[str]:
        """Get list of families that have results."""
        return list(set(r.family for r in results))
    
    def _calculate_family_score(self, family_results: List[MetricResult], family: str) -> float:
        """
        Calculate aggregated score for a metric family.
        
        Currently implements simple averaging. Future versions could implement
        more sophisticated normalization and weighting schemes.
        
        Args:
            family_results: Results for metrics in this family
            family: Family name for potential family-specific logic
            
        Returns:
            Aggregated score for the family (0.0 to 1.0)
        """
        if not family_results:
            return 0.0
        
        # Filter out failed metrics
        successful_results = [r for r in family_results if "error" not in r.details]
        if not successful_results:
            return 0.0
        
        # Simple averaging - could be enhanced with normalization
        values = [r.value for r in successful_results]
        
        # Family-specific score calculation
        if family == "fidelity":
            # For fidelity, higher is better (closer to real data)
            # Assume values are already normalized [0,1] or p-values
            return statistics.mean(values)
        
        elif family == "utility":
            # For utility, higher is better (better ML performance)
            # Assume values are already normalized [0,1]
            return statistics.mean(values)
        
        elif family == "privacy":
            # For privacy, interpretation depends on metric
            # For now, assume higher is better (more private)
            return statistics.mean(values)
        
        else:
            # Default: simple average
            return statistics.mean(values)
    
    def _calculate_composite_score(self, family_scores: Dict[str, float], 
                                 weights: Dict[str, float]) -> Optional[float]:
        """
        Calculate weighted composite score across families.
        
        Args:
            family_scores: Score for each family
            weights: Weight for each family
            
        Returns:
            Weighted composite score, or None if no valid scores
        """
        if not family_scores:
            return None
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for family, score in family_scores.items():
            weight = weights.get(family, 0.0)
            weighted_sum += score * weight
            total_weight += weight
        
        return weighted_sum if total_weight > 0 else None
    
    def get_family_summary(self, results: List[MetricResult], family: str) -> Dict[str, Any]:
        """
        Get detailed summary for a specific family.
        
        Args:
            results: All metric results
            family: Family to summarize
            
        Returns:
            Detailed summary dictionary for the family
        """
        family_results = [r for r in results if r.family == family]
        
        if not family_results:
            return {"count": 0, "score": 0.0, "metrics": []}
        
        successful = [r for r in family_results if "error" not in r.details]
        failed = [r for r in family_results if "error" in r.details]
        
        summary = {
            "count": len(family_results),
            "successful": len(successful),
            "failed": len(failed),
            "score": self._calculate_family_score(family_results, family),
            "metrics": [r.id for r in family_results]
        }
        
        if successful:
            values = [r.value for r in successful]
            summary["mean"] = statistics.mean(values)
            summary["median"] = statistics.median(values)
            summary["min"] = min(values)
            summary["max"] = max(values)
            summary["std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        
        return summary