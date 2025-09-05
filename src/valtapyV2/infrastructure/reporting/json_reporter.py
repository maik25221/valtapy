"""JSON reporter for structured evaluation results."""

import json
from pathlib import Path
from typing import Any, Dict
from ...domain.contracts import Reporter
from ...domain.entities import RunSummary, ReportSpec


class JSONReporter:
    """Reporter that generates JSON output for structured data consumption."""
    
    def render(self, run_summary: RunSummary, report_spec: ReportSpec) -> None:
        """Generate JSON report from run summary."""
        
        # Convert run summary to serializable format
        report_data = self._serialize_run_summary(run_summary, report_spec)
        
        # Write to file
        output_path = Path(report_spec.output_dir) / "summary.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
    
    def _serialize_run_summary(self, run_summary: RunSummary, report_spec: ReportSpec) -> Dict[str, Any]:
        """Convert RunSummary to JSON-serializable format."""
        
        # Serialize evaluation plan
        plan_data = {
            "metric_ids": run_summary.plan.metric_ids,
            "seed": run_summary.plan.seed,
            "cv_splits": run_summary.plan.cv_splits,
            "models": run_summary.plan.models,
            "purpose": run_summary.plan.purpose
        }
        
        # Serialize results
        results_data = []
        for result in run_summary.results:
            result_data = {
                "id": result.id,
                "value": result.value,
                "family": result.family,
                "purpose_tags": list(result.purpose_tags)
            }
            
            if report_spec.include_details:
                result_data["details"] = result.details
            
            results_data.append(result_data)
        
        # Serialize aggregates
        aggregates_data = dict(run_summary.aggregates)
        
        # Base report structure
        report_data = {
            "metadata": {
                "report_format": "json",
                "generated_by": "ValtaPyV2",
                "version": "0.1.0"
            },
            "plan": plan_data,
            "results": results_data,
            "aggregates": aggregates_data,
            "summary": {
                "total_metrics": len(run_summary.results),
                "successful_metrics": len([r for r in run_summary.results if "error" not in r.details]),
                "failed_metrics": len([r for r in run_summary.results if "error" in r.details])
            }
        }
        
        # Add family breakdown
        family_breakdown = {}
        for family in ["fidelity", "utility", "privacy"]:
            family_results = run_summary.get_results_by_family(family)
            family_breakdown[family] = {
                "count": len(family_results),
                "score": run_summary.aggregates.get(f"{family}_score", 0.0),
                "metrics": [r.id for r in family_results]
            }
        
        report_data["family_breakdown"] = family_breakdown
        
        # Add artifacts if requested
        if report_spec.include_artifacts and run_summary.artifacts:
            report_data["artifacts"] = run_summary.artifacts
        
        return report_data