import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Union

import pandas as pd

from valtapy.interfaces import IValidationResult, ValidationBranch


class ValidationResult(IValidationResult):
    """Concrete implementation of validation results with file saving capabilities"""

    def __init__(self):
        self._results: Dict[Union[ValidationBranch, str], Dict[str, Any]] = {}
        self._metadata = {"created_at": datetime.now().isoformat(), "version": "1.0.0"}

    def add_result(self, branch: ValidationBranch, result: Dict[str, Any]) -> None:
        self._results[branch] = result

    def add_result_with_key(self, key: str, result: Dict[str, Any]) -> None:
        """Add validation result with a custom key (for sub-branches)"""
        self._results[key] = result

    def get_results(self) -> Dict[Union[ValidationBranch, str], Dict[str, Any]]:
        return self._results.copy()

    def get_result_by_branch(
        self, branch: Union[ValidationBranch, str]
    ) -> Dict[str, Any]:
        return self._results.get(branch, {})

    def save_to_json(self, filepath: Union[str, Path]) -> None:
        """Save results to JSON file"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Convert enum keys to strings for JSON serialization
        serializable_results = {}
        for key, value in self._results.items():
            if isinstance(key, ValidationBranch):
                serializable_results[key.value] = value
            else:
                serializable_results[key] = value

        output_data = {"metadata": self._metadata, "results": serializable_results}

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)

    def save_to_pickle(self, filepath: Union[str, Path]) -> None:
        """Save results to pickle file (preserves object types)"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        output_data = {"metadata": self._metadata, "results": self._results}

        with open(filepath, "wb") as f:
            pickle.dump(output_data, f)

    def save_to_csv(self, filepath: Union[str, Path]) -> None:
        """Save results to CSV file (flattened structure)"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Flatten results for CSV format
        flattened_data = []

        for branch_key, branch_results in self._results.items():
            branch_name = (
                branch_key.value
                if isinstance(branch_key, ValidationBranch)
                else branch_key
            )

            if isinstance(branch_results, dict):
                self._flatten_dict(branch_results, flattened_data, branch_name)

        if flattened_data:
            df = pd.DataFrame(flattened_data)
            df.to_csv(filepath, index=False)

    def save_summary_report(self, filepath: Union[str, Path]) -> None:
        """Save a human-readable summary report"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("VALTAPY VALIDATION REPORT")
        report_lines.append("=" * 60)
        report_lines.append(f"Generated: {self._metadata['created_at']}")
        report_lines.append(f"Version: {self._metadata['version']}")
        report_lines.append("")

        for branch_key, branch_results in self._results.items():
            branch_name = (
                branch_key.value
                if isinstance(branch_key, ValidationBranch)
                else branch_key
            )
            report_lines.append(f"{'='*20} {branch_name.upper()} {'='*20}")

            if isinstance(branch_results, dict):
                self._format_results_for_report(branch_results, report_lines, level=0)

            report_lines.append("")

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))

    def _flatten_dict(
        self, data: Dict[str, Any], output_list: list, prefix: str = ""
    ) -> None:
        """Helper method to flatten nested dictionaries for CSV export"""
        for key, value in data.items():
            new_key = f"{prefix}.{key}" if prefix else key

            if isinstance(value, dict):
                self._flatten_dict(value, output_list, new_key)
            else:
                output_list.append(
                    {
                        "branch": prefix.split(".")[0] if "." in prefix else prefix,
                        "metric": new_key,
                        "value": value,
                    }
                )

    def _format_results_for_report(
        self, data: Dict[str, Any], lines: list, level: int = 0
    ) -> None:
        """Helper method to format results for human-readable report"""
        indent = "  " * level

        for key, value in data.items():
            if isinstance(value, dict):
                lines.append(f"{indent}{key}:")
                self._format_results_for_report(value, lines, level + 1)
            else:
                if isinstance(value, float):
                    formatted_value = f"{value:.4f}"
                else:
                    formatted_value = str(value)
                lines.append(f"{indent}{key}: {formatted_value}")
