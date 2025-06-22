from datetime import datetime
from pathlib import Path
from typing import Optional, Union, Dict

from valtapy.results.validation_result import ValidationResult


class ResultSaver:
    """Utility class for saving validation results in different formats"""

    def __init__(self, base_directory: Union[str, Path] = "results"):
        self.base_directory = Path(base_directory)
        self.base_directory.mkdir(parents=True, exist_ok=True)

    def save_all_formats(
        self, results: ValidationResult, experiment_name: Optional[str] = None
    ) -> Dict[str, Path]:
        """Save results in all available formats"""
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"validation_{timestamp}"

        experiment_dir = self.base_directory / experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)

        saved_files = {}

        # Save as JSON
        json_path = experiment_dir / f"{experiment_name}.json"
        results.save_to_json(json_path)
        saved_files["json"] = json_path

        # Save as pickle
        pickle_path = experiment_dir / f"{experiment_name}.pkl"
        results.save_to_pickle(pickle_path)
        saved_files["pickle"] = pickle_path

        # Save as CSV
        csv_path = experiment_dir / f"{experiment_name}.csv"
        results.save_to_csv(csv_path)
        saved_files["csv"] = csv_path

        # Save summary report
        report_path = experiment_dir / f"{experiment_name}_report.txt"
        results.save_summary_report(report_path)
        saved_files["report"] = report_path

        return saved_files

    def save_format(
        self,
        results: ValidationResult,
        format_type: str,
        experiment_name: Optional[str] = None,
    ) -> Path:
        """Save results in specific format"""
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"validation_{timestamp}"

        experiment_dir = self.base_directory / experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)

        if format_type.lower() == "json":
            filepath = experiment_dir / f"{experiment_name}.json"
            results.save_to_json(filepath)
        elif format_type.lower() == "pickle":
            filepath = experiment_dir / f"{experiment_name}.pkl"
            results.save_to_pickle(filepath)
        elif format_type.lower() == "csv":
            filepath = experiment_dir / f"{experiment_name}.csv"
            results.save_to_csv(filepath)
        elif format_type.lower() == "report":
            filepath = experiment_dir / f"{experiment_name}_report.txt"
            results.save_summary_report(filepath)
        else:
            raise ValueError(f"Unsupported format: {format_type}")

        return filepath
