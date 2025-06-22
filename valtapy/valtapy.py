from pathlib import Path
from typing import List, Optional, Union, Dict

import pandas as pd

from valtapy.factory import ValidatorFactory
from valtapy.interfaces import IValidationResult, ValidationBranch, ValidationSubBranch
from valtapy.results.result_saver import ResultSaver
from valtapy.results.validation_result import ValidationResult


class Valtapy:
    """Main validation orchestrator following SOLID principles"""

    def __init__(self, results_directory: Union[str, Path] = "results"):
        self._validator_factory = ValidatorFactory()
        self._all_branches = list(ValidationBranch)
        self._result_saver = ResultSaver(results_directory)

    def validate(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        branches: Optional[
            List[Union[str, ValidationBranch, ValidationSubBranch]]
        ] = None,
        save_results: bool = False,
        experiment_name: Optional[str] = None,
        save_format: str = "all",
    ) -> IValidationResult:
        """
        Validate synthetic data against real data

        Args:
            real_data: DataFrame with real data
            synthetic_data: DataFrame with synthetic data
            branches: List of validation branches/sub-branches to run. If None, runs all branches.
                     Can contain: "Privacy", "Quality", "Utility", "Efficiency", etc.
            save_results: Whether to save results to files
            experiment_name: Name for the experiment (used in file naming)
            save_format: Format to save results ("json", "pickle", "csv", "report", "all")

        Returns:
            IValidationResult: Results of all validations
        """
        # Input validation
        self._validate_inputs(real_data, synthetic_data)

        # Determine branches and sub-branches to validate
        target_branches, target_subbranches = self._resolve_branches_and_subbranches(
            branches
        )

        # Execute validations
        result = ValidationResult()

        # Execute main branches
        for branch in target_branches:
            validator = self._validator_factory.create_validator(branch)
            validation_result = validator.validate(real_data, synthetic_data)
            result.add_result(branch, validation_result)

        # Execute specific sub-branches (if any)
        for subbranch in target_subbranches:
            validator = self._validator_factory.create_subbranch_validator(subbranch)
            validation_result = validator.validate(real_data, synthetic_data)

            # Add to results with a special key to distinguish from main branches
            branch_key = f"{subbranch.value}_standalone"
            result.add_result_with_key(branch_key, validation_result)

        # Save results if requested
        if save_results:
            if save_format.lower() == "all":
                saved_files = self._result_saver.save_all_formats(
                    result, experiment_name
                )
                print(f"Results saved in multiple formats:")
                for format_type, filepath in saved_files.items():
                    print(f"  {format_type}: {filepath}")
            else:
                filepath = self._result_saver.save_format(
                    result, save_format, experiment_name
                )
                print(f"Results saved as {save_format}: {filepath}")

        return result

    def save_results(
        self,
        results: IValidationResult,
        experiment_name: Optional[str] = None,
        save_format: str = "all",
    ) -> Union[Path, Dict[str, Path]]:
        """
        Save validation results to files

        Args:
            results: ValidationResult object to save
            experiment_name: Name for the experiment
            save_format: Format to save ("json", "pickle", "csv", "report", "all")

        Returns:
            Path or dict of paths where files were saved
        """
        if save_format.lower() == "all":
            return self._result_saver.save_all_formats(results, experiment_name)
        else:
            return self._result_saver.save_format(results, save_format, experiment_name)

    def _validate_inputs(
        self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame
    ) -> None:
        """Validate input parameters"""
        if not isinstance(real_data, pd.DataFrame):
            raise TypeError("real_data must be a pandas DataFrame")

        if not isinstance(synthetic_data, pd.DataFrame):
            raise TypeError("synthetic_data must be a pandas DataFrame")

        if real_data.empty:
            raise ValueError("real_data cannot be empty")

        if synthetic_data.empty:
            raise ValueError("synthetic_data cannot be empty")

    def _resolve_branches_and_subbranches(
        self,
        branches: Optional[List[Union[str, ValidationBranch, ValidationSubBranch]]],
    ) -> tuple[List[ValidationBranch], List[ValidationSubBranch]]:
        """Resolve and validate the branches and sub-branches to process"""
        if branches is None:
            return self._all_branches, []

        resolved_branches = []
        resolved_subbranches = []

        for branch in branches:
            if isinstance(branch, str):
                # Try to match with ValidationBranch first
                try:
                    resolved_branches.append(ValidationBranch(branch))
                    continue
                except ValueError:
                    pass

                # Try to match with ValidationSubBranch
                try:
                    resolved_subbranches.append(ValidationSubBranch(branch))
                    continue
                except ValueError:
                    pass

                raise ValueError(f"Invalid branch/sub-branch name: {branch}")

            elif isinstance(branch, ValidationBranch):
                resolved_branches.append(branch)
            elif isinstance(branch, ValidationSubBranch):
                resolved_subbranches.append(branch)
            else:
                raise TypeError(
                    f"Branch must be str, ValidationBranch, or ValidationSubBranch, got {type(branch)}"
                )

        return resolved_branches, resolved_subbranches
