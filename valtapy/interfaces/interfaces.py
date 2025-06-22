from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Union

import pandas as pd


class ValidationBranch(Enum):
    PRIVACY = "Privacy"
    QUALITY = "Quality"
    UTILITY = "Utility"


class ValidationSubBranch(Enum):
    # Utility sub-branches
    EFFICIENCY = "Efficiency"

    # Privacy sub-branches
    INFERENCE = "Inference"

    # Quality sub-branches (se pueden agregar más en el futuro)
    # COMPLETENESS = "Completeness"
    # CONSISTENCY = "Consistency"

    @classmethod
    def get_subbranches_for_branch(
        cls, branch: ValidationBranch
    ) -> List["ValidationSubBranch"]:
        """Get all sub-branches for a given branch"""
        mapping = {
            ValidationBranch.UTILITY: [cls.EFFICIENCY],
            ValidationBranch.PRIVACY: [cls.INFERENCE],
            ValidationBranch.QUALITY: [],  # Agregar sub-ramas cuando las implementes
        }
        return mapping.get(branch, [])


class IValidator(ABC):
    """Interface for validation strategies"""

    @abstractmethod
    def validate(
        self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Execute validation and return results"""
        pass

    @abstractmethod
    def get_branch_name(self) -> ValidationBranch:
        """Get the validation branch this validator belongs to"""
        pass


class IValidationResult(ABC):
    """Interface for validation results"""

    @abstractmethod
    def add_result(self, branch: ValidationBranch, result: Dict[str, Any]) -> None:
        """Add validation result for a specific branch"""
        pass

    @abstractmethod
    def add_result_with_key(self, key: str, result: Dict[str, Any]) -> None:
        """Add validation result with a custom key"""
        pass

    @abstractmethod
    def get_results(self) -> Dict[Union[ValidationBranch, str], Dict[str, Any]]:
        """Get all validation results"""
        pass

    @abstractmethod
    def get_result_by_branch(
        self, branch: Union[ValidationBranch, str]
    ) -> Dict[str, Any]:
        """Get validation result for a specific branch or key"""
        pass


class IValidatorFactory(ABC):
    """Factory interface for creating validators"""

    @abstractmethod
    def create_validator(self, branch: ValidationBranch) -> IValidator:
        """Create a validator for a main branch"""
        pass

    @abstractmethod
    def create_subbranch_validator(self, subbranch: ValidationSubBranch) -> IValidator:
        """Create a validator for a sub-branch"""
        pass


class ISubValidator(ABC):
    """Interface for sub-branch validators (like inference techniques)"""

    @abstractmethod
    def validate_technique(
        self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Execute specific validation technique"""
        pass

    @abstractmethod
    def get_technique_name(self) -> str:
        """Get the name of this validation technique"""
        pass
