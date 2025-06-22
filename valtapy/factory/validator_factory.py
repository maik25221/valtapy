from valtapy.interfaces import (
    IValidator,
    IValidatorFactory,
    ValidationBranch,
    ValidationSubBranch,
)
from valtapy.validators.privacy import PrivacyValidator
from valtapy.validators.quality import QualityValidator
from valtapy.validators.utility.efficiency.ml_effiency.ml_efficiency import (
    MLEfficiencyValidator,
)
from valtapy.validators.utility import UtilityOrchestrator


class ValidatorFactory(IValidatorFactory):
    """Factory for creating validators based on branch or sub-branch type"""

    def create_validator(self, branch: ValidationBranch) -> IValidator:
        validators = {
            ValidationBranch.PRIVACY: PrivacyValidator,
            ValidationBranch.QUALITY: QualityValidator,
            ValidationBranch.UTILITY: UtilityOrchestrator,
        }

        validator_class = validators.get(branch)
        if not validator_class:
            raise ValueError(f"Unknown validation branch: {branch}")

        return validator_class()

    def create_subbranch_validator(self, subbranch: ValidationSubBranch) -> IValidator:
        subbranch_validators = {
            ValidationSubBranch.EFFICIENCY: MLEfficiencyValidator,
        }

        validator_class = subbranch_validators.get(subbranch)
        if not validator_class:
            raise ValueError(f"Unknown validation sub-branch: {subbranch}")

        return validator_class()
