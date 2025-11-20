"""Registry for preprocessors."""

from typing import Dict, Type, List
from ...domain.contracts import Preprocessor
from ...domain.errors import RegistryError
from .transformers import TabularPreprocessor, IdentityPreprocessor, StandardScaler


class PreprocessorRegistryImpl:
    """Concrete implementation of preprocessor registry."""
    
    def __init__(self):
        self._preprocessors: Dict[str, Type[Preprocessor]] = {}
        self._register_defaults()
    
    def register(self, preprocessor_name: str, preprocessor_class: Type[Preprocessor]) -> None:
        """Register a preprocessor implementation."""
        if not preprocessor_name:
            raise RegistryError("Preprocessor name cannot be empty", "preprocessor")
        
        self._preprocessors[preprocessor_name] = preprocessor_class
    
    def get(self, preprocessor_name: str) -> Type[Preprocessor]:
        """Retrieve preprocessor class by name."""
        if preprocessor_name not in self._preprocessors:
            raise RegistryError(
                f"Preprocessor not found: {preprocessor_name}. Available: {list(self._preprocessors.keys())}",
                "preprocessor", 
                preprocessor_name
            )
        return self._preprocessors[preprocessor_name]
    
    def list_names(self) -> List[str]:
        """List available preprocessor names."""
        return list(self._preprocessors.keys())
    
    def _register_defaults(self) -> None:
        """Register default preprocessors."""
        self.register("tabular", TabularPreprocessor)
        self.register("identity", IdentityPreprocessor)
        self.register("standard_scaler", StandardScaler)


# Global registry instance
_preprocessor_registry = PreprocessorRegistryImpl()


def get_preprocessor_registry() -> PreprocessorRegistryImpl:
    """Get global preprocessor registry instance."""
    return _preprocessor_registry


def register_preprocessor(name: str, preprocessor_class: Type[Preprocessor]) -> None:
    """Register a preprocessor in the global registry."""
    _preprocessor_registry.register(name, preprocessor_class)