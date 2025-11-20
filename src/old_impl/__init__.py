"""ValtaPyV2: A modular framework for synthetic tabular data evaluation."""

__version__ = "0.1.0"

# Conditional import to handle missing dependencies gracefully
try:
    from .interface.sdk import Evaluator
    __all__ = ["Evaluator"]
except ImportError:
    # If dependencies are missing, provide basic access to domain layer
    __all__ = []