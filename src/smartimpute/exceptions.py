"""
Custom exceptions for the SmartImpute system.
"""

class SmartImputeError(Exception):
    """Base exception for all SmartImpute errors."""
    pass


class DataLeakageError(SmartImputeError):
    """Raised when potential data leakage is detected (e.g., target used for feature imputation)."""
    pass


class InvalidDataError(SmartImputeError):
    """Raised when data input fails validation (empty data, unsupported types, etc.)."""
    pass


class ImputationStrategyError(SmartImputeError):
    """Raised when an imputation strategy fails to fit or transform."""
    pass


class ProfilingError(SmartImputeError):
    """Raised when profiling fails on a dataset."""
    pass


class OptimizationError(SmartImputeError):
    """Raised when Optuna hyperparameter optimization fails."""
    pass
