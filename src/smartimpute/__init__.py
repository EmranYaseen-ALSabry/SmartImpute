"""
SmartImpute: Intelligent Missing Data Imputation System
======================================================
An enterprise-grade, leakage-free imputation system with adaptive strategy selection,
artificial masking evaluation, and full Scikit-Learn pipeline compatibility.
"""

from smartimpute.pipeline import SmartImputer
from smartimpute.profiling import DataProfiler
from smartimpute.evaluation import ImputationEvaluator
from smartimpute.selector import AdaptiveImputationSelector
from smartimpute.similarity import SimilarityBasedImputer

__version__ = "0.1.0"
__all__ = [
    "SmartImputer",
    "DataProfiler",
    "ImputationEvaluator",
    "AdaptiveImputationSelector",
    "SimilarityBasedImputer",
]
