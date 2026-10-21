"""
Base interfaces and registry for SmartImpute imputation strategies.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class BaseImputerStrategy(BaseEstimator, TransformerMixin, ABC):
    """
    Abstract Base Class for all individual imputation strategies.
    Ensures scikit-learn compatibility.
    """

    def __init__(self, columns: Optional[List[str]] = None):
        self.columns = columns
        self.fitted_ = False

    @abstractmethod
    def fit(self, X: pd.DataFrame, y=None) -> "BaseImputerStrategy":
        """Fit the imputer on observed values of X."""
        pass

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values in X without modifying known observed values."""
        pass

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Fit to X, then transform X."""
        return self.fit(X, y).transform(X)
