"""
Time-series imputation strategies.
"""
from __future__ import annotations

from typing import List, Optional
import pandas as pd
import numpy as np

from smartimpute.exceptions import ImputationStrategyError
from smartimpute.strategies import BaseImputerStrategy


class TimeSeriesImputer(BaseImputerStrategy):
    """
    Imputation for temporal / ordered sequences.
    Supports forward-fill (causal), backward-fill, and linear interpolation.
    """

    def __init__(
        self,
        method: str = "ffill",
        allow_future: bool = False,
        columns: Optional[List[str]] = None
    ):
        super().__init__(columns=columns)
        self.method = method.lower()
        self.allow_future = allow_future
        self.fallback_values_: dict = {}

    def fit(self, X: pd.DataFrame, y=None) -> "TimeSeriesImputer":
        target_cols = self.columns or X.columns.tolist()
        self.fallback_values_ = {}
        for col in target_cols:
            if col in X.columns:
                if pd.api.types.is_numeric_dtype(X[col]):
                    self.fallback_values_[col] = float(X[col].median()) if X[col].dropna().shape[0] > 0 else 0.0
                else:
                    mode_val = X[col].dropna().mode()
                    self.fallback_values_[col] = mode_val.iloc[0] if len(mode_val) > 0 else "missing"
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise ImputationStrategyError("TimeSeriesImputer is not fitted.")

        X_out = X.copy()
        target_cols = self.columns or X.columns.tolist()

        for col in target_cols:
            if col not in X_out.columns:
                continue

            if self.method == "ffill":
                # Strict forward fill (preserves causality, no future leakage)
                X_out[col] = X_out[col].ffill()
                if self.allow_future:
                    X_out[col] = X_out[col].bfill()
            elif self.method == "bfill":
                if not self.allow_future:
                    raise ImputationStrategyError(
                        "Backward fill (bfill) uses future data and is forbidden when allow_future=False to prevent look-ahead bias."
                    )
                X_out[col] = X_out[col].bfill()
            elif self.method in ["interpolate", "linear_interpolation"]:
                if pd.api.types.is_numeric_dtype(X_out[col]):
                    X_out[col] = X_out[col].interpolate(method="linear", limit_direction="forward" if not self.allow_future else "both")
                else:
                    X_out[col] = X_out[col].ffill()
            else:
                X_out[col] = X_out[col].ffill()

            # Final safety fallback for leading NaNs
            X_out[col] = X_out[col].fillna(self.fallback_values_.get(col, 0))

        return X_out
