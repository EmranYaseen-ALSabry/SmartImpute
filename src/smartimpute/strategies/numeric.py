"""
Numeric imputation strategies implementation.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
# IterativeImputer is experimental in scikit-learn:
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.preprocessing import MinMaxScaler

from smartimpute.exceptions import ImputationStrategyError
from smartimpute.strategies import BaseImputerStrategy


class MeanImputer(BaseImputerStrategy):
    """Imputes missing numeric values using column mean."""

    def __init__(self, columns: Optional[List[str]] = None):
        super().__init__(columns=columns)
        self.means_: Dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y=None) -> "MeanImputer":
        target_cols = self.columns or X.select_dtypes(include=[np.number]).columns.tolist()
        self.means_ = {}
        for col in target_cols:
            if col in X.columns:
                valid_vals = X[col].dropna()
                self.means_[col] = float(valid_vals.mean()) if len(valid_vals) > 0 else 0.0
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise ImputationStrategyError("MeanImputer is not fitted.")
        X_out = X.copy()
        for col, mean_val in self.means_.items():
            if col in X_out.columns:
                X_out[col] = X_out[col].fillna(mean_val)
        return X_out


class MedianImputer(BaseImputerStrategy):
    """Imputes missing numeric values using column median (robust to outliers)."""

    def __init__(self, columns: Optional[List[str]] = None):
        super().__init__(columns=columns)
        self.medians_: Dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y=None) -> "MedianImputer":
        target_cols = self.columns or X.select_dtypes(include=[np.number]).columns.tolist()
        self.medians_ = {}
        for col in target_cols:
            if col in X.columns:
                valid_vals = X[col].dropna()
                self.medians_[col] = float(valid_vals.median()) if len(valid_vals) > 0 else 0.0
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise ImputationStrategyError("MedianImputer is not fitted.")
        X_out = X.copy()
        for col, med_val in self.medians_.items():
            if col in X_out.columns:
                X_out[col] = X_out[col].fillna(med_val)
        return X_out


class ConstantNumericImputer(BaseImputerStrategy):
    """Imputes missing numeric values with a fixed constant value (e.g., 0 or -1)."""

    def __init__(self, fill_value: float = 0.0, columns: Optional[List[str]] = None):
        super().__init__(columns=columns)
        self.fill_value = fill_value

    def fit(self, X: pd.DataFrame, y=None) -> "ConstantNumericImputer":
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        target_cols = self.columns or X.select_dtypes(include=[np.number]).columns.tolist()
        X_out = X.copy()
        for col in target_cols:
            if col in X_out.columns:
                X_out[col] = X_out[col].fillna(self.fill_value)
        return X_out


class KNNNumericImputer(BaseImputerStrategy):
    """
    K-Nearest Neighbors Imputation for numeric features with internal Min-Max scaling
    to prevent features with large scales from dominating distance calculation.
    Uses all available numeric features to compute distances, imputing the requested columns.
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        weights: str = "uniform",
        columns: Optional[List[str]] = None
    ):
        super().__init__(columns=columns)
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.scaler_ = MinMaxScaler()
        self.imputer_ = None
        self.all_numeric_cols_: List[str] = []
        self.target_cols_: List[str] = []

    def fit(self, X: pd.DataFrame, y=None) -> "KNNNumericImputer":
        self.all_numeric_cols_ = X.select_dtypes(include=[np.number]).columns.tolist()
        self.target_cols_ = self.columns or self.all_numeric_cols_

        if not self.all_numeric_cols_:
            self.fitted_ = True
            return self

        num_data = X[self.all_numeric_cols_]
        # Fit scaler on non-null portions (or min/max per column)
        self.scaler_.fit(num_data.fillna(num_data.median()))
        scaled_data = self.scaler_.transform(num_data)

        # Ensure n_neighbors does not exceed available rows
        n_samples = max(1, len(X))
        actual_k = min(self.n_neighbors, max(1, n_samples - 1))

        self.imputer_ = KNNImputer(n_neighbors=actual_k, weights=self.weights)
        self.imputer_.fit(scaled_data)
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise ImputationStrategyError("KNNNumericImputer is not fitted.")
        if not self.all_numeric_cols_ or self.imputer_ is None:
            return X.copy()

        X_out = X.copy()
        num_data = X_out[self.all_numeric_cols_]
        scaled = self.scaler_.transform(num_data.fillna(num_data.median()))
        # Restore actual NaNs in scaled matrix so KNNImputer knows what to impute
        scaled[num_data.isna().values] = np.nan

        imputed_scaled = self.imputer_.transform(scaled)
        imputed_unscaled = self.scaler_.inverse_transform(imputed_scaled)

        # Only impute target_cols_ and strictly preserve known values
        for idx, col in enumerate(self.all_numeric_cols_):
            if col in self.target_cols_ and col in X_out.columns:
                mask = num_data[col].isna()
                if mask.any():
                    X_out.loc[mask, col] = imputed_unscaled[mask, idx]

        return X_out


class IterativeNumericImputer(BaseImputerStrategy):
    """
    Multivariate Imputation by Chained Equations (MICE) using BayesianRidge.
    Uses all available numeric features to build iterative regression models.
    """

    def __init__(
        self,
        max_iter: int = 10,
        random_state: int = 42,
        columns: Optional[List[str]] = None
    ):
        super().__init__(columns=columns)
        self.max_iter = max_iter
        self.random_state = random_state
        self.imputer_ = None
        self.all_numeric_cols_: List[str] = []
        self.target_cols_: List[str] = []

    def fit(self, X: pd.DataFrame, y=None) -> "IterativeNumericImputer":
        self.all_numeric_cols_ = X.select_dtypes(include=[np.number]).columns.tolist()
        self.target_cols_ = self.columns or self.all_numeric_cols_

        if not self.all_numeric_cols_:
            self.fitted_ = True
            return self

        estimator = BayesianRidge()
        self.imputer_ = IterativeImputer(
            estimator=estimator,
            max_iter=self.max_iter,
            random_state=self.random_state,
            initial_strategy="median"
        )
        self.imputer_.fit(X[self.all_numeric_cols_])
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise ImputationStrategyError("IterativeNumericImputer is not fitted.")
        if not self.all_numeric_cols_ or self.imputer_ is None:
            return X.copy()

        X_out = X.copy()
        orig_cols_data = X_out[self.all_numeric_cols_]
        imputed_vals = self.imputer_.transform(orig_cols_data)

        # Only impute target_cols_ and preserve known values strictly
        for idx, col in enumerate(self.all_numeric_cols_):
            if col in self.target_cols_ and col in X_out.columns:
                mask = orig_cols_data[col].isna()
                if mask.any():
                    X_out.loc[mask, col] = imputed_vals[mask, idx]

        return X_out


class GroupBasedNumericImputer(BaseImputerStrategy):
    """
    Imputes numeric values based on the median/mean of a grouping categorical column.
    Falls back to global median if group is missing or unseen.
    """

    def __init__(
        self,
        group_column: str,
        strategy: str = "median",
        columns: Optional[List[str]] = None
    ):
        super().__init__(columns=columns)
        self.group_column = group_column
        self.strategy = strategy
        self.group_map_: Dict[str, Dict[Any, float]] = {}
        self.global_fallback_: Dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y=None) -> "GroupBasedNumericImputer":
        target_cols = self.columns or X.select_dtypes(include=[np.number]).columns.tolist()
        if self.group_column not in X.columns:
            # Fallback to standard median
            med_imp = MedianImputer(columns=target_cols).fit(X)
            self.global_fallback_ = med_imp.medians_
            self.fitted_ = True
            return self

        for col in target_cols:
            if col == self.group_column or col not in X.columns:
                continue
            if self.strategy == "mean":
                grp_vals = X.groupby(self.group_column, observed=False)[col].mean().to_dict()
                fallback = float(X[col].mean()) if X[col].dropna().shape[0] > 0 else 0.0
            else:
                grp_vals = X.groupby(self.group_column, observed=False)[col].median().to_dict()
                fallback = float(X[col].median()) if X[col].dropna().shape[0] > 0 else 0.0

            self.group_map_[col] = grp_vals
            self.global_fallback_[col] = fallback

        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise ImputationStrategyError("GroupBasedNumericImputer is not fitted.")
        X_out = X.copy()
        if self.group_column not in X_out.columns:
            for col, fallback in self.global_fallback_.items():
                if col in X_out.columns:
                    X_out[col] = X_out[col].fillna(fallback)
            return X_out

        for col, grp_dict in self.group_map_.items():
            if col in X_out.columns:
                missing_mask = X_out[col].isna()
                if missing_mask.any():
                    # Map group value
                    group_series = X_out.loc[missing_mask, self.group_column]
                    imputed = group_series.map(grp_dict).fillna(self.global_fallback_[col])
                    X_out.loc[missing_mask, col] = imputed

        return X_out
