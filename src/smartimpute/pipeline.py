"""
SmartImputer: Production-grade Scikit-Learn transformer with adaptive column-wise imputation,
zero data leakage, and full serialization support.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from smartimpute.exceptions import DataLeakageError, InvalidDataError, SmartImputeError
from smartimpute.profiling import DataProfiler
from smartimpute.missingness import MissingnessAnalyzer
from smartimpute.selector import AdaptiveImputationSelector
from smartimpute.strategies.numeric import (
    MeanImputer,
    MedianImputer,
    ConstantNumericImputer,
    KNNNumericImputer,
    IterativeNumericImputer,
)
from smartimpute.strategies.categorical import (
    ModeImputer,
    ConstantCategoryImputer,
    ModelBasedCategoricalImputer,
)
from smartimpute.similarity import SimilarityBasedImputer


class SmartImputer(BaseEstimator, TransformerMixin):
    """
    Main entry point for SmartImpute.
    Fits customized imputation models for each column adaptively, guarantees zero leakage,
    and supports saving/loading for inference.
    """

    def __init__(
        self,
        target_column: Optional[str] = None,
        mode: str = "general",  # 'general' or 'ml_pipeline'
        add_indicators: bool = False,
        indicator_threshold: float = 0.05,
        use_optimization: bool = False,
        optuna_trials: int = 10,
        random_state: int = 42
    ):
        self.target_column = target_column
        self.mode = mode.lower()
        self.add_indicators = add_indicators
        self.indicator_threshold = indicator_threshold
        self.use_optimization = use_optimization
        self.optuna_trials = optuna_trials
        self.random_state = random_state

        # Internal state
        self.fitted_ = False
        self.expected_columns_: List[str] = []
        self.column_dtypes_: Dict[str, str] = {}
        self.strategies_map_: Dict[str, Dict[str, Any]] = {}
        self.fitted_imputers_: Dict[str, Any] = {}
        self.indicator_cols_: List[str] = []
        self.profile_summary_: Dict[str, Any] = {}
        self.missingness_summary_: Dict[str, Any] = {}

    def fit(self, X: pd.DataFrame, y=None) -> "SmartImputer":
        """
        Fit adaptive imputation strategies across columns.
        Strictly excludes target column from feature imputation to prevent data leakage.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if X.empty:
            raise InvalidDataError("Cannot fit SmartImputer on an empty dataset.")

        # Data Leakage checks
        if self.mode == "ml_pipeline" and self.target_column:
            if self.target_column not in X.columns and (y is None):
                raise DataLeakageError(f"Target column '{self.target_column}' specified but not found in data or y.")

        # Record expected columns
        self.expected_columns_ = list(X.columns)
        self.column_dtypes_ = {c: str(X[c].dtype) for c in X.columns}

        # 1. Profile data
        profiler = DataProfiler(target_column=self.target_column)
        self.profile_summary_ = profiler.profile(X)

        # 2. Analyze missingness
        analyzer = MissingnessAnalyzer()
        self.missingness_summary_ = analyzer.analyze(X)

        # 3. Determine missingness indicators to add
        self.indicator_cols_ = []
        if self.add_indicators:
            for col in X.columns:
                if col == self.target_column:
                    continue
                missing_ratio = float(X[col].isna().mean())
                if missing_ratio >= self.indicator_threshold:
                    self.indicator_cols_.append(col)

        # 4. Adaptive Strategy Selection
        selector = AdaptiveImputationSelector(
            use_optimization=self.use_optimization,
            optuna_trials=self.optuna_trials,
            random_state=self.random_state
        )
        self.strategies_map_ = selector.select_strategies(X, target_column=self.target_column)

        # 5. Fit individual column imputers
        self.fitted_imputers_ = {}
        for col, decision in self.strategies_map_.items():
            strat_name = decision["strategy"]
            params = decision.get("params", {})
            is_numeric = pd.api.types.is_numeric_dtype(X[col])

            imputer_instance = self._instantiate_imputer(strat_name, col, is_numeric, params)
            imputer_instance.fit(X)
            self.fitted_imputers_[col] = imputer_instance

        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values using the fitted models without altering existing values.
        """
        if not self.fitted_:
            raise SmartImputeError("SmartImputer is not fitted. Call .fit() first.")

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X_out = X.copy()

        # Check for unexpected missing columns
        for c in self.expected_columns_:
            if c not in X_out.columns:
                raise InvalidDataError(f"Column '{c}' was present during training but missing in transform data.")

        # 1. Add missingness indicators if enabled (computed BEFORE imputing values)
        if self.add_indicators:
            for c in self.indicator_cols_:
                if c in X_out.columns:
                    X_out[f"{c}_isna"] = X_out[c].isna().astype(int)

        # 2. Apply fitted imputers per column
        for col, imputer in self.fitted_imputers_.items():
            if col in X_out.columns:
                orig_missing_mask = X_out[col].isna()
                if orig_missing_mask.any():
                    imputed_res = imputer.transform(X_out)
                    # Strictly only replace NaNs, preserve observed
                    X_out.loc[orig_missing_mask, col] = imputed_res.loc[orig_missing_mask, col]

        return X_out

    def _instantiate_imputer(
        self,
        strategy_name: str,
        col: str,
        is_numeric: bool,
        params: Dict[str, Any]
    ):
        name = strategy_name.lower()
        if is_numeric:
            if name == "mean":
                return MeanImputer(columns=[col])
            elif name == "median":
                return MedianImputer(columns=[col])
            elif name == "constant":
                return ConstantNumericImputer(fill_value=params.get("fill_value", 0.0), columns=[col])
            elif name == "knn":
                k = params.get("n_neighbors", 5)
                w = params.get("weights", "uniform")
                return KNNNumericImputer(columns=[col], n_neighbors=k, weights=w)
            elif name == "iterative":
                return IterativeNumericImputer(columns=[col], max_iter=params.get("max_iter", 10))
            elif name == "similarity":
                k = params.get("n_neighbors", 5)
                w = params.get("weights", "distance")
                return SimilarityBasedImputer(columns=[col], n_neighbors=k, weights=w)
            else:
                return MedianImputer(columns=[col])
        else:
            if name == "mode":
                return ModeImputer(columns=[col])
            elif name == "constant":
                return ConstantCategoryImputer(fill_value=params.get("fill_value", "missing"), columns=[col])
            elif name == "model_based":
                return ModelBasedCategoricalImputer(columns=[col], max_depth=params.get("max_depth", 5))
            elif name == "similarity":
                k = params.get("n_neighbors", 5)
                w = params.get("weights", "distance")
                return SimilarityBasedImputer(columns=[col], n_neighbors=k, weights=w)
            else:
                return ModeImputer(columns=[col])

    def save(self, file_path: Union[str, Path]) -> None:
        """Serialize the entire fitted SmartImputer pipeline to disk."""
        if not self.fitted_:
            raise SmartImputeError("Cannot save an unfitted SmartImputer.")
        p = Path(file_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, p)

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> "SmartImputer":
        """Load a serialized SmartImputer pipeline from disk."""
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"Pipeline file not found: {p}")
        obj = joblib.load(p)
        if not isinstance(obj, cls):
            raise SmartImputeError(f"Loaded object is not a valid {cls.__name__}.")
        return obj
