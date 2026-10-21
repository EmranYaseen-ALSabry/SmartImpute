"""
Categorical imputation strategies implementation.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from smartimpute.exceptions import ImputationStrategyError
from smartimpute.strategies import BaseImputerStrategy


class ModeImputer(BaseImputerStrategy):
    """Imputes missing categorical values using the most frequent category (mode)."""

    def __init__(self, columns: Optional[List[str]] = None):
        super().__init__(columns=columns)
        self.modes_: Dict[str, Any] = {}

    def fit(self, X: pd.DataFrame, y=None) -> "ModeImputer":
        target_cols = self.columns or X.select_dtypes(exclude=[np.number]).columns.tolist()
        self.modes_ = {}
        for col in target_cols:
            if col in X.columns:
                mode_series = X[col].dropna().mode()
                self.modes_[col] = mode_series.iloc[0] if len(mode_series) > 0 else "missing"
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise ImputationStrategyError("ModeImputer is not fitted.")
        X_out = X.copy()
        for col, mode_val in self.modes_.items():
            if col in X_out.columns:
                X_out[col] = X_out[col].fillna(mode_val)
        return X_out


class ConstantCategoryImputer(BaseImputerStrategy):
    """Imputes missing categorical values with a dedicated explicit missing token (e.g., 'missing')."""

    def __init__(self, fill_value: str = "missing", columns: Optional[List[str]] = None):
        super().__init__(columns=columns)
        self.fill_value = fill_value

    def fit(self, X: pd.DataFrame, y=None) -> "ConstantCategoryImputer":
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        target_cols = self.columns or X.select_dtypes(exclude=[np.number]).columns.tolist()
        X_out = X.copy()
        for col in target_cols:
            if col in X_out.columns:
                X_out[col] = X_out[col].fillna(self.fill_value)
        return X_out


class GroupBasedCategoricalImputer(BaseImputerStrategy):
    """
    Imputes missing categorical values using the mode within each group defined by group_column.
    """

    def __init__(self, group_column: str, columns: Optional[List[str]] = None):
        super().__init__(columns=columns)
        self.group_column = group_column
        self.group_modes_: Dict[str, Dict[Any, Any]] = {}
        self.global_modes_: Dict[str, Any] = {}

    def fit(self, X: pd.DataFrame, y=None) -> "GroupBasedCategoricalImputer":
        target_cols = self.columns or X.select_dtypes(exclude=[np.number]).columns.tolist()
        for col in target_cols:
            if col == self.group_column or col not in X.columns:
                continue

            # Global mode
            glob_mode = X[col].dropna().mode()
            self.global_modes_[col] = glob_mode.iloc[0] if len(glob_mode) > 0 else "missing"

            if self.group_column in X.columns:
                # Group mode
                grp_dict = {}
                for grp, sub_df in X.groupby(self.group_column, observed=False):
                    m = sub_df[col].dropna().mode()
                    if len(m) > 0:
                        grp_dict[grp] = m.iloc[0]
                self.group_modes_[col] = grp_dict

        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise ImputationStrategyError("GroupBasedCategoricalImputer is not fitted.")
        X_out = X.copy()

        for col, grp_dict in self.group_modes_.items():
            if col in X_out.columns:
                missing_mask = X_out[col].isna()
                if missing_mask.any():
                    if self.group_column in X_out.columns:
                        group_series = X_out.loc[missing_mask, self.group_column]
                        imputed = group_series.map(grp_dict).fillna(self.global_modes_[col])
                        X_out.loc[missing_mask, col] = imputed
                    else:
                        X_out.loc[missing_mask, col] = self.global_modes_[col]

        return X_out


class ModelBasedCategoricalImputer(BaseImputerStrategy):
    """
    Model-based imputation for categorical columns using a Decision Tree Classifier.
    Trains on observed rows using other available features (numeric or frequency-encoded).
    """

    def __init__(self, max_depth: int = 5, columns: Optional[List[str]] = None):
        super().__init__(columns=columns)
        self.max_depth = max_depth
        self.models_: Dict[str, DecisionTreeClassifier] = {}
        self.global_modes_: Dict[str, Any] = {}
        self.numeric_features_: List[str] = []

    def fit(self, X: pd.DataFrame, y=None) -> "ModelBasedCategoricalImputer":
        target_cols = self.columns or X.select_dtypes(exclude=[np.number]).columns.tolist()
        self.numeric_features_ = X.select_dtypes(include=[np.number]).columns.tolist()

        for col in target_cols:
            if col not in X.columns:
                continue

            # Fallback mode
            m = X[col].dropna().mode()
            self.global_modes_[col] = m.iloc[0] if len(m) > 0 else "missing"

            # Check if we have numeric features and enough complete cases
            observed_mask = X[col].notna()
            if observed_mask.sum() >= 10 and len(self.numeric_features_) > 0:
                X_train = X.loc[observed_mask, self.numeric_features_].fillna(0)
                y_train = X.loc[observed_mask, col].astype(str)

                # Ensure at least 2 distinct classes
                if y_train.nunique() > 1:
                    clf = DecisionTreeClassifier(max_depth=self.max_depth, random_state=42)
                    clf.fit(X_train, y_train)
                    self.models_[col] = clf

        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise ImputationStrategyError("ModelBasedCategoricalImputer is not fitted.")
        X_out = X.copy()

        for col in self.columns or X.select_dtypes(exclude=[np.number]).columns.tolist():
            if col in X_out.columns:
                missing_mask = X_out[col].isna()
                if missing_mask.any():
                    if col in self.models_ and len(self.numeric_features_) > 0:
                        X_pred = X_out.loc[missing_mask, self.numeric_features_].fillna(0)
                        preds = self.models_[col].predict(X_pred)
                        X_out.loc[missing_mask, col] = preds
                    else:
                        X_out.loc[missing_mask, col] = self.global_modes_.get(col, "missing")

        return X_out
