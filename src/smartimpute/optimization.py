"""
Hyperparameter optimization using Optuna for advanced imputation strategies.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
import optuna
from sklearn.metrics import mean_squared_error, f1_score

from smartimpute.exceptions import OptimizationError
from smartimpute.strategies.numeric import KNNNumericImputer, IterativeNumericImputer
from smartimpute.similarity import SimilarityBasedImputer

# Suppress Optuna chatty logs
optuna.logging.set_verbosity(optuna.logging.WARNING)


class StrategyOptimizer:
    """
    Optimizes hyperparameters for tunable imputation strategies using Optuna
    with internal cross-masked validation to strictly prevent data leakage.
    """

    def __init__(
        self,
        n_trials: int = 15,
        timeout: int = 45,
        random_state: int = 42
    ):
        self.n_trials = n_trials
        self.timeout = timeout
        self.random_state = random_state

    def optimize_knn(
        self,
        df: pd.DataFrame,
        target_column: str
    ) -> Dict[str, Any]:
        """Optimize KNN parameters (n_neighbors, weights) on target column."""
        observed_idx = df[df[target_column].notna()].index.to_numpy()
        if len(observed_idx) < 15:
            return {"n_neighbors": 5, "weights": "uniform"}

        # Mask 15% for evaluation
        rng = np.random.RandomState(self.random_state)
        n_mask = max(2, int(len(observed_idx) * 0.15))
        mask_idx = rng.choice(observed_idx, size=n_mask, replace=False)

        masked_df = df.copy()
        true_vals = df.loc[mask_idx, target_column].copy()
        masked_df.loc[mask_idx, target_column] = np.nan

        max_k = min(30, max(2, len(observed_idx) - n_mask - 1))

        def objective(trial: optuna.Trial) -> float:
            k = trial.suggest_int("n_neighbors", 2, max_k)
            w = trial.suggest_categorical("weights", ["uniform", "distance"])

            imputer = KNNNumericImputer(n_neighbors=k, weights=w, columns=[target_column])
            try:
                res = imputer.fit_transform(masked_df)
                preds = pd.to_numeric(res.loc[mask_idx, target_column], errors="coerce").fillna(true_vals.median())
                rmse = float(np.sqrt(mean_squared_error(true_vals, preds)))
                return rmse
            except Exception as e:
                raise optuna.TrialPruned() from e

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)

        best_params = study.best_params if len(study.trials) > 0 else {"n_neighbors": 5, "weights": "uniform"}
        return best_params

    def optimize_similarity(
        self,
        df: pd.DataFrame,
        target_column: str
    ) -> Dict[str, Any]:
        """Optimize Similarity parameters (n_neighbors, weights) on target column."""
        observed_idx = df[df[target_column].notna()].index.to_numpy()
        if len(observed_idx) < 15:
            return {"n_neighbors": 5, "weights": "distance"}

        rng = np.random.RandomState(self.random_state)
        n_mask = max(2, int(len(observed_idx) * 0.15))
        mask_idx = rng.choice(observed_idx, size=n_mask, replace=False)

        masked_df = df.copy()
        true_vals = df.loc[mask_idx, target_column].copy()
        masked_df.loc[mask_idx, target_column] = np.nan

        is_numeric = pd.api.types.is_numeric_dtype(df[target_column])
        max_k = min(20, max(2, len(observed_idx) - n_mask - 1))

        def objective(trial: optuna.Trial) -> float:
            k = trial.suggest_int("n_neighbors", 2, max_k)
            w = trial.suggest_categorical("weights", ["distance", "uniform"])

            imputer = SimilarityBasedImputer(n_neighbors=k, weights=w, columns=[target_column])
            try:
                res = imputer.fit_transform(masked_df)
                preds = res.loc[mask_idx, target_column]
                if is_numeric:
                    clean_preds = pd.to_numeric(preds, errors="coerce").fillna(true_vals.median())
                    return float(np.sqrt(mean_squared_error(true_vals, clean_preds)))
                else:
                    score = float(f1_score(true_vals.astype(str), preds.astype(str), average="macro", zero_division=0))
                    return -score  # Minimize negative F1
            except Exception as e:
                raise optuna.TrialPruned() from e

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)

        best_params = study.best_params if len(study.trials) > 0 else {"n_neighbors": 5, "weights": "distance"}
        return best_params
