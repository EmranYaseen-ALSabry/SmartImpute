"""
Scientific evaluation engine for imputation strategies using Artificial Masking.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, accuracy_score, f1_score

from smartimpute.exceptions import SmartImputeError
from smartimpute.strategies.numeric import (
    MeanImputer,
    MedianImputer,
    KNNNumericImputer,
    IterativeNumericImputer,
)
from smartimpute.strategies.categorical import (
    ModeImputer,
    ConstantCategoryImputer,
    ModelBasedCategoricalImputer,
)
from smartimpute.similarity import SimilarityBasedImputer


class ImputationEvaluator:
    """
    Evaluates candidate imputation strategies scientifically by artificially masking
    known values and computing reconstruction errors across multiple seeds.
    """

    def __init__(
        self,
        mask_ratio: float = 0.15,
        n_seeds: int = 3,
        random_state: int = 42
    ):
        self.mask_ratio = max(0.01, min(0.40, mask_ratio))
        self.n_seeds = max(1, n_seeds)
        self.random_state = random_state

    def evaluate_column_strategies(
        self,
        df: pd.DataFrame,
        column: str,
        candidate_strategies: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate candidate strategies specifically for a target column via artificial masking.
        """
        if column not in df.columns:
            raise SmartImputeError(f"Column '{column}' not in DataFrame.")

        observed_idx = df[df[column].notna()].index.to_numpy()
        n_observed = len(observed_idx)
        if n_observed < 10:
            return {
                "column": column,
                "warning": "Insufficient observed samples (<10) for reliable artificial masking.",
                "strategies": {},
                "recommended_strategy": "fallback_baseline",
            }

        is_numeric = pd.api.types.is_numeric_dtype(df[column])

        # Default strategy pool if none specified
        if candidate_strategies is None:
            if is_numeric:
                candidate_strategies = ["median", "mean", "knn", "iterative", "similarity"]
            else:
                candidate_strategies = ["mode", "constant", "model_based", "similarity"]

        # Limit expensive strategies if dataset is large or has few features
        has_other_features = len(df.columns) > 1
        num_features = len(df.select_dtypes(include=[np.number]).columns)

        results: Dict[str, Dict[str, Any]] = {}

        n_mask = max(1, int(n_observed * self.mask_ratio))
        rng = np.random.RandomState(self.random_state)
        seeds = [int(rng.randint(0, 10000)) for _ in range(self.n_seeds)]

        for strat_name in candidate_strategies:
            scores: Dict[str, List[float]] = {}
            elapsed_times: List[float] = []

            for seed in seeds:
                sub_rng = np.random.RandomState(seed)
                mask_indices = sub_rng.choice(observed_idx, size=n_mask, replace=False)

                # Create masked copy
                df_masked = df.copy()
                true_values = df.loc[mask_indices, column].copy()
                df_masked.loc[mask_indices, column] = np.nan

                # Instantiate strategy
                imputer = self._build_imputer(strat_name, column, is_numeric, num_features)
                if imputer is None:
                    continue

                t0 = time.perf_counter()
                try:
                    imputed_df = imputer.fit_transform(df_masked)
                    elapsed = time.perf_counter() - t0
                    elapsed_times.append(elapsed)

                    predicted_values = imputed_df.loc[mask_indices, column]

                    if is_numeric:
                        # Handle potential remaining NaNs
                        pred_clean = pd.to_numeric(predicted_values, errors="coerce").fillna(true_values.median())
                        rmse = float(np.sqrt(mean_squared_error(true_values, pred_clean)))
                        mae = float(mean_absolute_error(true_values, pred_clean))
                        med_ae = float(median_absolute_error(true_values, pred_clean))

                        scores.setdefault("rmse", []).append(rmse)
                        scores.setdefault("mae", []).append(mae)
                        scores.setdefault("med_ae", []).append(med_ae)
                    else:
                        pred_clean = predicted_values.astype(str).fillna("missing")
                        true_clean = true_values.astype(str)
                        acc = float(accuracy_score(true_clean, pred_clean))
                        f1 = float(f1_score(true_clean, pred_clean, average="macro", zero_division=0))

                        scores.setdefault("accuracy", []).append(acc)
                        scores.setdefault("macro_f1", []).append(f1)

                except Exception as e:
                    # In case strategy failed for this seed
                    continue

            if scores:
                summary: Dict[str, Any] = {
                    "execution_time_sec": round(float(np.mean(elapsed_times)), 4) if elapsed_times else 0.0,
                    "metrics": {}
                }
                for m_name, vals in scores.items():
                    summary["metrics"][m_name] = {
                        "mean": round(float(np.mean(vals)), 4),
                        "std": round(float(np.std(vals)), 4),
                    }
                results[strat_name] = summary

        # Choose best candidate
        best_strat, justification = self._rank_strategies(results, is_numeric)

        return {
            "column": column,
            "is_numeric": is_numeric,
            "n_observed_tested": n_observed,
            "n_masked_per_seed": n_mask,
            "n_seeds": len(seeds),
            "strategies": results,
            "recommended_strategy": best_strat,
            "justification": justification,
        }

    def _build_imputer(
        self,
        strategy_name: str,
        column: str,
        is_numeric: bool,
        num_numeric_features: int
    ):
        name = strategy_name.lower()
        if is_numeric:
            if name == "mean":
                return MeanImputer(columns=[column])
            elif name == "median":
                return MedianImputer(columns=[column])
            elif name == "knn":
                return KNNNumericImputer(columns=[column], n_neighbors=5)
            elif name == "iterative":
                if num_numeric_features < 2:
                    return MedianImputer(columns=[column])
                return IterativeNumericImputer(columns=[column], max_iter=5)
            elif name == "similarity":
                return SimilarityBasedImputer(columns=[column], n_neighbors=5)
            else:
                return MedianImputer(columns=[column])
        else:
            if name == "mode":
                return ModeImputer(columns=[column])
            elif name == "constant":
                return ConstantCategoryImputer(fill_value="missing", columns=[column])
            elif name == "model_based":
                return ModelBasedCategoricalImputer(columns=[column], max_depth=4)
            elif name == "similarity":
                return SimilarityBasedImputer(columns=[column], n_neighbors=5)
            else:
                return ModeImputer(columns=[column])

    def _rank_strategies(
        self,
        results: Dict[str, Dict[str, Any]],
        is_numeric: bool
    ) -> Tuple[str, str]:
        if not results:
            return ("median" if is_numeric else "mode", "Fallback baseline: no strategies completed evaluation.")

        if is_numeric:
            # Lower RMSE is better; check stability (std) as tie breaker
            sorted_strats = sorted(
                results.items(),
                key=lambda x: (
                    x[1]["metrics"].get("rmse", {}).get("mean", float("inf")),
                    x[1]["metrics"].get("rmse", {}).get("std", float("inf")),
                )
            )
            best_name, best_data = sorted_strats[0]
            rmse_mean = best_data["metrics"].get("rmse", {}).get("mean", 0.0)
            justification = f"Lowest RMSE ({rmse_mean}) on artificial masking across seeds."
            return best_name, justification
        else:
            # Higher Macro F1 is better
            sorted_strats = sorted(
                results.items(),
                key=lambda x: (
                    -x[1]["metrics"].get("macro_f1", {}).get("mean", -1.0),
                    x[1]["metrics"].get("macro_f1", {}).get("std", float("inf")),
                )
            )
            best_name, best_data = sorted_strats[0]
            f1_mean = best_data["metrics"].get("macro_f1", {}).get("mean", 0.0)
            justification = f"Highest Macro-F1 ({f1_mean}) on artificial masking across seeds."
            return best_name, justification
