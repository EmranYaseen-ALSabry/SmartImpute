"""
Adaptive Strategy Selector: Chooses optimal, verified imputation strategies per column.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from smartimpute.evaluation import ImputationEvaluator
from smartimpute.optimization import StrategyOptimizer
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


class AdaptiveImputationSelector:
    """
    Selects the best imputation strategy for each column individually based on data characteristics,
    distribution skewness, artificial masking benchmark scores, and computation limits.
    """

    def __init__(
        self,
        evaluator: Optional[ImputationEvaluator] = None,
        use_optimization: bool = False,
        optuna_trials: int = 10,
        random_state: int = 42
    ):
        self.evaluator = evaluator or ImputationEvaluator(mask_ratio=0.15, n_seeds=2, random_state=random_state)
        self.use_optimization = use_optimization
        self.optimizer = StrategyOptimizer(n_trials=optuna_trials, random_state=random_state) if use_optimization else None
        self.random_state = random_state

    def select_strategies(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Inspect each column with missing values and select optimal strategy.

        Returns:
            Dictionary mapping column name -> {strategy, params, justification, eval_summary}
        """
        columns_to_impute = [c for c in df.columns if df[c].isna().sum() > 0 and c != target_column]
        decisions: Dict[str, Dict[str, Any]] = {}

        for col in columns_to_impute:
            s = df[col]
            n_missing = int(s.isna().sum())
            missing_ratio = float(n_missing / len(df))
            is_numeric = pd.api.types.is_numeric_dtype(s)
            n_observed = len(s.dropna())

            # Rule 1: High missingness (>80%) or very small observed cases (<10)
            if missing_ratio > 0.80 or n_observed < 10:
                strat = "median" if is_numeric else "constant"
                decisions[col] = {
                    "strategy": strat,
                    "params": {},
                    "justification": f"High missingness ({missing_ratio:.1%}) or low observed count ({n_observed}), using safe fallback.",
                    "evaluation": None
                }
                continue

            # Rule 2: Single feature dataset (no auxiliary features for multivariate methods)
            usable_features = [c for c in df.columns if c != col and c != target_column]
            if len(usable_features) == 0:
                # Univariate only
                if is_numeric:
                    skew = abs(s.dropna().skew()) if n_observed > 2 else 0
                    strat = "median" if skew > 1.0 else "mean"
                    just = "Univariate dataset with skewed distribution; median selected." if skew > 1.0 else "Univariate dataset with symmetric distribution; mean selected."
                else:
                    strat = "mode"
                    just = "Univariate categorical column; mode selected."

                decisions[col] = {
                    "strategy": strat,
                    "params": {},
                    "justification": just,
                    "evaluation": None
                }
                continue

            # Rule 3: Artificial Masking Evaluation across candidates
            eval_res = self.evaluator.evaluate_column_strategies(df, column=col)
            recommended = eval_res.get("recommended_strategy", "median" if is_numeric else "mode")
            justification = eval_res.get("justification", "Best performing on empirical masking.")

            params: Dict[str, Any] = {}
            # Optional Optuna tuning if strategy is tunable
            if self.use_optimization and self.optimizer is not None:
                if recommended == "knn" and is_numeric:
                    params = self.optimizer.optimize_knn(df, target_column=col)
                    justification += f" (Tuned with Optuna: {params})"
                elif recommended == "similarity":
                    params = self.optimizer.optimize_similarity(df, target_column=col)
                    justification += f" (Tuned with Optuna: {params})"

            decisions[col] = {
                "strategy": recommended,
                "params": params,
                "justification": justification,
                "evaluation": eval_res.get("strategies", {}),
            }

        return decisions
