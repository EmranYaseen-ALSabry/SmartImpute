"""
SimilarityBasedImputation: Intelligent neighbor-based imputation for mixed-type tabular data.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from smartimpute.exceptions import ImputationStrategyError
from smartimpute.strategies import BaseImputerStrategy


class SimilarityBasedImputer(BaseImputerStrategy):
    """
    Advanced similarity-based imputer that leverages Gower-like normalized distance
    over available features, handles mixed numeric and categorical types,
    excludes target/hidden columns from distance calculation, and provides full explainability.
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        weights: str = "distance",
        min_common_features: int = 1,
        exclude_columns: Optional[List[str]] = None,
        columns: Optional[List[str]] = None
    ):
        super().__init__(columns=columns)
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.min_common_features = min_common_features
        self.exclude_columns = set(exclude_columns or [])
        
        self.donor_data_: Optional[pd.DataFrame] = None
        self.numeric_cols_: List[str] = []
        self.categorical_cols_: List[str] = []
        self.ranges_: Dict[str, float] = {}
        self.fallback_values_: Dict[str, Any] = {}
        self.explanations_: Dict[int, Dict[str, Any]] = {}

    def fit(self, X: pd.DataFrame, y=None) -> "SimilarityBasedImputer":
        self.donor_data_ = X.copy()
        
        all_cols = [c for c in X.columns if c not in self.exclude_columns]
        self.numeric_cols_ = [c for c in all_cols if pd.api.types.is_numeric_dtype(X[c])]
        self.categorical_cols_ = [c for c in all_cols if not pd.api.types.is_numeric_dtype(X[c])]

        # Precompute ranges for numeric columns
        self.ranges_ = {}
        for c in self.numeric_cols_:
            s = X[c].dropna()
            r = float(s.max() - s.min()) if len(s) > 1 else 1.0
            self.ranges_[c] = r if r > 1e-8 else 1.0

        # Precompute fallback values
        self.fallback_values_ = {}
        for c in X.columns:
            if c in self.numeric_cols_:
                s = X[c].dropna()
                self.fallback_values_[c] = float(s.median()) if len(s) > 0 else 0.0
            else:
                m = X[c].dropna().mode()
                self.fallback_values_[c] = m.iloc[0] if len(m) > 0 else "missing"

        self.fitted_ = True
        return self

    def _compute_gower_distances(
        self,
        query_row: pd.Series,
        donor_pool: pd.DataFrame,
        feature_subset: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Gower distance between a single query row and a pool of donor rows
        using only the feature_subset (which strictly excludes the column being imputed).
        """
        n_donors = len(donor_pool)
        dist_sum = np.zeros(n_donors, dtype=np.float64)
        weight_sum = np.zeros(n_donors, dtype=np.float64)

        for col in feature_subset:
            q_val = query_row[col]
            if pd.isna(q_val):
                continue

            d_vals = donor_pool[col].values
            valid_mask = ~pd.isna(d_vals)
            weight_sum += valid_mask.astype(np.float64)

            if col in self.numeric_cols_:
                r = self.ranges_.get(col, 1.0)
                # Absolute difference normalized by range
                diff = np.abs(d_vals[valid_mask] - q_val) / r
                dist_sum[valid_mask] += diff
            else:
                # 0 if match, 1 if mismatch
                mismatch = (d_vals[valid_mask] != q_val).astype(np.float64)
                dist_sum[valid_mask] += mismatch

        # Where weight_sum < min_common_features, distance is infinity (unreliable)
        valid_donors = weight_sum >= max(1, self.min_common_features)
        distances = np.full(n_donors, np.inf, dtype=np.float64)
        distances[valid_donors] = dist_sum[valid_donors] / weight_sum[valid_donors]

        return distances, weight_sum

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_ or self.donor_data_ is None:
            raise ImputationStrategyError("SimilarityBasedImputer is not fitted.")

        X_out = X.copy()
        target_cols = self.columns or [c for c in X.columns if c not in self.exclude_columns]
        self.explanations_ = {}

        for col in target_cols:
            if col not in X_out.columns:
                continue

            missing_mask = X_out[col].isna()
            if not missing_mask.any():
                continue

            # Candidate donor pool: donor rows that HAVE observed values for this column
            observed_donors = self.donor_data_[self.donor_data_[col].notna()]
            if len(observed_donors) == 0:
                # Fallback if no donors have observed values
                X_out[col] = X_out[col].fillna(self.fallback_values_[col])
                continue

            # Features used for distance strictly exclude the column being imputed!
            features_for_dist = [
                c for c in (self.numeric_cols_ + self.categorical_cols_)
                if c != col and c in observed_donors.columns
            ]

            if not features_for_dist:
                X_out[col] = X_out[col].fillna(self.fallback_values_[col])
                continue

            is_numeric = col in self.numeric_cols_

            for row_idx in X_out.index[missing_mask]:
                query_row = X_out.loc[row_idx]
                distances, weights_used = self._compute_gower_distances(
                    query_row, observed_donors, features_for_dist
                )

                finite_mask = np.isfinite(distances)
                if not np.any(finite_mask):
                    # No reliable neighbor found, fallback
                    X_out.loc[row_idx, col] = self.fallback_values_[col]
                    continue

                finite_indices = np.where(finite_mask)[0]
                k = min(self.n_neighbors, len(finite_indices))
                
                # Sort finite distances to find nearest k
                sorted_idx_within_finite = np.argsort(distances[finite_indices])[:k]
                best_donor_pool_indices = finite_indices[sorted_idx_within_finite]

                best_distances = distances[best_donor_pool_indices]
                donor_sub = observed_donors.iloc[best_donor_pool_indices]
                donor_values = donor_sub[col].values

                # Calculate donor weights (inverse distance weighting)
                eps = 1e-4
                inv_distances = 1.0 / (best_distances + eps)
                norm_weights = inv_distances / inv_distances.sum()

                # Impute value
                if is_numeric:
                    if self.weights == "distance":
                        imputed_val = float(np.sum(norm_weights * donor_values))
                    else:
                        imputed_val = float(np.median(donor_values))
                else:
                    # Categorical: weighted voting
                    cat_votes: Dict[Any, float] = {}
                    for v, w in zip(donor_values, norm_weights):
                        cat_votes[v] = cat_votes.get(v, 0.0) + w
                    imputed_val = max(cat_votes.items(), key=lambda item: item[1])[0]

                X_out.loc[row_idx, col] = imputed_val

                # Record explainability data
                if row_idx not in self.explanations_:
                    self.explanations_[row_idx] = {}
                self.explanations_[row_idx][col] = {
                    "donor_indices": donor_sub.index.tolist(),
                    "distances": [round(float(d), 4) for d in best_distances],
                    "donor_values": [v if not isinstance(v, (np.floating, float)) else round(float(v), 3) for v in donor_values],
                    "reliability_score": round(float(1.0 / (1.0 + np.mean(best_distances))), 3),
                }

        return X_out
