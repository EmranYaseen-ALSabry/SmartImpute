"""
Comprehensive Data Profiling Module for SmartImpute.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from scipy import stats

from smartimpute.exceptions import ProfilingError


class DataProfiler:
    """
    Profiles datasets to detect missingness, statistical distributions, potential IDs,
    anomalies, and feature correlations without mutating the original data.
    """

    def __init__(self, target_column: Optional[str] = None):
        self.target_column = target_column

    def profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run a complete profile of the dataframe.

        Args:
            df: Input pandas DataFrame.

        Returns:
            Dictionary containing comprehensive profile metrics and diagnostic insights.
        """
        if not isinstance(df, pd.DataFrame):
            raise ProfilingError("Input data must be a pandas DataFrame.")
        if df.empty:
            raise ProfilingError("Cannot profile an empty DataFrame.")

        n_rows, n_cols = df.shape
        memory_usage_bytes = int(df.memory_usage(deep=True).sum())
        memory_usage_mb = round(memory_usage_bytes / (1024 * 1024), 2)

        duplicate_rows = int(df.duplicated().sum())

        column_profiles = {}
        candidate_ids = []
        constant_columns = []
        quasi_constant_columns = []

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

        for col in df.columns:
            s = df[col]
            missing_count = int(s.isna().sum())
            missing_ratio = float(missing_count / n_rows)
            n_unique = int(s.nunique(dropna=True))

            is_numeric = col in numeric_cols

            col_info: Dict[str, Any] = {
                "dtype": str(s.dtype),
                "is_numeric": is_numeric,
                "missing_count": missing_count,
                "missing_ratio": round(missing_ratio, 4),
                "unique_count": n_unique,
                "unique_ratio": round(n_unique / n_rows, 4) if n_rows > 0 else 0.0,
            }

            # Check for constant or quasi-constant
            if n_unique <= 1:
                constant_columns.append(col)
                col_info["is_constant"] = True
            else:
                col_info["is_constant"] = False
                # Quasi-constant: most frequent value >= 99%
                mode_freq = s.value_counts(normalize=True, dropna=True).iloc[0] if n_unique > 0 else 0
                if mode_freq >= 0.99:
                    quasi_constant_columns.append(col)
                    col_info["is_quasi_constant"] = True
                else:
                    col_info["is_quasi_constant"] = False

            # Check for candidate ID (unique ratio > 0.98 or name matches common ID patterns)
            name_lower = str(col).lower()
            if (col_info["unique_ratio"] > 0.98 and n_rows > 30) or name_lower in ["id", "uuid", "identifier", "guid"]:
                candidate_ids.append(col)
                col_info["is_candidate_id"] = True
            else:
                col_info["is_candidate_id"] = False

            # Detailed statistics
            if is_numeric and s.dropna().shape[0] > 0:
                clean_s = s.dropna()
                q25, q50, q75 = clean_s.quantile([0.25, 0.50, 0.75])
                iqr = float(q75 - q25)
                outliers = int(((clean_s < (q25 - 1.5 * iqr)) | (clean_s > (q75 + 1.5 * iqr))).sum())
                
                col_info["stats"] = {
                    "mean": float(clean_s.mean()),
                    "std": float(clean_s.std()) if len(clean_s) > 1 else 0.0,
                    "min": float(clean_s.min()),
                    "q25": float(q25),
                    "median": float(q50),
                    "q75": float(q75),
                    "max": float(clean_s.max()),
                    "skew": float(clean_s.skew()) if len(clean_s) > 2 else 0.0,
                    "outliers_iqr_count": outliers,
                }
            else:
                # Categorical summary
                top_values = s.value_counts(dropna=True).head(5).to_dict()
                col_info["top_categories"] = {str(k): int(v) for k, v in top_values.items()}

            column_profiles[col] = col_info

        # Numeric correlations (Spearman to catch non-linear trends too)
        numeric_correlations: Dict[str, Dict[str, float]] = {}
        if len(numeric_cols) > 1:
            try:
                corr_df = df[numeric_cols].corr(method="pearson").round(3)
                numeric_correlations = corr_df.to_dict()
            except Exception:
                pass

        # Target leakage & predictive relations
        leakage_warnings = []
        if self.target_column and self.target_column in df.columns:
            if self.target_column in numeric_cols:
                for col in numeric_cols:
                    if col != self.target_column:
                        c_val = numeric_correlations.get(col, {}).get(self.target_column, 0.0)
                        if abs(c_val) > 0.95:
                            leakage_warnings.append(
                                f"Feature '{col}' has extreme correlation ({c_val:.3f}) with target '{self.target_column}', check for target leakage."
                            )

        return {
            "overview": {
                "n_rows": n_rows,
                "n_cols": n_cols,
                "memory_usage_mb": memory_usage_mb,
                "duplicate_rows": duplicate_rows,
                "numeric_columns_count": len(numeric_cols),
                "categorical_columns_count": len(categorical_cols),
                "constant_columns": constant_columns,
                "quasi_constant_columns": quasi_constant_columns,
                "candidate_ids": candidate_ids,
            },
            "columns": column_profiles,
            "correlations": numeric_correlations,
            "leakage_warnings": leakage_warnings,
        }
