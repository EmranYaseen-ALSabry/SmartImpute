"""
Missingness pattern diagnosis and mechanism analysis module.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats


class MissingnessAnalyzer:
    """
    Analyzes missing data patterns, nullity correlations, and infers likely mechanisms (MCAR / MAR / MNAR).
    """

    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze missingness across columns and rows.

        Args:
            df: Input pandas DataFrame.

        Returns:
            Dictionary with missingness statistics, nullity correlations, and mechanism diagnostics.
        """
        n_rows, n_cols = df.shape
        missing_mask = df.isna()
        
        # Columns with missing values
        cols_with_missing = {
            col: int(missing_mask[col].sum())
            for col in df.columns
            if missing_mask[col].sum() > 0
        }
        
        # Sort by count descending
        sorted_missing = dict(sorted(cols_with_missing.items(), key=lambda item: item[1], reverse=True))
        
        # Row-level null distribution
        row_null_counts = missing_mask.sum(axis=1)
        row_distribution = {
            int(k): int(v) for k, v in row_null_counts.value_counts().sort_index().items()
        }

        # Nullity correlation (correlation between missing indicators)
        nullity_corr: Dict[str, Dict[str, float]] = {}
        missing_cols_list = list(sorted_missing.keys())
        if len(missing_cols_list) > 1:
            try:
                corr_matrix = missing_mask[missing_cols_list].corr().round(3)
                nullity_corr = corr_matrix.to_dict()
            except Exception:
                pass

        # Diagnostic testing for MAR/MNAR heuristics
        # For each missing column, check if other numeric columns have different means when this column is missing vs present
        mechanism_insights: List[Dict[str, Any]] = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        for target_col in missing_cols_list:
            is_missing_mask = missing_mask[target_col]
            n_missing = int(is_missing_mask.sum())
            if n_missing < 5 or (n_rows - n_missing) < 5:
                continue

            significant_relations = []
            for other_col in numeric_cols:
                if other_col == target_col or missing_mask[other_col].sum() > 0.5 * n_rows:
                    continue
                
                vals_when_missing = df.loc[is_missing_mask, other_col].dropna()
                vals_when_present = df.loc[~is_missing_mask, other_col].dropna()

                if len(vals_when_missing) >= 5 and len(vals_when_present) >= 5:
                    try:
                        stat, p_val = stats.mannwhitneyu(vals_when_missing, vals_when_present)
                        if p_val < 0.05:
                            diff_mean = float(vals_when_missing.mean() - vals_when_present.mean())
                            significant_relations.append({
                                "related_feature": other_col,
                                "p_value": round(float(p_val), 5),
                                "mean_diff": round(diff_mean, 3),
                            })
                    except Exception:
                        pass

            if significant_relations:
                mechanism_insights.append({
                    "column": target_col,
                    "likely_mechanism": "MAR (Missing at Random) or MNAR",
                    "explanation": f"Missingness in '{target_col}' statistically correlates with observed values in: " +
                                   ", ".join([r["related_feature"] for r in significant_relations[:3]]),
                    "evidence": significant_relations[:3],
                })
            else:
                mechanism_insights.append({
                    "column": target_col,
                    "likely_mechanism": "MCAR (Missing Completely at Random) candidate",
                    "explanation": f"No strong statistical dependency found between missingness in '{target_col}' and other observed numeric variables.",
                    "evidence": [],
                })

        return {
            "total_missing_values": int(missing_mask.sum().sum()),
            "overall_missing_ratio": round(float(missing_mask.sum().sum() / (n_rows * n_cols)), 4) if n_rows * n_cols > 0 else 0.0,
            "columns_with_missing": sorted_missing,
            "row_missing_distribution": row_distribution,
            "nullity_correlation": nullity_corr,
            "mechanism_diagnostics": mechanism_insights,
        }
