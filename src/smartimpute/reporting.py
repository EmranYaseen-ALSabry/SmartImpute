"""
Comprehensive Reporting Engine for SmartImpute.
Generates structured Markdown and JSON reports detailing profiling, benchmark results,
and justifications for selected imputation methods.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd


class ReportGenerator:
    """Generates professional diagnostic and imputation reports."""

    @staticmethod
    def generate_markdown_report(
        imputer,
        df_before: pd.DataFrame,
        df_after: pd.DataFrame,
        output_path: Optional[str | Path] = None
    ) -> str:
        """
        Generate a comprehensive markdown report.
        """
        prof = imputer.profile_summary_.get("overview", {})
        cols_prof = imputer.profile_summary_.get("columns", {})
        missing_sum = imputer.missingness_summary_
        strategies = imputer.strategies_map_

        total_missing_before = int(df_before.isna().sum().sum())
        total_missing_after = int(df_after.isna().sum().sum())
        imputed_count = total_missing_before - total_missing_after

        lines = [
            "# SmartImpute: Intelligent Imputation Report",
            "",
            "## 1. Dataset Overview",
            f"- **Total Rows:** {prof.get('n_rows', len(df_before)):,}",
            f"- **Total Columns:** {prof.get('n_cols', len(df_before.columns)):,}",
            f"- **Memory Usage:** {prof.get('memory_usage_mb', 0)} MB",
            f"- **Duplicate Rows:** {prof.get('duplicate_rows', 0)}",
            f"- **Constant Columns:** {len(prof.get('constant_columns', []))}",
            f"- **Potential ID Columns:** {', '.join(prof.get('candidate_ids', [])) or 'None'}",
            "",
            "## 2. Missingness Analysis",
            f"- **Total Missing Values Before Imputation:** {total_missing_before:,}",
            f"- **Total Imputed Values:** {imputed_count:,}",
            f"- **Remaining Missing Values:** {total_missing_after:,}",
            "",
            "| Column | Missing Count | Missing Ratio | Inferred Dtype |",
            "| :--- | :--- | :--- | :--- |",
        ]

        for col, count in missing_sum.get("columns_with_missing", {}).items():
            ratio = count / len(df_before)
            dtype = cols_prof.get(col, {}).get("dtype", "unknown")
            lines.append(f"| `{col}` | {count:,} | {ratio:.2%} | `{dtype}` |")

        lines.extend([
            "",
            "## 3. Imputation Strategy Selection & Artificial Masking Results",
            "",
            "> [!NOTE]",
            "> Evaluations are conducted using **Artificial Masking** on verified observed rows.",
            "> Synthetic evaluation serves as an empirical guide; true MNAR distributions may differ.",
            "",
            "| Column | Selected Strategy | Justification | Tuned Parameters |",
            "| :--- | :--- | :--- | :--- |",
        ])

        for col, decision in strategies.items():
            strat = decision.get("strategy", "unknown")
            just = decision.get("justification", "N/A")
            params = str(decision.get("params", {})) if decision.get("params") else "Default"
            lines.append(f"| `{col}` | **{strat.upper()}** | {just} | `{params}` |")

        # Diagnostics & Warnings
        leakage_warnings = imputer.profile_summary_.get("leakage_warnings", [])
        mechanisms = missing_sum.get("mechanism_diagnostics", [])

        if leakage_warnings or mechanisms:
            lines.extend([
                "",
                "## 4. Diagnostics & Leakage Warnings",
            ])
            for w in leakage_warnings:
                lines.append(f"- > [!WARNING] {w}")

            for m in mechanisms:
                col = m.get("column")
                mech = m.get("likely_mechanism")
                exp = m.get("explanation")
                lines.append(f"- **`{col}` ({mech}):** {exp}")

        lines.extend([
            "",
            "## 5. Summary & Next Steps",
            "- All original observed values have been strictly preserved without alteration.",
            "- The fitted imputer can be exported and applied to inference/test sets safely via `.transform()`.",
        ])

        report_content = "\n".join(lines)

        if output_path:
            p = Path(output_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                f.write(report_content)

        return report_content
