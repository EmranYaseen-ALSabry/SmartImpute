"""
Command-Line Interface (CLI) for SmartImpute.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import pandas as pd

from smartimpute.data_loader import load_dataset
from smartimpute.profiling import DataProfiler
from smartimpute.missingness import MissingnessAnalyzer
from smartimpute.evaluation import ImputationEvaluator
from smartimpute.pipeline import SmartImputer
from smartimpute.reporting import ReportGenerator


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="smartimpute",
        description="SmartImpute: Intelligent Missing Data Imputation System"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: analyze
    p_analyze = subparsers.add_parser("analyze", help="Profile dataset and analyze missingness")
    p_analyze.add_argument("data", type=str, help="Path to CSV dataset")
    p_analyze.add_argument("--target", type=str, default=None, help="Optional target column")

    # Command: evaluate
    p_eval = subparsers.add_parser("evaluate", help="Benchmark imputation strategies via artificial masking")
    p_eval.add_argument("data", type=str, help="Path to CSV dataset")
    p_eval.add_argument("--column", type=str, default=None, help="Specific column to evaluate (or all if omitted)")

    # Command: impute
    p_impute = subparsers.add_parser("impute", help="Impute dataset using adaptive strategies")
    p_impute.add_argument("data", type=str, help="Path to CSV dataset")
    p_impute.add_argument("--output", "-o", type=str, default="cleaned.csv", help="Output path for cleaned CSV")
    p_impute.add_argument("--report", "-r", type=str, default="imputation_report.md", help="Output path for report")
    p_impute.add_argument("--optimize", action="store_true", help="Enable Optuna hyperparameter optimization")

    # Command: run (ML Pipeline mode)
    p_run = subparsers.add_parser("run", help="Run ML-aware leakage-free imputation pipeline")
    p_run.add_argument("data", type=str, help="Path to CSV dataset")
    p_run.add_argument("--target", "-t", type=str, required=True, help="Target column name")
    p_run.add_argument("--output", "-o", type=str, default="cleaned_ml.csv", help="Output path for cleaned CSV")

    return parser


def main(args: list[str] | None = None) -> int:
    parser = create_parser()
    parsed_args = parser.parse_args(args)

    if not parsed_args.command:
        parser.print_help()
        return 0

    try:
        if parsed_args.command == "analyze":
            df = load_dataset(parsed_args.data)
            profiler = DataProfiler(target_column=parsed_args.target)
            prof = profiler.profile(df)
            analyzer = MissingnessAnalyzer()
            miss = analyzer.analyze(df)

            print("================ SmartImpute Analysis ================")
            print(f"Dataset: {parsed_args.data}")
            print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
            print(f"Total Missing Values: {miss['total_missing_values']} ({miss['overall_missing_ratio']:.2%})")
            print("\nColumns with Missing Values:")
            for c, cnt in miss["columns_with_missing"].items():
                print(f"  - {c}: {cnt} missing ({cnt/len(df):.2%})")
            print("======================================================")
            return 0

        elif parsed_args.command == "evaluate":
            df = load_dataset(parsed_args.data)
            evaluator = ImputationEvaluator(mask_ratio=0.15, n_seeds=2)
            cols = [parsed_args.column] if parsed_args.column else [c for c in df.columns if df[c].isna().sum() > 0]

            print("================ SmartImpute Evaluation ==============")
            for c in cols:
                res = evaluator.evaluate_column_strategies(df, column=c)
                print(f"\nTarget Column: {c}")
                print(f"Recommended Strategy: {res.get('recommended_strategy')}")
                print(f"Justification: {res.get('justification')}")
                for strat, data in res.get("strategies", {}).items():
                    metrics_str = ", ".join([f"{k}={v['mean']:.4f}" for k, v in data["metrics"].items()])
                    print(f"  - {strat.upper()}: {metrics_str} (time: {data['execution_time_sec']}s)")
            print("======================================================")
            return 0

        elif parsed_args.command == "impute":
            df = load_dataset(parsed_args.data)
            imputer = SmartImputer(use_optimization=parsed_args.optimize)
            df_cleaned = imputer.fit_transform(df)

            out_path = Path(parsed_args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df_cleaned.to_csv(out_path, index=False)

            ReportGenerator.generate_markdown_report(imputer, df, df_cleaned, output_path=parsed_args.report)
            print(f"Imputation complete! Cleaned dataset saved to: {out_path}")
            print(f"Report generated at: {parsed_args.report}")
            return 0

        elif parsed_args.command == "run":
            df = load_dataset(parsed_args.data)
            imputer = SmartImputer(target_column=parsed_args.target, mode="ml_pipeline")
            df_cleaned = imputer.fit_transform(df)

            out_path = Path(parsed_args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df_cleaned.to_csv(out_path, index=False)
            print(f"ML Pipeline imputation complete (Zero Leakage). Cleaned data saved to: {out_path}")
            return 0

    except Exception as e:
        print(f"Error during execution: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
