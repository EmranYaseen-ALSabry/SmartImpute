import pytest
import pandas as pd
import numpy as np
from smartimpute.profiling import DataProfiler
from smartimpute.missingness import MissingnessAnalyzer


def test_profiling_and_missingness_detection():
    df = pd.DataFrame({
        "id": [f"id_{i}" for i in range(50)],
        "num_val": [10.0] * 45 + [1000.0] * 5,  # has outliers
        "constant_col": [1] * 50,               # constant
        "cat_val": ["A", "B", "A", None, "B"] * 10,
        "missing_num": [1.0, np.nan, 3.0, 4.0, np.nan] * 10,
    })

    profiler = DataProfiler()
    profile_res = profiler.profile(df)

    assert profile_res["overview"]["n_rows"] == 50
    assert profile_res["overview"]["n_cols"] == 5
    assert "constant_col" in profile_res["overview"]["constant_columns"]
    assert "id" in profile_res["overview"]["candidate_ids"]
    assert profile_res["columns"]["num_val"]["stats"]["outliers_iqr_count"] > 0

    analyzer = MissingnessAnalyzer()
    miss_res = analyzer.analyze(df)
    assert "missing_num" in miss_res["columns_with_missing"]
    assert "cat_val" in miss_res["columns_with_missing"]
    assert miss_res["total_missing_values"] == 30
