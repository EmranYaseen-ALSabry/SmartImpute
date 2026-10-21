import pytest
import pandas as pd
import numpy as np

from smartimpute.evaluation import ImputationEvaluator


def test_artificial_masking_evaluator():
    # Linear relationship where KNN or Mean can be evaluated
    np.random.seed(42)
    x = np.linspace(10, 100, 50)
    y = 2.0 * x + np.random.normal(0, 2, 50)
    df = pd.DataFrame({"x": x, "y": y})

    evaluator = ImputationEvaluator(mask_ratio=0.20, n_seeds=2, random_state=42)
    results = evaluator.evaluate_column_strategies(df, column="y", candidate_strategies=["mean", "median", "knn"])

    assert results["column"] == "y"
    assert "mean" in results["strategies"]
    assert "knn" in results["strategies"]
    assert "rmse" in results["strategies"]["mean"]["metrics"]
    assert "recommended_strategy" in results
    # KNN should significantly outperform Mean on linear relationship
    knn_rmse = results["strategies"]["knn"]["metrics"]["rmse"]["mean"]
    mean_rmse = results["strategies"]["mean"]["metrics"]["rmse"]["mean"]
    assert knn_rmse < mean_rmse
