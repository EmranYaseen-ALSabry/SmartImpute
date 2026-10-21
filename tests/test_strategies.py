import pytest
import pandas as pd
import numpy as np

from smartimpute.strategies.numeric import (
    MeanImputer,
    MedianImputer,
    KNNNumericImputer,
    IterativeNumericImputer,
    GroupBasedNumericImputer,
)
from smartimpute.strategies.categorical import (
    ModeImputer,
    ConstantCategoryImputer,
    GroupBasedCategoricalImputer,
    ModelBasedCategoricalImputer,
)
from smartimpute.strategies.timeseries import TimeSeriesImputer


def test_numeric_mean_and_median():
    df = pd.DataFrame({"val": [10.0, 20.0, 30.0, np.nan]})
    
    mean_imp = MeanImputer().fit(df)
    res_mean = mean_imp.transform(df)
    assert res_mean["val"].iloc[3] == 20.0
    # Original known values unchanged
    assert res_mean["val"].iloc[0] == 10.0

    med_imp = MedianImputer().fit(df)
    res_med = med_imp.transform(df)
    assert res_med["val"].iloc[3] == 20.0


def test_knn_and_iterative_numeric():
    df = pd.DataFrame({
        "x": [1.0, 2.0, 3.0, 4.0, 5.0, np.nan],
        "y": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
    })
    
    knn = KNNNumericImputer(n_neighbors=2).fit(df)
    res_knn = knn.transform(df)
    assert not res_knn.isna().any().any()
    assert res_knn["x"].iloc[5] > 3.0  # Should be close to 4 or 5

    iter_imp = IterativeNumericImputer(max_iter=5).fit(df)
    res_iter = iter_imp.transform(df)
    assert not res_iter.isna().any().any()
    assert np.isclose(res_iter["x"].iloc[5], 6.0, atol=0.5)


def test_group_based_imputers():
    df = pd.DataFrame({
        "group": ["A", "A", "A", "B", "B", "B"],
        "num_val": [10.0, 12.0, np.nan, 100.0, 110.0, np.nan],
        "cat_val": ["high", "high", None, "low", "low", None],
    })

    grp_num = GroupBasedNumericImputer(group_column="group").fit(df)
    res_num = grp_num.transform(df)
    assert res_num["num_val"].iloc[2] == 11.0
    assert res_num["num_val"].iloc[5] == 105.0

    grp_cat = GroupBasedCategoricalImputer(group_column="group").fit(df)
    res_cat = grp_cat.transform(df)
    assert res_cat["cat_val"].iloc[2] == "high"
    assert res_cat["cat_val"].iloc[5] == "low"


def test_categorical_mode_and_model_based():
    df = pd.DataFrame({
        "feature1": [1.0, 2.0, 1.1, 10.0, 11.0, 10.5],
        "category": ["apple", "apple", "apple", "banana", "banana", None],
    })

    mode_imp = ModeImputer().fit(df)
    res_mode = mode_imp.transform(df)
    assert res_mode["category"].iloc[5] == "apple"

    model_imp = ModelBasedCategoricalImputer(columns=["category"]).fit(df)
    res_model = model_imp.transform(df)
    assert not res_model["category"].isna().any()


def test_timeseries_imputer_causality():
    df = pd.DataFrame({"ts": [1.0, np.nan, np.nan, 4.0, np.nan]})
    
    # ffill preserves causality (past -> future)
    ts_imp = TimeSeriesImputer(method="ffill", allow_future=False).fit(df)
    res = ts_imp.transform(df)
    assert res["ts"].iloc[1] == 1.0
    assert res["ts"].iloc[2] == 1.0
    assert res["ts"].iloc[4] == 4.0
