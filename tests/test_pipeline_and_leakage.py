import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path

from smartimpute.pipeline import SmartImputer
from smartimpute.exceptions import DataLeakageError, InvalidDataError


def test_case_1_zero_missing_data():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": ["x", "y", "z"]})
    imputer = SmartImputer()
    df_clean = imputer.fit_transform(df)
    pd.testing.assert_frame_equal(df, df_clean)


def test_case_2_and_3_and_4_mixed_numeric_categorical_missing():
    df = pd.DataFrame({
        "num": [10.0, 20.0, np.nan, 40.0, 50.0] * 5,
        "cat": ["A", "B", None, "A", "B"] * 5,
    })
    imputer = SmartImputer()
    df_clean = imputer.fit_transform(df)
    assert not df_clean.isna().any().any()
    # Ensure known values preserved
    assert df_clean["num"].iloc[0] == 10.0
    assert df_clean["cat"].iloc[0] == "A"


def test_case_5_all_missing_column():
    df = pd.DataFrame({
        "a": [1.0, 2.0, 3.0, 4.0, 5.0],
        "all_null": [np.nan, np.nan, np.nan, np.nan, np.nan],
    })
    imputer = SmartImputer()
    df_clean = imputer.fit_transform(df)
    # Shouldn't crash; falls back gracefully
    assert not df_clean["all_null"].isna().any()


def test_case_6_constant_column():
    df = pd.DataFrame({
        "const": [100.0, 100.0, 100.0, np.nan, 100.0],
        "other": [1, 2, 3, 4, 5],
    })
    imputer = SmartImputer()
    df_clean = imputer.fit_transform(df)
    assert (df_clean["const"] == 100.0).all()


def test_case_7_very_small_dataset():
    df = pd.DataFrame({"a": [1.0, np.nan]})
    imputer = SmartImputer()
    df_clean = imputer.fit_transform(df)
    assert df_clean["a"].iloc[1] == 1.0


def test_case_8_and_9_duplicates_and_outliers():
    df = pd.DataFrame({
        "val": [10.0, 10.0, 10.0, 10000.0, np.nan] * 4,
    })
    imputer = SmartImputer()
    df_clean = imputer.fit_transform(df)
    assert not df_clean.isna().any().any()


def test_case_10_high_cardinality_and_ids():
    df = pd.DataFrame({
        "id": [f"ID_{i}" for i in range(30)],
        "score": [float(i) for i in range(25)] + [np.nan] * 5,
    })
    imputer = SmartImputer()
    df_clean = imputer.fit_transform(df)
    assert not df_clean["score"].isna().any()
    assert (df_clean["id"] == df["id"]).all()


def test_case_11_and_13_high_missing_ratio():
    # 90% missing
    df = pd.DataFrame({
        "scarce": [10.0] + [np.nan] * 19,
        "feature": list(range(20)),
    })
    imputer = SmartImputer()
    df_clean = imputer.fit_transform(df)
    assert not df_clean["scarce"].isna().any()
    assert df_clean["scarce"].iloc[0] == 10.0


def test_strict_preservation_of_known_values():
    df = pd.DataFrame({
        "col1": [1.23, 4.56, np.nan, 7.89],
        "col2": ["alpha", "beta", "gamma", None],
    })
    imputer = SmartImputer()
    res = imputer.fit_transform(df)

    assert res.loc[0, "col1"] == 1.23
    assert res.loc[1, "col1"] == 4.56
    assert res.loc[3, "col1"] == 7.89
    assert res.loc[0, "col2"] == "alpha"
    assert res.loc[1, "col2"] == "beta"
    assert res.loc[2, "col2"] == "gamma"


def test_zero_data_leakage_in_ml_mode():
    df = pd.DataFrame({
        "feature1": [1.0, 2.0, np.nan, 4.0, 5.0],
        "target": [0, 1, 0, 1, 0],
    })
    imputer = SmartImputer(target_column="target", mode="ml_pipeline")
    res = imputer.fit_transform(df)

    # Target column must be completely untouched and NOT imputed
    assert (res["target"] == df["target"]).all()
    # Target column should not appear in strategies_map_
    assert "target" not in imputer.strategies_map_


def test_serialization_save_and_load(tmp_path):
    train_df = pd.DataFrame({
        "a": [1.0, 2.0, 3.0, 4.0, 5.0],
        "b": [10.0, 20.0, np.nan, 40.0, 50.0],
    })
    test_df = pd.DataFrame({
        "a": [6.0, 7.0],
        "b": [np.nan, 70.0],
    })

    imputer = SmartImputer()
    imputer.fit(train_df)

    save_path = tmp_path / "imputer_model.joblib"
    imputer.save(save_path)

    # Load in fresh object
    loaded_imputer = SmartImputer.load(save_path)
    res_test = loaded_imputer.transform(test_df)

    assert not res_test["b"].isna().any()
    assert res_test["b"].iloc[1] == 70.0  # Known value preserved
