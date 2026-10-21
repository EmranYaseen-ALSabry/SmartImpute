import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from smartimpute.data_loader import load_dataset
from smartimpute.exceptions import InvalidDataError


def test_load_dataframe_directly():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    loaded = load_dataset(df)
    assert loaded.shape == (3, 2)
    assert not loaded.empty


def test_load_numpy_array():
    arr = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    loaded = load_dataset(arr)
    assert loaded.shape == (3, 2)
    assert "feature_0" in loaded.columns


def test_empty_dataset_raises_error():
    empty_df = pd.DataFrame()
    with pytest.raises(InvalidDataError):
        load_dataset(empty_df)


def test_nonexistent_file_raises_error():
    with pytest.raises(InvalidDataError):
        load_dataset("non_existent_file_12345.csv")
