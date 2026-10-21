"""
Data loading and ingestion utilities for SmartImpute.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union
import pandas as pd
import numpy as np

from smartimpute.exceptions import InvalidDataError


def load_dataset(
    source: Union[str, Path, pd.DataFrame, np.ndarray],
    **kwargs
) -> pd.DataFrame:
    """
    Safely load a dataset into a pandas DataFrame with initial validation.

    Args:
        source: Path to CSV, pandas DataFrame, or NumPy array.
        **kwargs: Extra parameters passed to pd.read_csv if source is a file path.

    Returns:
        pd.DataFrame: Validated DataFrame copy.

    Raises:
        InvalidDataError: If data is empty or unsupported format.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise InvalidDataError(f"File not found: {path}")
        try:
            df = pd.read_csv(path, **kwargs)
        except Exception as e:
            raise InvalidDataError(f"Failed to read CSV at {path}: {e}") from e
    elif isinstance(source, pd.DataFrame):
        df = source.copy()
    elif isinstance(source, np.ndarray):
        if source.ndim == 1:
            df = pd.DataFrame(source, columns=["feature_0"])
        elif source.ndim == 2:
            cols = [f"feature_{i}" for i in range(source.shape[1])]
            df = pd.DataFrame(source, columns=cols)
        else:
            raise InvalidDataError(f"Unsupported numpy array with {source.ndim} dimensions.")
    else:
        raise InvalidDataError(f"Unsupported data source type: {type(source)}")

    if df.empty or df.shape[0] == 0:
        raise InvalidDataError("Dataset is empty (0 rows).")
    if df.shape[1] == 0:
        raise InvalidDataError("Dataset has 0 columns.")

    return df
