"""
Configuration manager for SmartImpute.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict
import yaml


DEFAULT_CONFIG: Dict[str, Any] = {
    "system": {
        "random_seed": 42,
        "verbose": True,
    },
    "evaluation": {
        "test_mask_ratio": 0.15,
        "n_splits": 3,
        "metrics": {
            "numeric": ["rmse", "mae"],
            "categorical": ["accuracy", "macro_f1"],
        },
    },
    "strategies": {
        "numeric": ["mean", "median", "knn", "iterative", "similarity"],
        "categorical": ["mode", "constant", "similarity"],
        "timeseries": ["ffill", "bfill", "linear_interpolation"],
    },
    "optimization": {
        "enabled": False,
        "n_trials": 15,
        "timeout": 60,
    },
    "similarity": {
        "n_neighbors": 5,
        "weights": "distance",
    },
    "missingness_indicator": {
        "add_indicator": False,
        "threshold": 0.05,
    },
}


def load_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file or return defaults.

    Args:
        config_path: Optional path to a YAML configuration file.

    Returns:
        Merged configuration dictionary.
    """
    config = DEFAULT_CONFIG.copy()

    if config_path is not None:
        p = Path(config_path)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                user_config = yaml.safe_load(f)
                if user_config and isinstance(user_config, dict):
                    # Deep merge
                    for k, v in user_config.items():
                        if isinstance(v, dict) and k in config and isinstance(config[k], dict):
                            config[k].update(v)
                        else:
                            config[k] = v

    return config
