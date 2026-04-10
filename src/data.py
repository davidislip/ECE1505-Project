"""
Data loading and preprocessing helpers.
"""

import pickle
import numpy as np
import pandas as pd


def load_stocks(path="stocks.pkl"):
    """
    Load the pre-downloaded S&P 500 OHLCV dataset.

    Parameters
    ----------
    path : str
        Path to the pickle file (default: stocks.pkl).

    Returns
    -------
    pd.DataFrame or dict
        Raw dataset as stored.
    """
    with open(path, "rb") as f:
        return pickle.load(f)


def compute_returns(prices):
    """
    Compute daily log returns from a price DataFrame, dropping assets
    with any missing observations.

    Parameters
    ----------
    prices : pd.DataFrame
        Price matrix (T × N).

    Returns
    -------
    R : pd.DataFrame
        Return matrix (T-1 × N) — only columns with complete data.
    P : pd.DataFrame
        Aligned price matrix (T-1 × N).
    """
    # Drop any column that has NaN anywhere
    prices_clean = prices.dropna(axis=1, how="any")
    R = np.log(prices_clean).diff().dropna()
    P = prices_clean.loc[R.index]
    return R, P


def load_results(filepath):
    """
    Load a changepoint detection result file.

    Parses parameters from the filename convention ``{delta}_{arl}_{n}.pkl``
    and returns both the result dict and the decoded parameters.

    Parameters
    ----------
    filepath : str
        Path to a results .pkl file, e.g. ``Results/0.1_10000_30.pkl``.

    Returns
    -------
    result : dict
        Keys: "times", "mid_points", "Ts", "ARL"
    params : dict
        Keys: "delta" (float), "arl" (float), "n" (int)
    """
    with open(filepath, "rb") as f:
        result = pickle.load(f)

    stem = filepath[filepath.rfind("/") + 1:filepath.rfind(".pkl")]
    parts = stem.split("_")
    params = {
        "delta": float(parts[0]),
        "arl": float(parts[1]),
        "n": int(parts[2]),
    }
    return result, params
