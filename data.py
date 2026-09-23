"""Load the S&P 500 adjusted close prices used in the project."""
import pandas as pd


def load_prices(path="stocks.pkl"):
    """Adjusted close prices with empty days and incomplete series dropped (as in Boyd et al.).

    Returns ``(P, R)`` where ``R = P.pct_change().dropna()``.
    """
    data = pd.read_pickle(path)
    P = data['Adj Close']
    P = P[~P.isna().all(axis=1)]  # dropping rows with all missing values
    P = P.loc[:, ~P.isna().any(axis=0)]
    R = P.pct_change().dropna(axis=0)
    return P, R
