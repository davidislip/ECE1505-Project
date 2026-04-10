"""
Backtesting utilities: fixed-schedule and changepoint-driven rebalancing.
"""

import numpy as np


# Simple round-trip transaction cost model
TRANSACTION_COST_BPS = 10  # 10 basis points round-trip


def apply_transaction_costs(old_weights, new_weights, portfolio_value, cost_bps=TRANSACTION_COST_BPS):
    """
    Deduct trading costs proportional to portfolio turnover.

    Parameters
    ----------
    old_weights : np.ndarray
        Weights before rebalancing.
    new_weights : np.ndarray
        Weights after rebalancing.
    portfolio_value : float
        Current portfolio value before costs.
    cost_bps : float
        Round-trip cost in basis points (default 10 bps).

    Returns
    -------
    float
        Portfolio value after transaction costs.
    """
    turnover = np.abs(new_weights - old_weights).sum()
    return portfolio_value * (1 - turnover * cost_bps / 10_000)


def test_fixed_strategy(period, R, P, strategy, function_params, cost_bps=0):
    """
    Backtest a strategy that rebalances on a fixed schedule.

    Parameters
    ----------
    period : int
        Number of days between rebalancing events.
    R : pd.DataFrame
        Return matrix (T × N).
    P : pd.DataFrame
        Price matrix (T × N).
    strategy : callable
        Portfolio allocation function, e.g. markowitz_mvo.
    function_params : tuple
        Extra arguments forwarded to strategy.
    cost_bps : float
        Round-trip transaction cost in basis points (default 0).

    Returns
    -------
    holdings_t : np.ndarray  (N × T)
    weights_t  : np.ndarray  (N × T)
    Wealth     : pd.Series
    """
    N_assets = len(R.columns)
    Portfolio_Value = 1.0

    holdings_t = np.zeros([N_assets, len(P)])
    weights_t = np.zeros([N_assets, len(P)])

    w = np.ones(N_assets) / N_assets
    h = Portfolio_Value * w / P.iloc[0, :].values

    t = 0
    holdings_t[:, 0:period] = h.reshape(-1, 1) * np.ones([N_assets, period])
    weights_t[:, 0:period] = w.reshape(-1, 1) * np.ones([N_assets, period])

    while t + period <= len(P):
        prev_t = t
        t = t + period
        next_dt = min(period, len(P) - t)

        Portfolio_Value_t = np.dot(holdings_t[:, t - 1], P.iloc[t, :].values)

        w_t = strategy(R.iloc[prev_t:t], *function_params)

        if cost_bps > 0:
            Portfolio_Value_t = apply_transaction_costs(w, w_t, Portfolio_Value_t, cost_bps)

        w = w_t
        h_t = Portfolio_Value_t * w_t / P.iloc[t, :].values

        holdings_t[:, t:t + next_dt] = h_t.reshape(-1, 1) * np.ones([N_assets, next_dt])
        weights_t[:, t:t + next_dt] = w_t.reshape(-1, 1) * np.ones([N_assets, next_dt])

    Wealth = (holdings_t.T * P.values).sum(axis=1)
    import pandas as pd
    Wealth = pd.Series(Wealth, index=P.index)
    return holdings_t, weights_t, Wealth


def test_variable_strategy(R, P, rebal, est, strategy, function_params, cost_bps=0):
    """
    Backtest a strategy that rebalances at detected changepoint times.

    Parameters
    ----------
    R : pd.DataFrame
        Return matrix (T × N).
    P : pd.DataFrame
        Price matrix (T × N).
    rebal : array-like
        Rebalancing time indices.
    est : array-like
        Estimation window start indices; est[i] must be <= rebal[i].
    strategy : callable
        Portfolio allocation function.
    function_params : tuple
        Extra arguments forwarded to strategy.
    cost_bps : float
        Round-trip transaction cost in basis points (default 0).

    Returns
    -------
    holdings_t : np.ndarray  (N × T)
    weights_t  : np.ndarray  (N × T)
    Wealth     : pd.Series
    """
    if len(rebal) != len(est):
        raise ValueError("rebal and est must have the same length")

    for i in range(len(rebal)):
        assert est[i] <= rebal[i], (
            f"Look-ahead bias at index {i}: estimation window end ({est[i]}) "
            f"exceeds rebalancing time ({rebal[i]})"
        )

    N_assets = len(R.columns)
    Portfolio_Value = 1.0

    holdings_t = np.zeros([N_assets, len(P)])
    weights_t = np.zeros([N_assets, len(P)])

    w = np.ones(N_assets) / N_assets
    h = Portfolio_Value * w / P.iloc[0, :].values

    holdings_t[:, 0:rebal[0]] = h.reshape(-1, 1) * np.ones([N_assets, rebal[0]])
    weights_t[:, 0:rebal[0]] = w.reshape(-1, 1) * np.ones([N_assets, rebal[0]])

    holding_intervals = np.append(np.diff(rebal), len(P) - rebal[-1])

    for i in range(len(rebal)):
        prev_t = est[i]
        t = rebal[i]
        next_dt = holding_intervals[i]

        Portfolio_Value_t = np.dot(holdings_t[:, t - 1], P.iloc[t, :].values)

        w_t = strategy(R.iloc[prev_t:t], *function_params)

        if cost_bps > 0:
            Portfolio_Value_t = apply_transaction_costs(w, w_t, Portfolio_Value_t, cost_bps)

        w = w_t
        h_t = Portfolio_Value_t * w_t / P.iloc[t, :].values

        holdings_t[:, t:t + next_dt] = h_t.reshape(-1, 1) * np.ones([N_assets, next_dt])
        weights_t[:, t:t + next_dt] = w_t.reshape(-1, 1) * np.ones([N_assets, next_dt])

    Wealth = (holdings_t.T * P.values).sum(axis=1)
    import pandas as pd
    Wealth = pd.Series(Wealth, index=P.index)
    return holdings_t, weights_t, Wealth


def max_dd(returns):
    """
    Compute maximum drawdown of a return series.

    Parameters
    ----------
    returns : pd.Series
        Daily returns.

    Returns
    -------
    mdd   : float  — maximum drawdown (negative number, 0.0 if none)
    start : int    — index of drawdown peak
    end   : int    — index of drawdown trough
    """
    r = returns.add(1).cumprod()
    dd = r.div(r.cummax()).sub(1)
    mdd = dd.min()
    end = dd.argmin()
    if end == 0:
        return 0.0, 0, 0
    start = r.iloc[:end].argmax()
    return mdd, start, end


def RetStats(returns):
    """
    Compute annualized performance statistics.

    Parameters
    ----------
    returns : pd.Series
        Daily returns.

    Returns
    -------
    tuple : (annualized_return, annualized_std, sharpe_ratio)
    """
    P = 250
    P05 = P ** 0.5
    ann_return = P * returns.mean()
    ann_std = P05 * returns.std()
    sharpe = ann_return / ann_std
    return ann_return, ann_std, sharpe
