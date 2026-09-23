"""Portfolio strategies and the rebalancing backtest."""
import warnings

import cvxpy as cp
import numpy as np
import pandas as pd

TRADING_DAYS = 250


def _cov_sqrt(R):
    """Symmetric PSD square root of the sample covariance.

    With fewer observations than assets the sample covariance is singular, so
    tiny negative eigenvalues from round-off are clipped instead of letting
    scipy.linalg.sqrtm return complex values.
    """
    w, V = np.linalg.eigh(R.cov().values)
    return (V * np.sqrt(np.clip(w, 0, None))) @ V.T


def equal_weight(R):
    N_assets = len(R.columns)
    return 1 / N_assets * np.ones(N_assets)


def _solve_or_equal_weight(prob, x, name):
    try:
        prob.solve()
    except cp.SolverError:
        pass
    if x.value is None or prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        warnings.warn("%s: problem %s, falling back to equal weight" % (name, prob.status))
        m = x.shape[0]
        return 1 / m * np.ones(m)
    return x.value


def markowitz_mvo(R, alpha_val):
    """Long-only minimum variance portfolio with daily expected return >= alpha_val."""
    cov_half = _cov_sqrt(R)
    mean_ = R.mean().values
    m = R.shape[1]

    x = cp.Variable(m)
    constraints = [x >= 0,
                   cp.sum(x) == 1,
                   mean_ @ x >= alpha_val]
    objective = cp.Minimize(cp.sum_squares(cov_half @ x))
    return _solve_or_equal_weight(cp.Problem(objective, constraints), x, "markowitz_mvo")


def blanchet_mvo(R, delta_val, alpha_val, norm_p=2):
    """Wasserstein distributionally robust mean-variance (Blanchet, Chen & Zhou).

    With transport cost ||u - v||_q^2 the robust problem penalises ||x||_p,
    where 1/p + 1/q = 1; ``norm_p=2`` is the Euclidean cost. Note that
    ``norm_p=1`` makes the penalty constant under the long-only budget
    constraint, which reduces the problem to MVO with target
    ``alpha_val + sqrt(delta_val)``.
    """
    cov_half = _cov_sqrt(R)
    mean_ = R.mean().values
    m = R.shape[1]
    x = cp.Variable(m)
    penalty = np.sqrt(delta_val) * cp.norm(x, norm_p)

    constraints = [x >= 0,
                   cp.sum(x) == 1,
                   mean_ @ x >= alpha_val + penalty]
    objective = cp.Minimize(cp.norm(cov_half @ x, 2) + penalty)
    return _solve_or_equal_weight(cp.Problem(objective, constraints), x, "blanchet_mvo")


def fixed_schedule(period, n_prices):
    """Rebalance every ``period`` days, estimating on the preceding ``period`` returns."""
    rebal = np.arange(period, n_prices, period)
    return rebal, rebal - period


def test_variable_strategy(R, P, rebal, est, strategy, function_params=()):
    """Backtest a strategy that rebalances at ``P.iloc[rebal[i]]``.

    The weights at ``rebal[i]`` are ``strategy(R.iloc[est[i]:rebal[i]], *function_params)``,
    i.e. estimated from returns known at that time. The portfolio starts
    as $1 equally weighted until the first rebalance.
    """
    rebal = np.asarray(rebal, dtype=int)
    est = np.asarray(est, dtype=int)
    if len(rebal) != len(est):
        raise ValueError("rebal and est must have the same length")
    if len(rebal) and (rebal[0] < 1 or rebal[-1] >= len(P) or np.any(np.diff(rebal) <= 0)):
        raise ValueError("rebal must be increasing and within 1..len(P)-1")

    N_assets = len(R.columns)
    prices = P.values

    #vectors to keep track of the weights and holdings
    holdings_t = np.zeros([N_assets, len(P)])
    weights_t = np.zeros([N_assets, len(P)])

    #1 over n dollars for each stock
    w = equal_weight(R)
    starts = np.concatenate(([0], rebal))
    ends = np.append(rebal, len(P))
    h = w / prices[0]
    holdings_t[:, :ends[0]] = h[:, None]
    weights_t[:, :ends[0]] = w[:, None]

    for i, t in enumerate(rebal):
        Portfolio_Value_t = np.dot(holdings_t[:, t - 1], prices[t])
        w_t = strategy(R.iloc[est[i]:t], *function_params)

        #reinvest the portfolio
        h_t = Portfolio_Value_t * w_t / prices[t]
        holdings_t[:, t:ends[i + 1]] = h_t[:, None]
        weights_t[:, t:ends[i + 1]] = w_t[:, None]

    Wealth = (holdings_t.transpose() * P).sum(axis=1)
    return holdings_t, weights_t, Wealth


def test_fixed_strategy(period, R, P, strategy, function_params=()):
    rebal, est = fixed_schedule(period, len(P))
    return test_variable_strategy(R, P, rebal, est, strategy, function_params)


def max_dd(returns):
    """Maximum drawdown of a returns Series and its (start, end) positions."""
    r = returns.add(1).cumprod()
    dd = r.div(r.cummax()).sub(1)
    end = int(np.argmin(dd.values))
    start = int(np.argmax(r.values[:end + 1]))
    return dd.iloc[end], start, end


def RetStats(returns):
    """Annualised mean, volatility and Sharpe ratio (zero risk-free rate)."""
    ann = TRADING_DAYS
    mean, sd = ann * returns.mean(), ann ** 0.5 * returns.std()
    return mean, sd, mean / sd


def summary_stats(Results):
    """Table of drawdown and return statistics for each backtest in ``Results``."""
    Stats = {}
    for key, (_, _, Wealth) in Results.items():
        returns = Wealth.pct_change().dropna(axis=0)
        mdd, start, end = max_dd(returns)
        AR, SD, SR = RetStats(returns)
        Stats[key] = [mdd, start, end, AR, SD, SR]
    Out = pd.DataFrame(Stats).transpose().sort_index()
    Out.columns = ['mdd', 'start', 'end', 'AR', 'SD', 'SR']
    return Out
