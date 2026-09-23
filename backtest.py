"""Portfolio strategies and the rebalancing backtest."""
import warnings
from collections import Counter

import cvxpy as cp
import numpy as np
import pandas as pd
from sklearn.covariance import ledoit_wolf_shrinkage

TRADING_DAYS = 250


def cov_factor(R, shrink=False):
    """Matrix F with F.T @ F equal to the covariance estimate of ``R``.

    ``shrink=False`` gives the sample covariance ``R.cov()``. ``shrink=True``
    gives the Ledoit-Wolf estimate ``(1 - s) S + s mu I`` with ``S`` the
    maximum likelihood covariance and ``mu = trace(S) / m``. Using a factor
    instead of the m x m matrix keeps the conic problems small (the sample
    covariance has rank < T << m) and avoids matrix square roots of a
    singular matrix.
    """
    X = R.values - R.values.mean(axis=0)
    T, m = X.shape
    if not shrink:
        return X / np.sqrt(T - 1)
    s = ledoit_wolf_shrinkage(X, assume_centered=True)
    mu = np.sum(X ** 2) / (T * m)
    return np.vstack([np.sqrt((1 - s) / T) * X, np.sqrt(s * mu) * np.eye(m)])


def equal_weight(R):
    N_assets = len(R.columns)
    return 1 / N_assets * np.ones(N_assets)


# number of times each strategy fell back to equal weight
FALLBACKS = Counter()


def _solve_or_equal_weight(prob, x, name):
    try:
        prob.solve()
    except cp.SolverError:
        pass
    if x.value is None or prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        FALLBACKS[name] += 1
        warnings.warn("%s: problem %s, falling back to equal weight" % (name, prob.status))
        m = x.shape[0]
        return 1 / m * np.ones(m)
    return x.value


def markowitz_mvo(R, alpha_val, shrink=False):
    """Long-only minimum variance portfolio with daily expected return >= alpha_val."""
    F = cov_factor(R, shrink)
    mean_ = R.mean().values
    m = R.shape[1]

    x = cp.Variable(m)
    constraints = [x >= 0,
                   cp.sum(x) == 1,
                   mean_ @ x >= alpha_val]
    objective = cp.Minimize(cp.sum_squares(F @ x))
    return _solve_or_equal_weight(cp.Problem(objective, constraints), x, "markowitz_mvo" + ("_shrink" if shrink else ""))


def blanchet_mvo(R, delta_val, alpha_val, norm="l2", shrink=False, shrink_norm=None):
    """Wasserstein distributionally robust mean-variance (Blanchet, Chen & Zhou).

    For an order-2 Wasserstein ball of radius ``delta_val`` around the
    empirical distribution with transport cost ``||u - v||^2`` the problem is

        min  sqrt(x' S x) + sqrt(delta) ||x||_*
        s.t. mean' x >= alpha + sqrt(delta) ||x||_*,  x >= 0,  sum(x) = 1

    where ``||.||_*`` is the dual of the norm in the transport cost and ``S``
    is the covariance estimate (the square of the objective in the paper has
    the same minimiser).

    ``norm``:

    * ``"l2"``: Euclidean cost, ``||x||_* = ||x||_2``. (``"l1"`` is accepted
      but is constant under the long-only budget constraint, so it reduces
      to MVO with target ``alpha + sqrt(delta)``.)
    * ``"mahalanobis"``: cost ``(u - v)' Lambda (u - v)`` with
      ``Lambda = sigma2 * Sigma^-1``, so ``||x||_* = sqrt(x' Sigma x) / sigma``
      where ``sigma2 = trace(Sigma) / m``. The scaling makes the cost equal
      to the Euclidean one when ``Sigma = sigma2 * I``, so ``delta`` is on
      the same scale as for ``"l2"``. Moving mass along high-variance
      directions is cheap, so the adversary perturbs the returns in the
      directions the data already varies in.

    ``shrink`` selects Ledoit-Wolf shrinkage for the covariance in the
    variance term and ``shrink_norm`` (default: same as ``shrink``) for the
    ``Sigma`` in the Mahalanobis norm.
    """
    if shrink_norm is None:
        shrink_norm = shrink
    F = cov_factor(R, shrink)
    mean_ = R.mean().values
    m = R.shape[1]
    x = cp.Variable(m)
    name = "blanchet_mvo_%s%s" % (norm, "_shrink" if shrink else "")
    if norm == "mahalanobis":
        name += "_shrinknorm" if shrink_norm else ""
        F_norm = F if shrink_norm == shrink else cov_factor(R, shrink_norm)
        sigma = np.sqrt(np.sum(F_norm ** 2) / m)
        dual_norm = cp.norm(F_norm @ x, 2) / sigma
    elif norm in ("l1", "l2"):
        dual_norm = cp.norm(x, int(norm[1]))
    else:
        raise ValueError("unknown norm %r" % norm)
    penalty = np.sqrt(delta_val) * dual_norm

    constraints = [x >= 0,
                   cp.sum(x) == 1,
                   mean_ @ x >= alpha_val + penalty]
    objective = cp.Minimize(cp.norm(F @ x, 2) + penalty)
    return _solve_or_equal_weight(cp.Problem(objective, constraints), x, name)


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
    """Table of drawdown and return statistics for each wealth series in ``Results``."""
    Stats = {}
    for key, Wealth in Results.items():
        returns = Wealth.pct_change().dropna(axis=0)
        mdd, start, end = max_dd(returns)
        AR, SD, SR = RetStats(returns)
        Stats[key] = [mdd, start, end, AR, SD, SR]
    Out = pd.DataFrame(Stats).transpose().sort_index()
    Out.columns = ['mdd', 'start', 'end', 'AR', 'SD', 'SR']
    return Out
