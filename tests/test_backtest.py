"""
Unit tests for src/backtest.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from src.backtest import (
    apply_transaction_costs,
    test_fixed_strategy as run_fixed_strategy,
    test_variable_strategy as run_variable_strategy,
    max_dd,
    RetStats,
)
from src.optimization import equal_weight, markowitz_mvo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

np.random.seed(0)

def make_prices(T=200, N=5, start=100.0):
    """Geometric random walk prices."""
    returns = np.random.randn(T, N) * 0.01
    prices = start * np.exp(np.cumsum(returns, axis=0))
    idx = pd.date_range("2015-01-01", periods=T, freq="B")
    return pd.DataFrame(prices, index=idx, columns=[f"A{i}" for i in range(N)])


def make_returns_from_prices(P):
    return np.log(P).diff().dropna()


# ---------------------------------------------------------------------------
# apply_transaction_costs
# ---------------------------------------------------------------------------

class TestTransactionCosts:
    def test_zero_turnover_no_cost(self):
        w = np.array([0.25, 0.25, 0.25, 0.25])
        val = apply_transaction_costs(w, w, portfolio_value=1.0, cost_bps=10)
        assert abs(val - 1.0) < 1e-10

    def test_full_turnover_reduces_value(self):
        old = np.array([1.0, 0.0, 0.0, 0.0])
        new = np.array([0.0, 1.0, 0.0, 0.0])
        val = apply_transaction_costs(old, new, portfolio_value=1.0, cost_bps=10)
        # turnover = 2.0 (sell 1, buy 1); cost = 2 * 10/10000 = 0.002
        assert abs(val - (1.0 - 2 * 10 / 10_000)) < 1e-10

    def test_cost_scales_with_bps(self):
        old = np.array([0.5, 0.5])
        new = np.array([1.0, 0.0])
        v10 = apply_transaction_costs(old, new, 1.0, cost_bps=10)
        v20 = apply_transaction_costs(old, new, 1.0, cost_bps=20)
        assert v20 < v10


# ---------------------------------------------------------------------------
# test_fixed_strategy
# ---------------------------------------------------------------------------

class TestFixedStrategy:
    def setup_method(self):
        P = make_prices(T=200, N=5)
        R = make_returns_from_prices(P)
        self.P = P.iloc[1:]   # align with returns
        self.R = R

    def test_returns_three_items(self):
        h, w, W = run_fixed_strategy(30, self.R, self.P, equal_weight, ())
        assert h.shape[0] == self.R.shape[1]
        assert w.shape[0] == self.R.shape[1]
        assert len(W) == len(self.P)

    def test_wealth_starts_near_one(self):
        _, _, W = run_fixed_strategy(30, self.R, self.P, equal_weight, ())
        assert abs(W.iloc[0] - 1.0) < 0.05

    def test_wealth_is_positive(self):
        _, _, W = run_fixed_strategy(30, self.R, self.P, equal_weight, ())
        assert (W > 0).all()

    def test_weights_nonnegative(self):
        _, weights, _ = run_fixed_strategy(30, self.R, self.P, equal_weight, ())
        assert np.all(weights >= -1e-8)


# ---------------------------------------------------------------------------
# test_variable_strategy
# ---------------------------------------------------------------------------

class TestVariableStrategy:
    def setup_method(self):
        P = make_prices(T=200, N=5)
        R = make_returns_from_prices(P)
        self.P = P.iloc[1:]
        self.R = R

    def _make_rebal_est(self):
        rebal = [40, 80, 120, 160]
        est = [10, 50, 90, 130]
        return rebal, est

    def test_basic_run(self):
        rebal, est = self._make_rebal_est()
        h, w, W = run_variable_strategy(
            self.R, self.P, rebal, est, equal_weight, ()
        )
        assert len(W) == len(self.P)

    def test_wealth_positive(self):
        rebal, est = self._make_rebal_est()
        _, _, W = run_variable_strategy(
            self.R, self.P, rebal, est, equal_weight, ()
        )
        assert (W > 0).all()

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            run_variable_strategy(self.R, self.P, [40, 80], [10], equal_weight, ())

    def test_look_ahead_bias_detected(self):
        """est > rebal should trigger assertion error."""
        with pytest.raises(AssertionError, match="Look-ahead bias"):
            run_variable_strategy(
                self.R, self.P,
                rebal=[40],
                est=[50],    # est AFTER rebal — look-ahead
                strategy=equal_weight,
                function_params=(),
            )


# ---------------------------------------------------------------------------
# max_dd
# ---------------------------------------------------------------------------

class TestMaxDD:
    def test_flat_returns_zero_drawdown(self):
        returns = pd.Series(np.zeros(100))
        mdd, _, _ = max_dd(returns)
        assert abs(mdd) < 1e-10

    def test_always_positive_returns_zero_drawdown(self):
        returns = pd.Series(np.ones(50) * 0.001)
        mdd, _, _ = max_dd(returns)
        assert mdd >= -1e-10

    def test_single_crash_drawdown(self):
        returns = pd.Series([0.0] * 50 + [-0.5] + [0.0] * 50)
        mdd, _, _ = max_dd(returns)
        assert mdd < -0.4

    def test_drawdown_is_negative(self):
        returns = pd.Series(np.random.randn(100) * 0.01)
        mdd, _, _ = max_dd(returns)
        assert mdd <= 0


# ---------------------------------------------------------------------------
# RetStats
# ---------------------------------------------------------------------------

class TestRetStats:
    def test_zero_returns(self):
        returns = pd.Series(np.zeros(250))
        ar, sd, sr = RetStats(returns)
        assert abs(ar) < 1e-10

    def test_positive_mean_positive_sharpe(self):
        returns = pd.Series(np.ones(250) * 0.001)
        ar, sd, sr = RetStats(returns)
        assert ar > 0

    def test_annualization_factor(self):
        daily_mean = 0.001
        returns = pd.Series(np.ones(250) * daily_mean)
        ar, _, _ = RetStats(returns)
        assert abs(ar - 250 * daily_mean) < 1e-8
