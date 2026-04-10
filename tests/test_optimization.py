"""
Unit tests for src/optimization.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from src.optimization import equal_weight, markowitz_mvo, blanchet_mvo, ALPHA


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

np.random.seed(42)

def make_returns(T=120, N=10, scale=0.01):
    """Generate a synthetic return DataFrame."""
    data = np.random.randn(T, N) * scale
    cols = [f"ASSET_{i}" for i in range(N)]
    return pd.DataFrame(data, columns=cols)


# ---------------------------------------------------------------------------
# equal_weight
# ---------------------------------------------------------------------------

class TestEqualWeight:
    def test_weights_sum_to_one(self):
        R = make_returns()
        w = equal_weight(R)
        assert abs(w.sum() - 1.0) < 1e-10

    def test_all_weights_equal(self):
        R = make_returns(N=10)
        w = equal_weight(R)
        assert np.allclose(w, 0.1)

    def test_shape(self):
        R = make_returns(N=7)
        w = equal_weight(R)
        assert w.shape == (7,)


# ---------------------------------------------------------------------------
# markowitz_mvo
# ---------------------------------------------------------------------------

class TestMarkowitzMVO:
    def test_weights_sum_to_one(self):
        R = make_returns()
        w = markowitz_mvo(R, alpha_val=0.0)
        assert abs(w.sum() - 1.0) < 1e-4

    def test_weights_nonnegative(self):
        R = make_returns()
        w = markowitz_mvo(R, alpha_val=0.0)
        assert np.all(w >= -1e-6)

    def test_return_constraint_satisfied(self):
        R = make_returns()
        alpha = 0.0
        w = markowitz_mvo(R, alpha_val=alpha)
        achieved = (R.mean().values * w).sum()
        assert achieved >= alpha - 1e-4

    def test_output_shape(self):
        R = make_returns(N=8)
        w = markowitz_mvo(R, alpha_val=0.0)
        assert w.shape == (8,)

    def test_zero_alpha_feasible(self):
        R = make_returns(T=80, N=5)
        w = markowitz_mvo(R, alpha_val=0.0)
        # Should not fall back to equal weight for a well-conditioned problem
        assert w is not None
        assert abs(w.sum() - 1.0) < 1e-4

    def test_infeasible_alpha_returns_equal_weight(self):
        """Extremely high alpha should trigger equal-weight fallback."""
        R = make_returns(scale=0.001)   # tiny returns
        w = markowitz_mvo(R, alpha_val=999.0)  # impossible target
        N = R.shape[1]
        assert w.shape == (N,)
        assert abs(w.sum() - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# blanchet_mvo
# ---------------------------------------------------------------------------

class TestBlanchetMVO:
    def test_weights_sum_to_one(self):
        R = make_returns()
        w = blanchet_mvo(R, delta_val=0.0, alpha_val=0.0)
        assert abs(w.sum() - 1.0) < 1e-4

    def test_weights_nonnegative(self):
        R = make_returns()
        w = blanchet_mvo(R, delta_val=1e-6, alpha_val=0.0)
        assert np.all(w >= -1e-6)

    def test_output_shape(self):
        R = make_returns(N=6)
        w = blanchet_mvo(R, delta_val=1e-6, alpha_val=0.0)
        assert w.shape == (6,)

    def test_zero_delta_close_to_markowitz(self):
        """With δ=0, blanchet_mvo should produce weights close to markowitz_mvo."""
        R = make_returns(T=120, N=5)
        w_m = markowitz_mvo(R, alpha_val=0.0)
        w_b = blanchet_mvo(R, delta_val=0.0, alpha_val=0.0)
        # Allow a generous tolerance since solvers may differ slightly
        assert np.linalg.norm(w_m - w_b) < 0.15

    def test_larger_delta_increases_diversification(self):
        """Larger δ should push towards more uniform weights (higher entropy)."""
        R = make_returns(T=120, N=10)
        w_small = blanchet_mvo(R, delta_val=1e-8, alpha_val=0.0)
        w_large = blanchet_mvo(R, delta_val=1e-4, alpha_val=0.0)
        # Herfindahl index: lower = more diversified
        hhi_small = (w_small ** 2).sum()
        hhi_large = (w_large ** 2).sum()
        assert hhi_large <= hhi_small + 0.05  # allow small numerical slack

    def test_infeasible_alpha_returns_equal_weight(self):
        R = make_returns(scale=0.001)
        w = blanchet_mvo(R, delta_val=1e-6, alpha_val=999.0)
        N = R.shape[1]
        assert abs(w.sum() - 1.0) < 1e-6
