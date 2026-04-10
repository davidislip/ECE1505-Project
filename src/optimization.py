"""
Portfolio optimization strategies.

References:
    Markowitz (1952) — Mean-Variance Optimization
    Blanchet et al.  — Distributionally Robust MVO with Wasserstein ambiguity
"""

import warnings
import numpy as np
import cvxpy as cp
import scipy.linalg as la

# Target annualized return used throughout the project
TARGET_ANNUAL_RETURN = 0.14        # 14%
TRADING_DAYS_PER_YEAR = 250
ALPHA = TARGET_ANNUAL_RETURN / TRADING_DAYS_PER_YEAR  # daily equivalent


def equal_weight(R):
    """Return a uniform 1/N allocation across all assets."""
    N = R.shape[1] if hasattr(R, "shape") else len(R.columns)
    return np.ones(N) / N


def markowitz_mvo(R, alpha_val=ALPHA):
    """
    Standard Markowitz minimum-variance portfolio (long-only).

    Solves:
        min  x' Σ x
        s.t. sum(x) = 1
             x >= 0
             mean(R) @ x >= alpha_val

    Parameters
    ----------
    R : pd.DataFrame
        Return matrix (T × N).
    alpha_val : float
        Minimum required daily return (default: 14% / 250).

    Returns
    -------
    np.ndarray
        Portfolio weights of shape (N,).
    """
    cov_ = R.cov().values
    mean_ = R.mean().values
    (T, m) = R.shape

    x = cp.Variable(m)
    alpha = cp.Parameter()
    alpha.value = alpha_val

    constraints = [
        x >= 0,
        cp.sum(x) == 1,
        cp.sum(cp.multiply(x, mean_)) >= alpha,
    ]
    objective = cp.Minimize(cp.quad_form(x, cov_))
    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=False)

    if prob.status not in ("optimal", "optimal_inaccurate") or x.value is None:
        warnings.warn(
            f"markowitz_mvo: solver status={prob.status}. Falling back to equal weights."
        )
        return equal_weight(R)

    return x.value


def blanchet_mvo(R, delta_val, alpha_val=ALPHA):
    """
    Distributionally Robust MVO over a Wasserstein ambiguity ball.

    Solves:
        min  (‖Σ^½ x‖₂ + √δ ‖x‖₁)²
        s.t. sum(x) = 1
             x >= 0
             mean(R) @ x >= alpha_val + √δ ‖x‖₁

    Parameters
    ----------
    R : pd.DataFrame
        Return matrix (T × N).
    delta_val : float
        Wasserstein ambiguity radius δ. Set to 0 to recover standard MVO.
    alpha_val : float
        Minimum required daily return.

    Returns
    -------
    np.ndarray
        Portfolio weights of shape (N,).
    """
    cov_ = R.cov()
    mean_ = R.mean().values
    cov_half = la.sqrtm(cov_)
    (T, m) = R.shape

    x = cp.Variable(m)
    delta = cp.Parameter(nonneg=True)
    alpha = cp.Parameter(nonneg=True)
    delta.value = delta_val
    alpha.value = alpha_val

    constraints = [
        x >= 0,
        cp.sum(x) == 1,
        cp.sum(cp.multiply(x, mean_)) >= alpha + (delta ** 0.5) * cp.norm(x, 1),
    ]
    objective = cp.Minimize(
        cp.square(cp.norm(cp.matmul(cov_half, x), 2) + (delta ** 0.5) * cp.norm(x, 1))
    )
    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=False)

    if prob.status not in ("optimal", "optimal_inaccurate") or x.value is None:
        warnings.warn(
            f"blanchet_mvo: solver status={prob.status} (delta={delta_val:.4f}). "
            "Falling back to equal weights."
        )
        return equal_weight(R)

    return x.value
