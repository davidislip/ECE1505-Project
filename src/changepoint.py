"""
Wasserstein-based changepoint detection.

References:
    Cao et al., "Robust Sequential Change-Point Detection by Convex Optimization"
"""

import warnings
import numpy as np
import cvxpy as cp
from sklearn.metrics.pairwise import euclidean_distances


def robust_hypothesis(t, n, theta_val, R):
    """
    Wasserstein hypothesis test comparing return distributions before and after
    candidate changepoint t.

    Parameters
    ----------
    t : int
        Candidate changepoint index.
    n : int
        Look-back / look-forward window size (e.g. 30, 60, 90 days).
    theta_val : float
        Wasserstein ball radius θ (e.g. 0.05, 0.1, 0.5).
    R : pd.DataFrame
        Return matrix of shape (T, N).

    Returns
    -------
    list : [p1, p2, gamma_1, gamma_2, prob, Q]
        cvxpy variables and the stacked sample matrix Q.
    """
    Q_1 = R.values[t - n:t, :]
    Q_2 = R.values[t + 1:t + n + 1, :]
    n_1 = len(Q_1)
    n_2 = len(Q_2)

    Q = np.concatenate((Q_1, Q_2))
    A = euclidean_distances(Q, Q)

    b_1 = np.concatenate((1 / n_1 * np.ones(n_1), np.zeros(n_2)))
    b_2 = np.concatenate((np.zeros(n_2), 1 / n_2 * np.ones(n_2)))

    theta = cp.Parameter()
    p1 = cp.Variable(n_1 + n_2)
    p2 = cp.Variable(n_1 + n_2)
    gamma_1 = cp.Variable((n_1 + n_2, n_1 + n_2))
    gamma_2 = cp.Variable((n_1 + n_2, n_1 + n_2))

    theta.value = theta_val

    constraints = [
        cp.sum(cp.multiply(gamma_1, A)) <= theta,
        cp.sum(cp.multiply(gamma_2, A)) <= theta,
        cp.sum(gamma_1, axis=1) == b_1,
        cp.sum(gamma_2, axis=1) == b_2,
        cp.sum(gamma_1, axis=0) == p1,
        cp.sum(gamma_2, axis=0) == p2,
        p1 >= 0,
        p2 >= 0,
        gamma_1 >= 0,
        gamma_2 >= 0,
    ]

    objective = cp.Minimize(-2 * cp.sum(cp.minimum(p1, p2)))
    prob = cp.Problem(objective, constraints)

    try:
        prob.solve()
    except Exception:
        try:
            prob.solve(verbose=False, solver=cp.SCS, max_iters=20000)
        except Exception:
            pass

    if prob.status not in ("optimal", "optimal_inaccurate"):
        warnings.warn(
            f"robust_hypothesis solver status={prob.status} at t={t}. "
            "Results may be unreliable."
        )

    return [p1, p2, gamma_1, gamma_2, prob, Q]


def detect_change(phi, b):
    """
    CUSUM-style sequential detection over signal phi.

    Parameters
    ----------
    phi : np.ndarray
        Sign of distributional shift (output of robust_hypothesis).
    b : float
        Detection threshold.

    Returns
    -------
    t : int
        Detection time (index when cumulative sum first exceeds b).
    S : np.ndarray
        CUSUM statistic at each time step.
    """
    S = np.zeros_like(phi)
    for t in range(len(phi)):
        max_sum = 0
        for k in range(t):
            if np.sum(-1 * phi[k:t]) > max_sum:
                max_sum = np.sum(-1 * phi[k:t])
        S[t] = max_sum
        if max_sum >= b:
            break
    return t, S


def calculate_ARL(p1, p2, phi, b, plots=False):
    """
    Compute the Average Run Length (ARL) for the given threshold b.

    Used to calibrate b so that false-alarm rate matches a target ARL
    (typically 1000 or 10000 days).

    Parameters
    ----------
    p1, p2 : cp.Variable
        Solved transport plan variables from robust_hypothesis.
    phi : np.ndarray
        Detection signal.
    b : float
        Detection threshold.
    plots : bool
        If True, plot the ARL series terms (requires matplotlib).

    Returns
    -------
    float
        Estimated ARL for this threshold.
    """
    import matplotlib.pyplot as plt  # optional, only used when plots=True

    l = []
    sum_ = 0
    k = 1
    err = 1e-9

    mu = np.dot(p1.value, np.maximum(1 - np.sign(p1.value - p2.value), 0))
    if mu >= 1:
        return 0

    M = np.round(np.log(err * (1 - mu)) / np.log(mu))

    while k <= M:
        term = np.dot(p1.value, np.maximum(1 - phi - b / k, 0))
        l.append(term ** k)
        sum_ += term ** k
        k += 1

    if plots:
        plt.plot(range(1, k), l)

    if err / sum_ > 0.01:
        print("Adjusted due to relative error")
        sum_ += (mu ** M) / (1 - mu)

    return 1 / sum_


def get_list_of_trade_times(R, theta_val, ARL_target, n, limit=None):
    """
    Run sequential changepoint detection over the full return series R.

    Slides a window of size n through R, running robust_hypothesis at each
    candidate changepoint, calibrating the detection threshold b to meet
    ARL_target, then recording rebalancing times.

    Parameters
    ----------
    R : pd.DataFrame
        Return matrix (T × N).
    theta_val : float
        Wasserstein ball radius θ.
    ARL_target : float
        Target average run length (e.g. 1000 or 10000).
    n : int
        Window size (look-back / look-forward).
    limit : int, optional
        Maximum number of iterations. Defaults to 2 * (T // n).

    Returns
    -------
    dict with keys:
        "times"       – rebalancing indices
        "mid_points"  – candidate changepoint indices tested
        "Ts"          – detection times within each window
        "ARL"         – calibrated ARL values at each rebalancing point
    """
    if limit is None:
        limit = 2 * int(len(R) / n)

    t_mid = n + 1
    t_mids = []
    Ts = []
    ARLs = []
    balance = []
    k = 0

    while t_mid <= len(R) - n - 1 and k < limit:
        t_mids.append(t_mid)
        t_now = t_mid + n

        [p1, p2, gamma_1, gamma_2, prob, Q] = robust_hypothesis(t_mid, n, theta_val, R)
        signal = p1.value - p2.value
        signal[np.abs(signal) < 1e-3 * np.max(signal)] = 0
        phi = np.sign(signal)

        b = 1
        for b in range(1, n):
            arl = calculate_ARL(p1, p2, phi, b, plots=False)
            if arl > ARL_target:
                break

        T, S = detect_change(phi, b)
        t_mid = t_now + T - n

        if T <= n + n:
            balance.append(t_now)
            Ts.append(T)
            ARLs.append(arl)

        k += 1
        if k == limit:
            print("Too many rebalancing points")

    return {"times": balance, "mid_points": t_mids, "Ts": Ts, "ARL": ARLs}
