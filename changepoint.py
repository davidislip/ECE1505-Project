"""Distributionally robust change point detection.

Robust hypothesis test (Gao et al.) combined with a CUSUM stopping rule
(Cao et al.). See Data_ChangeDetection.ipynb for the derivations.

Index conventions: ``R`` is the returns frame ``P.pct_change().dropna()``, so
``R.iloc[i]`` is the return from ``P.iloc[i]`` to ``P.iloc[i + 1]`` and is only
known at price index ``i + 1``. Everything in this module is in R indices;
``retrieve_rebalancing_info`` converts rebalancing times to P indices.
"""
import os
import pickle as pkl
import warnings

import cvxpy as cp
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


def robust_hypothesis(t, n, theta_val, R, rand=False, rng=None):
    """Solve the Wasserstein robust hypothesis test around time ``t``.

    Q_1 is the ``n`` returns before ``t`` and Q_2 the ``n`` returns after
    ``t`` (``R[t]`` itself is in neither sample).
    """
    values = np.asarray(R)
    Q_1 = values[t - n:t, :]
    Q_2 = values[t + 1:t + n + 1, :]
    n_1 = len(Q_1)
    n_2 = len(Q_2)

    if rand:
        # bootstrap rows (np.random.choice only accepts 1-D arrays)
        rng = np.random.default_rng() if rng is None else rng
        Q_1 = values[rng.integers(0, len(values), size=n_1)]
        Q_2 = values[rng.integers(0, len(values), size=n_2)]

    Q = np.concatenate((Q_1, Q_2))

    A = euclidean_distances(Q, Q)
    b_1 = np.concatenate((1 / n_1 * np.ones(n_1), np.zeros(n_2)))
    b_2 = np.concatenate((np.zeros(n_1), 1 / n_2 * np.ones(n_2)))

    ## Construct the problem.
    p1 = cp.Variable(n_1 + n_2)
    p2 = cp.Variable(n_1 + n_2)

    gamma_1 = cp.Variable((n_1 + n_2, n_1 + n_2))
    gamma_2 = cp.Variable((n_1 + n_2, n_1 + n_2))

    constraints = [cp.sum(cp.multiply(gamma_1, A)) <= theta_val,
                   cp.sum(cp.multiply(gamma_2, A)) <= theta_val,
                   cp.sum(gamma_1, axis=1) == b_1,
                   cp.sum(gamma_2, axis=1) == b_2,
                   cp.sum(gamma_1, axis=0) == p1,
                   cp.sum(gamma_2, axis=0) == p2,
                   p1 >= 0,
                   p2 >= 0,
                   gamma_1 >= 0,
                   gamma_2 >= 0]

    # (p1+p2)*psi(p1/(p1+p2)) with hinge loss psi(p) = 2*min(p, 1-p)
    objective = cp.Minimize(-2 * cp.sum(cp.minimum(p1, p2)))

    prob = cp.Problem(objective, constraints)
    try:
        prob.solve()
    except cp.SolverError:
        prob.solve(solver=cp.SCS, max_iters=20000)
    if p1.value is None or prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError("robust_hypothesis: solver failed at t=%d (status %s)" % (t, prob.status))
    return [p1, p2, gamma_1, gamma_2, prob, Q]


def detector(p1, p2, tol=1e-3):
    """phi = sign(p1 - p2), with differences below ``tol * max|p1 - p2|`` set to 0."""
    signal = p1.value - p2.value
    signal[np.abs(signal) < tol * np.max(np.abs(signal))] = 0
    return np.sign(signal)


def detect_change(phi, b):
    """CUSUM stopping time.

    Returns ``(T, S)`` where ``S[t] = max(0, max_{k<t} sum(-phi[k:t]))`` and
    ``T`` is the first ``t`` with ``S[t] >= b``, or ``None`` if the statistic
    never crosses ``b``. ``S`` is left at zero after ``T``.
    """
    S = np.zeros(len(phi))
    for t in range(1, len(phi)):
        S[t] = max(0.0, S[t - 1] - phi[t - 1])
        if S[t] >= b:
            return t, S
    return None, S


def calculate_ARL(p1, p2, phi, b, plots=False, err=1e-9):
    """Lower bound on the average run length of the CUSUM detector with threshold ``b``.

    The first M terms of the series are summed exactly and the remainder is
    bounded by the geometric series in mu.
    """
    p1_val = p1.value
    mu = np.dot(p1_val, np.maximum(1 - phi, 0))
    if mu >= 1:
        return 0.0
    if mu <= 0:
        # every term is <= mu**k = 0 so the false alarm probability is zero
        return np.inf

    M = max(1, int(np.ceil(np.log(err * (1 - mu)) / np.log(mu))))
    k = np.arange(1, M + 1)
    terms = np.maximum(1 - phi[None, :] - b / k[:, None], 0) @ p1_val
    l = terms ** k
    if plots:
        import matplotlib.pyplot as plt
        plt.plot(k, l)
    sum_ = l.sum() + mu ** (M + 1) / (1 - mu)
    return 1 / sum_


def get_list_of_trade_times(theta_val, ARL_target, n, R, limit=None):
    """Slide the robust test + CUSUM over ``R`` and collect detected changes.

    Returns a dict with, for each detected change, the time of the end of the
    detection window (``times``, R index), the CUSUM stopping time within the
    window (``Ts``) and the ARL of the chosen threshold (``ARL``). Windows where
    no change was detected are not stored.
    """
    if limit is None:
        limit = 2 * int(len(R) / n)

    t_mid = n + 1
    t_mids = []
    Ts = []
    ARLs = []
    balance = []
    (n1, n2) = (n, n)
    k = 0
    while t_mid <= len(R) - n2 - 1 and k < limit:
        t_mids.append(t_mid)
        t_now = t_mid + n2
        #single step change point algorithm
        [p1, p2, gamma_1, gamma_2, prob, Q] = robust_hypothesis(t_mid, n, theta_val, R)
        phi = detector(p1, p2)

        for b in range(1, n):
            arl = calculate_ARL(p1, p2, phi, b)
            if arl > ARL_target:
                break

        T, S = detect_change(phi, b)
        if T is None:
            #no change in this window
            t_mid = t_mid + n1 + n2 - 1
        else:
            t_mid = t_now + T - n2
            balance.append(t_now)
            Ts.append(T)
            ARLs.append(arl)

        k = k + 1

    terminated = t_mid > len(R) - n2 - 1
    if not terminated:
        warnings.warn("get_list_of_trade_times hit limit=%d before reaching the end of the series" % limit)
    return {"times": balance, "mid_points": t_mids, "Ts": Ts, "ARL": ARLs,
            "detected_only": True, "terminated": terminated}


def results_filename(theta, arl_t, n, directory="Results"):
    return os.path.join(directory, "%s_%s_%s.pkl" % (theta, arl_t, n))


def retrieve_rebalancing_info(filename, n_prices, min_estimation=None):
    """Turn a saved result from ``get_list_of_trade_times`` into a rebalancing schedule.

    ``n_prices`` is ``len(P)``. ``min_estimation`` is the minimum number of
    returns after the estimated change point used to estimate the new
    distribution (defaults to the window size ``n``).

    Returns a dict with

    * ``res_times``: end of each accepted detection window (R index)
    * ``post_detections``: CUSUM stopping time within the window
    * ``post_detection_times``: estimated change point (R index), the start of
      the estimation window
    * ``res_times_updated``: rebalancing times (P index); ``R.iloc[est:rebal]``
      only uses returns known at ``P.iloc[rebal]``
    * ``post_detection_times_updated``: estimation window starts matching
      ``res_times_updated``
    """
    with open(filename, 'rb') as f:
        res = pkl.load(f)

    theta, arl_t, n = [float(x) for x in os.path.basename(filename)[:-len(".pkl")].split("_")]
    n = int(n)
    if min_estimation is None:
        min_estimation = n

    if not res.get("terminated", True):
        warnings.warn("%s: the change point search did not reach the end of the series" % filename)

    res_times = np.array(res['times'], dtype=int)
    post_detections = np.array(res['Ts'], dtype=int)
    keep = np.array(res['ARL']) > arl_t
    if not res.get("detected_only", False):
        # older result files stored every window; T == 2n - 1 meant no detection
        keep &= post_detections < 2 * n - 1
    res_times = res_times[keep]
    post_detections = post_detections[keep]

    # the window is R[t_now-2n : t_now-n] + R[t_now-n+1 : t_now+1] and CUSUM
    # stops after seeing window entry T-1
    last_seen = post_detections - 1
    post_detection_times = res_times - 2 * n + last_seen + (last_seen >= n)

    # R[res_times] is known at P[res_times + 1]; wait for min_estimation returns
    rebal = np.maximum(res_times + 1, post_detection_times + min_estimation)

    # skip a rebalance if another change is detected before it can happen
    detected_before = np.append(rebal[:-1] > res_times[1:] + 1, False)
    ok = ~detected_before & (rebal < n_prices)
    return {"res": res, "theta": theta, "arl_t": arl_t, "n": n,
            "res_times": res_times,
            "post_detections": post_detections,
            "post_detection_times": post_detection_times,
            "res_times_updated": rebal[ok],
            "post_detection_times_updated": post_detection_times[ok]}
