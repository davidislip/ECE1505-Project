"""Run the change point search over the parameter grid and save each result to Results/."""
import itertools
import os
import pickle as pkl
import sys

from changepoint import get_list_of_trade_times, results_filename
from data import load_prices

ns = [30, 60, 90]
theta_vals = [0.05, 0.075, 0.1, 0.2, 0.5, 1]
ave_run_length = [1000, 10000]

if __name__ == "__main__":
    P, R = load_prices()
    os.makedirs("Results", exist_ok=True)
    for (theta, arl_t, n) in itertools.product(theta_vals, ave_run_length, ns):
        res = get_list_of_trade_times(theta, arl_t, n, R)
        with open(results_filename(theta, arl_t, n), 'wb') as f:
            pkl.dump(res, f)
        print(theta, arl_t, n, "detections:", len(res["times"]), flush=True)
        sys.stdout.flush()
