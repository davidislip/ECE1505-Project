# ECE 1505 Project: Distributionally Robust Change Point Detection for Portfolio Rebalancing

David Islip and Gurjot Dhaliwal. The report is `ECE_1505_Project_David_Islip_Gurjot_Dhaliwal.pdf` (appendix in `Appendix.pdf`).

Rebalancing times are chosen by a Wasserstein robust hypothesis test (Gao et al.) combined with a CUSUM stopping rule (Cao et al.). The detected times are then used to rebalance equal-weight, mean-variance (MVO), and Wasserstein distributionally robust mean-variance (DRMVO, Blanchet et al.) portfolios of S&P 500 stocks from January 2012 to March 2020, and compared to fixed-interval rebalancing.

## Layout

| Path | Contents |
| --- | --- |
| `Data_ChangeDetection.ipynb` | Main notebook: data, derivations, change point detection, backtests |
| `Plots.ipynb` | Rebalancing-times figure for the report (`Images/cp.png`) |
| `changepoint.py` | Robust hypothesis test, CUSUM detector, ARL bound, rebalancing schedule |
| `backtest.py` | Portfolio strategies (equal weight, MVO, DRMVO with l2 or Mahalanobis transport cost, sample or Ledoit-Wolf covariance), backtest, summary statistics |
| `data.py` | Loads and cleans `stocks.pkl` |
| `generate_results.py` | Runs the change point search over the parameter grid into `Results/` |
| `Results/` | One pickle per `theta_ARLtarget_n` combination |
| `Experimental Outputs/` | Backtest results and summary tables |
| `stocks.pkl` | Yahoo Finance prices for the S&P 500 constituents (as of 2020) |

## Running

```
pip install -r requirements.txt
python generate_results.py          # regenerates Results/ (optional, they are committed)
jupyter notebook Data_ChangeDetection.ipynb
```

## Conventions

`R = P.pct_change().dropna()`, so `R.iloc[i]` is the return from `P.iloc[i]` to `P.iloc[i+1]` and is only known at price index `i+1`. Change points are R indices; rebalancing times are P indices, and the weights at `P.iloc[t]` are estimated from `R.iloc[est:t]`, so the backtest does not look ahead.

## Robust portfolio variants

`blanchet_mvo` supports two transport costs:

- `norm="l2"`: the Euclidean cost.
- `norm="mahalanobis"`: cost $(u-v)^\top \bar\sigma^2\Sigma^{-1}(u-v)$, which makes the robustness penalty $\sqrt{\delta}\sqrt{x^\top\Sigma x}/\bar\sigma$.

`shrink` (variance term) and `shrink_norm` (Mahalanobis norm) each choose between the sample covariance and the Ledoit-Wolf estimate. The derivation is in the notebook.

## Caveats

- The universe is the 2020 S&P 500 constituents with complete price history, so the backtest has survivorship bias.
- The ARL bound multiplies single-observation hinge-loss bounds across observations. That step is exact for the exponential loss (Chernoff), but for the hinge loss it is a heuristic rather than a proven bound.
- Sharpe ratios assume a zero risk-free rate.
