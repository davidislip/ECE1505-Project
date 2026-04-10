# ECE 1505 — Dynamic Portfolio Allocation via Wasserstein Changepoint Detection

**Authors:** David Islip & Gurjot Dhaliwal  
**Course:** ECE 1505 — Convex Optimization, University of Toronto

---

## Overview

This project implements a dynamic portfolio allocation strategy for S&P 500 equities.
Instead of rebalancing on a fixed calendar schedule, it detects structural breaks
(changepoints) in the return distribution using a Wasserstein-based hypothesis test
and rebalances only when the market regime changes.

Two portfolio optimization strategies are compared:

| Strategy | Description |
|---|---|
| **MVO** | Standard Markowitz minimum-variance (long-only) |
| **DRMVO** | Distributionally Robust MVO over a Wasserstein ambiguity ball (Blanchet et al.) |

Best result: **16.0% annualized return, Sharpe 1.24, max drawdown −21%**
(θ=0.1, n=30, ARL=10000, MVO strategy).

---

## Repository Structure

```
ECE1505-Project/
├── src/                         # Core algorithm library
│   ├── changepoint.py           # robust_hypothesis, detect_change, calculate_ARL
│   ├── optimization.py          # equal_weight, markowitz_mvo, blanchet_mvo
│   ├── backtest.py              # test_fixed_strategy, test_variable_strategy, RetStats, max_dd
│   └── data.py                  # load_stocks, compute_returns, load_results
├── tests/                       # pytest unit tests
│   ├── test_optimization.py
│   └── test_backtest.py
├── Data_ChangeDetection.ipynb   # Main pipeline notebook
├── Plots.ipynb                  # Visualization and sensitivity analysis
├── Results/                     # Serialized changepoint detection results (28 configs)
├── Experimental Outputs/        # CSV/XLSX performance summaries
├── Images/                      # Figures used in the report
├── requirements.txt
└── ECE_1505_Project_*.pdf       # Final report
```

> `stocks.pkl` (~49 MB, S&P 500 OHLCV Jan 2012–Mar 2020) is excluded from git.
> See **Data** section below to obtain it.

---

## Installation

```bash
pip install -r requirements.txt
```

Python 3.10+ recommended.

---

## Running the Notebooks

1. Ensure `stocks.pkl` is present in the project root (see **Data** below).
2. Run `Data_ChangeDetection.ipynb` top-to-bottom. Results are saved to `Results/`.
3. Run `Plots.ipynb` to reproduce all figures and performance tables.

---

## Running the Tests

```bash
pytest tests/ -v
```

33 tests covering:
- `equal_weight`, `markowitz_mvo`, `blanchet_mvo` — weight constraints, feasibility, fallback behaviour
- `test_fixed_strategy`, `test_variable_strategy` — portfolio value tracking, look-ahead bias detection
- `max_dd`, `RetStats` — drawdown and performance statistics

---

## Using the `src` Module

The notebooks import from `src/` directly. You can also use the library standalone:

```python
import pandas as pd
from src.data import load_stocks, compute_returns
from src.optimization import markowitz_mvo
from src.backtest import test_fixed_strategy, RetStats

# Load data
raw = load_stocks("stocks.pkl")
R, P = compute_returns(raw["Adj Close"])

# Run a fixed rebalancing backtest
alpha = 0.14 / 250
_, _, wealth = test_fixed_strategy(44, R, P, markowitz_mvo, (alpha,))

returns = wealth.pct_change().dropna()
ann_ret, ann_std, sharpe = RetStats(returns)
print(f"AR={ann_ret:.1%}  SR={sharpe:.2f}")
```

---

## Algorithm Summary

### Changepoint Detection

`robust_hypothesis(t, n, θ, R)` solves a Wasserstein optimal transport problem
comparing the empirical return distributions in a window of size `n` before and
after candidate changepoint `t`. The Wasserstein ball radius `θ` controls
sensitivity to distributional shift.

`detect_change(φ, b)` applies a CUSUM accumulator over the sign signal `φ`.
`calculate_ARL(p1, p2, φ, b)` calibrates threshold `b` to a target Average Run
Length (ARL = 1000 or 10000 days).

**Parameter sweep:** 6 θ values × 2 ARL targets × 2 window sizes = 24 configurations.

### Portfolio Optimization

**MVO** (Markowitz):
```
min  x' Σ x
s.t. 1'x = 1,  x ≥ 0,  μ'x ≥ α
```

**DRMVO** (Blanchet et al.):
```
min  (‖Σ^½ x‖₂ + √δ ‖x‖₁)²
s.t. 1'x = 1,  x ≥ 0,  μ'x ≥ α + √δ ‖x‖₁
```

δ values tested: `{0, (0.25α)², (0.5α)², α², (1.5α)²}`

---

## Data

The dataset is S&P 500 daily OHLCV data from Yahoo Finance (Jan 2012 – Mar 2020).
It is not committed to this repository due to its size (~49 MB).

To re-download:

```python
import yfinance as yf
import pandas as pd
import pickle

symbols = pd.read_csv("S&P500-Symbols.csv")["Symbol"].tolist()
data = yf.download(symbols, start="2012-01-01", end="2020-03-31", auto_adjust=False)
with open("stocks.pkl", "wb") as f:
    pickle.dump(data, f)
```

---

## Results Summary

| Strategy | Period / Config | Ann. Return | Std Dev | Sharpe | Max Drawdown |
|---|---|---|---|---|---|
| Equal Weight (fixed) | 44 days | 13.5% | 16.3% | 0.83 | −39.0% |
| MVO (fixed) | 44 days | 11.3% | 13.6% | 0.83 | −34.0% |
| **MVO (dynamic WRCP)** | **θ=0.1, n=30, ARL=10k** | **16.0%** | **12.9%** | **1.24** | **−21.0%** |
| DRMVO (dynamic WRCP) | θ=0.1, n=30, ARL=10k | 16.0% | 12.9% | 1.24 | −21.6% |

Dynamic changepoint-triggered rebalancing outperforms fixed schedules across all
metrics: higher returns, lower volatility, higher Sharpe, and shallower drawdowns.

---

## References

- H. Markowitz, "Portfolio Selection," *Journal of Finance*, 1952.
- J. Blanchet, K. Murthy, F. Zhang, "Optimal Transport–Based Distributionally Robust Optimization," 2022.
- X. Cao, K. Xie, Y. Xie, "Robust Sequential Change-Point Detection by Convex Optimization," *IEEE Trans. Information Theory*, 2019.
- P. Nystrup et al., "Detecting change points in VIX and S&P 500: A new approach to dynamic asset allocation," *Journal of Asset Management*, 2016.
- S. Boyd & L. Vandenberghe, *Convex Optimization*, Cambridge University Press, 2004.
