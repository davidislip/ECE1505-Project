# CLAUDE.md — ECE 1505 Convex Optimization Project

## Project Overview

This project implements a **dynamic portfolio allocation strategy** for S&P 500 equities that uses Wasserstein-based changepoint detection to trigger portfolio rebalancing. It was developed by David Islip and Gurjot Dhaliwal for ECE 1505 (Convex Optimization).

**Core idea:** Rather than rebalancing on a fixed schedule, detect structural breaks (changepoints) in return distributions and rebalance only when the market regime changes.

---

## Repo Structure

```
ECE1505-Project/
├── Data_ChangeDetection.ipynb   # Main pipeline: data → changepoint detection → optimization → backtest
├── Plots.ipynb                  # Visualization and sensitivity analysis
├── hyp.py                       # Helper: extract trade/changepoint times
├── stocks.pkl                   # S&P 500 daily OHLCV data (Jan 2012 – Mar 2020, ~49 MB)
├── S&P500-Symbols.csv           # 500 tickers
├── S&P500-Info.csv              # Company metadata
├── Results/                     # 28 .pkl files; each encodes {delta}_{arl}_{n} parameters
├── Experimental Outputs/        # CSV/XLSX performance summaries
├── Images/                      # Supporting figures for the report
└── ECE_1505_Project_*.pdf       # Final project report
```

---

## Key Algorithms

### Changepoint Detection — `robust_hypothesis(t, n, theta_val, R)`
Solves a Wasserstein hypothesis test comparing return distributions in the window before and after candidate changepoint `t`. Uses `cvxpy` to find optimal transport plans `p1`, `p2`.

**Parameters:**
- `t` — candidate changepoint index
- `n` — look-back/look-forward window size (tested: 30, 60, 90 days)
- `theta_val` (θ) — Wasserstein ball radius (tested: 0.05, 0.075, 0.1, 0.2, 0.5, 1.0)
- `R` — return matrix (T × N)

### Sequential Detection — `detect_change(phi, b)`
CUSUM-style accumulator over the `phi` signal (sign of distributional shift). Returns detection time `T` when cumulative sum exceeds threshold `b`.

`calculate_ARL(p1, p2, phi, b)` calibrates `b` for a target Average Run Length (1000 or 10000 days).

### Portfolio Optimization
| Function | Description |
|---|---|
| `markowitz_mvo(R, alpha_val)` | Standard Markowitz min-variance, long-only, return ≥ α |
| `blanchet_mvo(R, delta_val, alpha_val)` | Distributionally Robust MVO; penalizes variance by √δ·‖x‖₁ over a Wasserstein ambiguity ball |

δ values tested: `{0, (0.25α)², (0.5α)², α², (1.5α)²}`

### Backtesting
- `test_fixed_strategy(period, R, P, strategy, ...)` — rebalance every N days
- `test_variable_strategy(R, P, rebal, est, strategy, ...)` — rebalance at detected changepoints

---

## Running the Notebooks

1. Install dependencies (see **Suggested Improvements** below for a `requirements.txt`):
   ```bash
   pip install cvxpy numpy pandas matplotlib seaborn scikit-learn scipy yfinance openpyxl
   ```

2. Open and run `Data_ChangeDetection.ipynb` top-to-bottom. Results are saved to `Results/`.

3. Open `Plots.ipynb` to reproduce figures and performance tables.

> **Note:** The `stocks.pkl` file (~49 MB) must be present. It is the pre-downloaded yfinance dataset. Re-downloading with the snippet in Cell 2 of the main notebook requires an active internet connection.

---

## Parameter Sweep Summary

The full sweep covers **24 configurations** (6 θ × 2 ARL × 2 n), each producing changepoint detection results stored in `Results/{theta}_{arl}_{n}.pkl`.

Best-performing configuration (from `Output_accepted.csv`):
- θ = 0.1, n = 30, ARL = 10000, strategy = MVO
- Annualized Return: 16.0%, Sharpe: 1.24, Max Drawdown: −21%

---

## Suggested Improvements & Changes

These are actionable improvements to increase reproducibility, correctness, and code quality:

### 1. Add `requirements.txt`

No dependency file exists. This breaks reproducibility on any new machine.

```txt
# requirements.txt
cvxpy>=1.3
numpy>=1.24
pandas>=2.0
matplotlib>=3.7
seaborn>=0.12
scikit-learn>=1.3
scipy>=1.11
yfinance>=0.2
openpyxl>=3.1
notebook>=7.0
```

### 2. Extract Core Logic into a Python Module

All algorithm code lives inside notebook cells, making it untestable and hard to reuse. Refactor into a proper module:

```
src/
├── __init__.py
├── changepoint.py      # robust_hypothesis, detect_change, calculate_ARL
├── optimization.py     # markowitz_mvo, blanchet_mvo
├── backtest.py         # test_fixed_strategy, test_variable_strategy, RetStats, max_dd
└── data.py             # data loading / preprocessing helpers
```

The notebooks then become thin wrappers that import from `src/`.

### 3. Add Unit Tests

No tests exist. Key functions are mathematically precise enough to be testable:

```python
# tests/test_optimization.py
def test_markowitz_weights_sum_to_one():
    R = np.random.randn(60, 10) * 0.01
    w = markowitz_mvo(R, alpha_val=0.0002)
    assert abs(w.sum() - 1.0) < 1e-5

def test_markowitz_weights_nonnegative():
    ...

def test_blanchet_reduces_to_markowitz_at_zero_delta():
    ...
```

Run with `pytest tests/`.

### 4. Replace Magic Numbers with Named Constants

Threshold `b` values and return targets are scattered through cells with no context:

```python
# Before (opaque)
b = 7

# After (self-documenting)
DETECTION_THRESHOLD = 7          # calibrated to ARL=10000 for theta=0.1
TARGET_ANNUAL_RETURN = 0.14      # 14% annualized
ALPHA = TARGET_ANNUAL_RETURN / 250
```

### 5. Fix Potential Look-Ahead Bias

In `test_variable_strategy`, verify that the estimation window `est[i]` for optimization at rebalancing time `rebal[i]` uses **only** data up to `rebal[i]`. Add an explicit assertion:

```python
assert est[i][1] <= rebal[i], "Estimation window leaks future data"
```

### 6. Add Transaction Cost Model

The backtest ignores trading costs, which inflates performance — especially for the dynamic strategy (which rebalances more often). Add a simple round-trip cost:

```python
TRANSACTION_COST_BPS = 10  # 10 basis points round-trip

def apply_transaction_costs(old_weights, new_weights, portfolio_value, cost_bps):
    turnover = np.abs(new_weights - old_weights).sum()
    return portfolio_value * (1 - turnover * cost_bps / 10000)
```

### 7. Use Structured Result Storage (replace filename encoding)

Results are currently named `{delta}_{arl}_{n}.pkl` with parameters encoded in filenames. This is brittle. Switch to a structured format:

```python
import json

result = {
    "params": {"delta": 0.1, "arl": 10000, "n": 30},
    "changepoints": [...],
    "metadata": {"date_created": "...", "data_range": "2012-2020"}
}

with open("Results/result_001.json", "w") as f:
    json.dump(result, f)
```

Or use `pandas` with a results manifest CSV.

### 8. Improve Solver Robustness in `blanchet_mvo`

The DRMVO solver can fail silently and fall back to equal weights. Make failures explicit and logged:

```python
problem.solve(solver=cp.CLARABEL)
if problem.status not in ["optimal", "optimal_inaccurate"]:
    import warnings
    warnings.warn(f"Solver failed at t={t}: status={problem.status}. Using equal weights.")
    return np.ones(N) / N
```

### 9. Clarify / Complete `hyp.py`

`hyp.py` contains `get_list_of_trade_times()` but it's unclear if it's used in the notebooks (the main notebook reimplements similar logic inline). Either:
- Delete it if unused
- Or import from it in the notebooks if it is the canonical implementation

### 10. Add a `README.md`

There is no README. At minimum it should cover:
- Project description and academic context
- How to install dependencies and run the notebooks
- Summary of results
- Citations for Cao et al., Blanchet et al., Boyd et al.

---

## Code Conventions (for future contributions)

- Python 3.10+
- Notebooks: clear all outputs before committing (`Cell → All Output → Clear`)
- Variable naming: `R` = returns matrix (T×N), `P` = price matrix (T×N), `w` = portfolio weights (N,)
- Solver: prefer `cp.CLARABEL` (default in cvxpy ≥ 1.4) over deprecated ECOS/SCS for QP problems
- Do not commit `stocks.pkl` or other large binary files to git — use `.gitignore`
