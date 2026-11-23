| Assets | Vectorized (s) | Event-driven (s) | Event / vectorized |
|---:|---:|---:|---:|
| 1 | 0.1030 | 1.27 | 12x |
| 10 | 0.1040 | 9.60 | 92x |
| 50 | 0.0990 | 45.53 | 460x |

Median of 5 runs after a warm-up. Synthetic GBM with regime shifts, seed 7, 2520 daily bars (10 years); SMA crossover 50/200. CPU: Apple M5 Pro; Python 3.13.14, NumPy 2.5.4, Numba 0.68.1. Load average (1/5/15 min) [4.95, 6.47, 6.75] before, [6.74, 5.69, 6.21] after.
