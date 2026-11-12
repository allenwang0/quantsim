| Assets | Vectorized (s) | Event-driven (s) | Event / vectorized |
|---:|---:|---:|---:|
| 1 | 0.1107 | 0.98 | 9x |
| 10 | 0.1059 | 11.96 | 113x |
| 50 | 0.1090 | 54.73 | 502x |

Median of 5 runs after a warm-up. Synthetic GBM with regime shifts, seed 7, 2520 daily bars (10 years); SMA crossover 50/200. CPU: Apple M5 Pro; Python 3.12.13, NumPy 2.5.4, Numba 0.68.1.
