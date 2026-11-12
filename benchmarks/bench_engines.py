"""
Time the event-driven and vectorized engines on the same synthetic data.

    python benchmarks/bench_engines.py [--runs 5]

10 years (2520 bars) of seeded synthetic GBM with regime shifts for 1, 10
and 50 assets, SMA 50/200 crossover in both engines. Reports the median of
--runs timed runs after one warm-up run (which also compiles Numba).
Timed: the vectorized run including its analytics, and the event engine's
run() loop. Event-engine construction, which loads bars from SQLite, is not
timed. Writes benchmarks/results.json and benchmarks/results.md.
"""

import argparse
import json
import platform
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

import quantsim
from quantsim.backtesting.engine import BacktestEngine
from quantsim.backtesting.vectorized import (
    NUMBA_AVAILABLE,
    VectorizedBacktester,
    sma_crossover_signal,
)
from quantsim.core.event_queue import EventQueue
from quantsim.data.synthetic import regime_gbm_prices, write_prices_to_db
from quantsim.strategies.trend import SMAcrossover

SEED = 7
N_BARS = 2520
ASSET_COUNTS = (1, 10, 50)
FAST, SLOW = 50, 200
OUT_DIR = Path(__file__).parent


def cpu_name() -> str:
    try:
        out = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except FileNotFoundError:
        pass
    return platform.processor() or platform.machine()


def time_vectorized(prices) -> float:
    backtester = VectorizedBacktester(prices)
    start = time.perf_counter()
    backtester.run(sma_crossover_signal, {"fast": FAST, "slow": SLOW})
    return time.perf_counter() - start


def time_event(prices, db_path) -> float:
    strategy = SMAcrossover(list(prices.columns), EventQueue(), fast=FAST, slow=SLOW)
    engine = BacktestEngine(
        strategies=[strategy],
        start=prices.index[0].to_pydatetime(),
        end=prices.index[-1].to_pydatetime(),
        warmup_bars=0,
        db_path=db_path,
        verbose=False,
    )
    start = time.perf_counter()
    engine.run()
    return time.perf_counter() - start


def timed_run(engine: str, prices, db_path: Path) -> float:
    if engine == "vectorized":
        return time_vectorized(prices)
    write_prices_to_db(str(db_path), prices)
    return time_event(prices, str(db_path))


def numba_version() -> str:
    if not NUMBA_AVAILABLE:
        return "not installed"
    import numba

    return numba.__version__


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description="Benchmark the two engines")
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args(argv)

    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for n_assets in ASSET_COUNTS:
            prices, _ = regime_gbm_prices(n_assets=n_assets, n_bars=N_BARS, seed=SEED)
            row = {"assets": n_assets}
            for engine in ("vectorized", "event"):
                timed_run(engine, prices, Path(tmp) / f"warmup_{engine}_{n_assets}.db")
                samples = [
                    timed_run(engine, prices, Path(tmp) / f"{engine}_{n_assets}_{i}.db")
                    for i in range(args.runs)
                ]
                row[f"{engine}_median_s"] = statistics.median(samples)
                print(
                    f"{n_assets:>3} assets  {engine:<10} median {row[f'{engine}_median_s']:.4f} s",
                    flush=True,
                )
            row["speedup"] = row["event_median_s"] / row["vectorized_median_s"]
            rows.append(row)

    results = {
        "data": f"Synthetic GBM with regime shifts, seed {SEED}, {N_BARS} daily bars (10 years)",
        "strategy": f"SMA crossover {FAST}/{SLOW}",
        "runs": args.runs,
        "cpu": cpu_name(),
        "python": platform.python_version(),
        "numba": numba_version(),
        "numpy": np.__version__,
        "quantsim": quantsim.__version__,
        "rows": rows,
    }
    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    (OUT_DIR / "results.md").write_text(markdown_table(results))
    print(markdown_table(results))
    return results


def markdown_table(results: dict) -> str:
    lines = [
        "| Assets | Vectorized (s) | Event-driven (s) | Event / vectorized |",
        "|---:|---:|---:|---:|",
    ]
    for row in results["rows"]:
        lines.append(
            f"| {row['assets']} | {row['vectorized_median_s']:.4f} | "
            f"{row['event_median_s']:.2f} | {row['speedup']:.0f}x |"
        )
    lines.append("")
    lines.append(
        f"Median of {results['runs']} runs after a warm-up. {results['data']}; "
        f"{results['strategy']}. CPU: {results['cpu']}; Python {results['python']}, "
        f"NumPy {results['numpy']}, Numba {results['numba']}."
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
