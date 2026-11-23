"""
Build the backtest report: results.json plus a one-file HTML page.

Usage:
  # Synthetic demo data (python tests/fixtures/make_fixture.py writes it)
  quantsim report --prices tests/fixtures/synthetic_prices.csv --strategy sma momentum

  # Prices already in the local database (quantsim bootstrap)
  quantsim report --strategy sma --symbol SPY --start 2010-01-01

  # Also write site/card.json for the portfolio site
  quantsim report --prices tests/fixtures/synthetic_prices.csv --card
"""

import argparse
import json
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from quantsim.reporting.report_page import STRATEGIES, build_results, render_html


def git_sha() -> str:
    out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    return out.stdout.strip() if out.returncode == 0 else "unknown"


def load_csv(path: Path, symbol: str | None):
    frame = pd.read_csv(path, index_col="date", parse_dates=True)
    sidecar = path.with_suffix(".json")
    info = json.loads(sidecar.read_text()) if sidecar.exists() else {"synthetic": False}
    column = symbol or frame.columns[0]
    prices = frame[column].astype(float).rename(column)
    info.update(
        {
            "source": info.get("description", f"CSV file {path.name}"),
            "symbol": column,
            "start": f"{prices.index[0]:%Y-%m-%d}",
            "end": f"{prices.index[-1]:%Y-%m-%d}",
        }
    )
    return prices, info


def load_db(symbol: str, start: str, end: str, db_path: str | None):
    from quantsim.data.ingestion import get_bars

    kwargs = {"db_path": db_path} if db_path else {}
    bars = get_bars(symbol, datetime.fromisoformat(start), datetime.fromisoformat(end), **kwargs)
    if bars.empty:
        raise SystemExit(f"No bars for {symbol} in the database. Run 'quantsim bootstrap' first.")
    column = "adj_close" if "adj_close" in bars.columns else "close"
    prices = bars[column].astype(float)
    prices.index = pd.DatetimeIndex(prices.index).tz_localize(None).normalize()
    prices = prices.rename(symbol)
    info = {
        "synthetic": False,
        "source": "Local database (yfinance download)",
        "symbol": symbol,
        "start": f"{prices.index[0]:%Y-%m-%d}",
        "end": f"{prices.index[-1]:%Y-%m-%d}",
    }
    return prices, info


PARITY_TEST = Path("tests/test_engine_parity.py")


def parity_tolerance(path: Path = PARITY_TEST) -> float | None:
    """Return-gap tolerance the parity test enforces, read from the test itself."""
    if not path.exists():
        return None
    match = re.search(r"abs\(event_return - vector_return\) < ([\d.]+)", path.read_text())
    return float(match.group(1)) if match else None


def card(results: dict, parity_tol: float | None = None) -> dict:
    """Portfolio card. Headline = the measured engine speedup (median of the
    benchmark runs at the largest asset count), the same result as the report's
    hero; the deflated-Sharpe check stays in the report body."""
    bench = results.get("benchmark")
    if not bench:
        raise SystemExit("card needs benchmark results: run benchmarks/bench_engines.py first")
    big = max(bench["rows"], key=lambda r: r["assets"])
    speedup = big["event_median_s"] / big["vectorized_median_s"]
    parity = f"; engines agree within {parity_tol * 100:g} pp" if parity_tol else ""
    takeaway = (
        f"Vectorized engine {speedup:.0f}x faster than event-driven, {big['assets']} assets x "
        f"10 years, synthetic data{parity}."
    )

    def row(label: str, engine: str) -> dict:
        runs = big.get(f"{engine}_runs_s") or [big[f"{engine}_median_s"]]
        return {
            "label": label,
            "low": round(min(runs), 4),
            "mid": round(big[f"{engine}_median_s"], 4),
            "high": round(max(runs), 4),
        }

    return {
        "schema_version": 1,
        "repo": "quantsim",
        "sha": results["meta"]["sha"],
        "generated_at": results["meta"]["generated_at"],
        "command": results["meta"]["command"],
        "data_note": (
            f"{bench['data']}; {bench['strategy']}; median of {bench['runs']} runs, {bench['cpu']}"
        ),
        "metric": {
            "label": f"Vectorized vs event-driven speed, {big['assets']} assets",
            "value": round(speedup, 1),
            "unit": "x",
            "higher_is_better": True,
        },
        "takeaway": takeaway[:140],
        "chart": {
            "type": "interval",
            "x_label": "Seconds per backtest (min, median, max)",
            "x_unit": "s",
            "x_scale": "log",
            "rows": [row("Vectorized", "vectorized"), row("Event-driven", "event")],
        },
    }


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="quantsim report", description="Build the HTML backtest report"
    )
    parser.add_argument(
        "--prices", type=Path, help="CSV with a date column and one column per symbol"
    )
    parser.add_argument("--symbol", help="Column of --prices, or symbol in the database")
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    parser.add_argument("--strategy", nargs="+", default=["sma"], choices=sorted(STRATEGIES))
    parser.add_argument(
        "--cost-bps",
        default="0,5,10,25",
        help="Cost-sensitivity levels in bps; each charges that many bps of commission "
        "and of slippage per trade",
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="Output directory (default reports/<run>)"
    )
    parser.add_argument("--benchmark", type=Path, default=Path("benchmarks/results.json"))
    parser.add_argument("--seed", type=int, default=0, help="Seed for the bootstrap intervals")
    parser.add_argument("--card", action="store_true", help="Also write site/card.json")
    parser.add_argument("--db", default=None, help="Database path override (with --symbol)")
    args = parser.parse_args(argv)

    if args.prices:
        prices, info = load_csv(args.prices, args.symbol)
    elif args.symbol:
        prices, info = load_db(args.symbol, args.start, args.end, args.db)
    else:
        parser.error("give --prices CSV or --symbol from the database")

    costs = [float(c) for c in args.cost_bps.split(",")]
    generated_at = (
        datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    )
    meta = {
        "generated_at": generated_at,
        "sha": git_sha(),
        "command": "quantsim report " + shlex.join(argv),
        "seed": args.seed,
    }
    results = build_results(prices, args.strategy, costs, info, meta, args.benchmark, args.seed)

    out = args.out or Path("reports") / f"{prices.name}_{generated_at[:19].replace(':', '')}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    (out / "index.html").write_text(render_html(results))
    print(f"Wrote {out / 'index.html'} and {out / 'results.json'}")

    if args.card:
        site = Path("site")
        site.mkdir(exist_ok=True)
        (site / "card.json").write_text(
            json.dumps(card(results, parity_tolerance()), indent=2) + "\n"
        )
        print(f"Wrote {site / 'card.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
