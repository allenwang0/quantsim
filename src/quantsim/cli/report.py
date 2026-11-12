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


def load_db(symbol: str, start: str, end: str):
    from quantsim.data.ingestion import get_bars

    bars = get_bars(symbol, datetime.fromisoformat(start), datetime.fromisoformat(end))
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


def card(results: dict, strategy: str) -> dict:
    data = results["meta"]["data"]
    s = results["strategies"][strategy]
    windows = s["walk_forward"]["windows"][-12:]
    dsr = s["walk_forward"]["dsr"]
    if data.get("synthetic"):
        note = f"Seeded synthetic GBM with regime shifts, seed {data['seed']}"
    else:
        note = f"{data['source']}, {data['symbol']}, {data['start']} to {data['end']}"
    takeaway = (
        f"{s['label']}: deflated Sharpe {dsr} after {s['walk_forward']['total_trials']} trials; "
        f"out-of-sample return {s['walk_forward']['oos_return_pct']}%."
    )
    return {
        "schema_version": 1,
        "repo": "quantsim",
        "sha": results["meta"]["sha"],
        "generated_at": results["meta"]["generated_at"],
        "command": results["meta"]["command"],
        "data_note": note,
        "metric": {
            "label": "Deflated Sharpe ratio",
            "value": dsr,
            "unit": "probability",
            "higher_is_better": True,
        },
        "takeaway": takeaway[:140],
        "chart": {
            "type": "interval",
            "x_label": "Annualized Sharpe",
            "x_unit": "",
            "x_scale": "linear",
            "rows": [
                {
                    "label": w["label"],
                    "low": w["ci_low"],
                    "mid": w["oos_sharpe"],
                    "high": w["ci_high"],
                }
                for w in windows
            ],
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
        "--cost-bps", default="0,5,10,25", help="Comma-separated costs per trade in bps"
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="Output directory (default reports/<run>)"
    )
    parser.add_argument("--benchmark", type=Path, default=Path("benchmarks/results.json"))
    parser.add_argument("--seed", type=int, default=0, help="Seed for the bootstrap intervals")
    parser.add_argument("--card", action="store_true", help="Also write site/card.json")
    args = parser.parse_args(argv)

    if args.prices:
        prices, info = load_csv(args.prices, args.symbol)
    elif args.symbol:
        prices, info = load_db(args.symbol, args.start, args.end)
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
            json.dumps(card(results, args.strategy[0]), indent=2) + "\n"
        )
        print(f"Wrote {site / 'card.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
