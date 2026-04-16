"""
Bootstrap script: download historical data for the full universe.
Run this first before any backtesting.

Usage:
  python scripts/bootstrap_data.py
  python scripts/bootstrap_data.py --symbols SPY QQQ IWM --start 2010-01-01
  python scripts/bootstrap_data.py --fred-only
"""

import sys
import os
import argparse
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("bootstrap")


def main():
    parser = argparse.ArgumentParser(description="Bootstrap QuantSim historical data")
    parser.add_argument("--symbols", nargs="*", help="Specific symbols (default: full universe)")
    parser.add_argument("--start", default="2010-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD")
    parser.add_argument("--fred-only", action="store_true", help="Only download FRED macro data")
    parser.add_argument("--db", default=None, help="Database path override")
    args = parser.parse_args()

    from core.database import init_db, DB_PATH
    db_path = args.db or DB_PATH
    if args.db:
        os.environ["QUANTSIM_DB"] = args.db

    logger.info(f"Initializing database: {db_path}")
    init_db(db_path)

    # ── FRED Macro Data ──────────────────────────────────────────────────────
    logger.info("Downloading FRED macro series...")
    from data.ingestion import fetch_fred_series, FRED_SERIES

    fred_results = fetch_fred_series(
        series_ids=list(FRED_SERIES.keys()),
        start=args.start,
        db_path=db_path,
    )

    for series_id, count in fred_results.items():
        logger.info(f"  FRED {series_id}: {count} observations")

    if args.fred_only:
        logger.info("FRED-only mode complete.")
        return

    # ── Equity Data ──────────────────────────────────────────────────────────
    from data.ingestion import get_universe, fetch_equity_history

    if args.symbols:
        universe = args.symbols
        logger.info(f"Using {len(universe)} user-specified symbols")
    else:
        universe = get_universe()
        logger.info(f"Using default universe: {len(universe)} symbols")

    logger.info(f"Downloading equity data {args.start} → {args.end or 'today'}...")
    logger.info("This may take several minutes due to yfinance rate limiting.")
    logger.info("Rate limit: ~0.5s between batches of 50 symbols.")

    results = fetch_equity_history(
        symbols=universe,
        start=args.start,
        end=args.end,
        db_path=db_path,
    )

    success = sum(1 for v in results.values() if v > 0)
    total_bars = sum(results.values())

    logger.info(f"Bootstrap complete: {success}/{len(universe)} symbols, {total_bars:,} bars total")

    # Summary of failures
    failed = [s for s, v in results.items() if v == 0]
    if failed:
        logger.warning(f"Failed symbols ({len(failed)}): {failed[:10]}{'...' if len(failed)>10 else ''}")


if __name__ == "__main__":
    main()


def _bootstrap_with_progress(symbols, start, end, db_path):
    """Bootstrap with a simple text progress bar (no tqdm dependency)."""
    total = len(symbols)
    for i, sym in enumerate(symbols):
        pct = int((i + 1) / total * 40)
        bar = "█" * pct + "░" * (40 - pct)
        print(f"\r[{bar}] {i+1}/{total} {sym:<10}", end="", flush=True)
        try:
            from data.ingestion import fetch_equity_history
            fetch_equity_history([sym], start=start, end=end, db_path=db_path)
        except Exception as e:
            pass
    print(f"\r{'Done':50}")
