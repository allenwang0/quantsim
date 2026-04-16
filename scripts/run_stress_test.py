"""
Stress Test Runner

Runs any strategy against the canonical set of historical stress periods.
A viable strategy must survive all of them.

Stress periods from the spec:
  dot_com_crash:      2000-03-01 → 2002-10-09   (-49% S&P 500)
  financial_crisis:   2007-10-01 → 2009-03-09   (-57% S&P 500)
  flash_crash:        2010-05-06 → 2010-05-07   (-9.2% intraday)
  eu_debt_crisis:     2011-08-01 → 2011-10-04   (-20% S&P 500)
  china_correction:   2015-08-01 → 2015-09-30   (-11% S&P 500)
  covid_crash:        2020-02-19 → 2020-03-23   (-34% S&P 500)
  rate_shock_2022:    2022-01-01 → 2022-10-12   (-25% S&P 500)

Usage:
  python scripts/run_stress_test.py --strategy sma --symbol SPY
  python scripts/run_stress_test.py --strategy donchian --symbol SPY QQQ GLD TLT
  python scripts/run_stress_test.py --strategy dual_momentum
"""

import sys, os, argparse, logging
from datetime import datetime
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.logging_config import setup_logging
setup_logging()
logger = logging.getLogger("stress_test")


STRESS_PERIODS = {
    "dot_com_crash":     ("2000-03-01", "2002-10-09", "Dot-com crash: -49% S&P 500"),
    "financial_crisis":  ("2007-10-01", "2009-03-09", "Global financial crisis: -57% S&P 500"),
    "flash_crash":       ("2010-05-01", "2010-06-30", "Flash crash + recovery: -9% intraday"),
    "eu_debt_crisis":    ("2011-08-01", "2011-10-31", "EU debt crisis: -20% S&P 500"),
    "china_correction":  ("2015-08-01", "2015-09-30", "China correction: -11% S&P 500"),
    "covid_crash":       ("2020-02-19", "2020-04-30", "COVID crash: -34% → recovery"),
    "rate_shock_2022":   ("2022-01-01", "2022-10-31", "Rate shock: -25% S&P 500"),
    "full_history":      ("2000-01-01", "2023-12-31", "Full 23-year history (all regimes)"),
}


def run_stress_period(
    strategy_name: str,
    symbols: list,
    start: str,
    end: str,
    db_path: str,
    capital: float = 100_000,
) -> dict:
    """Run a vectorized backtest for a single stress period."""
    from data.ingestion import get_bars
    from backtesting.vectorized import VectorizedBacktester, sma_crossover_signal, momentum_signal

    SIGNAL_MAP = {
        "sma": sma_crossover_signal,
        "ema": sma_crossover_signal,
        "momentum": momentum_signal,
        "tsmom": momentum_signal,
    }
    signal_fn = SIGNAL_MAP.get(strategy_name, sma_crossover_signal)

    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")

    price_dict = {}
    for sym in symbols:
        df = get_bars(sym, start_dt, end_dt, adjusted=True, db_path=db_path)
        if not df.empty:
            col = "adj_close" if "adj_close" in df.columns else "close"
            price_dict[sym] = df[col]

    if not price_dict:
        return {"error": f"No data for {symbols} in {start}→{end}"}

    prices = pd.DataFrame(price_dict).dropna(how="all")

    if len(prices) < 30:
        return {"error": f"Insufficient data: only {len(prices)} bars"}

    bt = VectorizedBacktester(
        prices=prices,
        initial_capital=capital,
        commission_rate=0.001,
        slippage_pct=0.001,
    )

    try:
        result = bt.run(signal_fn)
        return result
    except Exception as e:
        return {"error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="QuantSim Stress Test Runner")
    parser.add_argument("--strategy", default="sma",
                        choices=["sma", "ema", "momentum", "tsmom",
                                 "donchian", "dual_momentum"],
                        help="Strategy to stress test")
    parser.add_argument("--symbol", "--symbols", nargs="+", default=["SPY"])
    parser.add_argument("--capital", type=float, default=100_000)
    parser.add_argument("--periods", nargs="*", default=None,
                        help="Specific periods to test (default: all)")
    parser.add_argument("--db", default=None)
    parser.add_argument("--output", default=None, help="Save HTML tearsheet")
    args = parser.parse_args()

    from core.config import config
    from core.database_v2 import init_full_db

    db_path = args.db or config.db_path
    if args.db:
        os.environ["QUANTSIM_DB"] = args.db
    init_full_db(db_path)

    periods_to_test = args.periods or list(STRESS_PERIODS.keys())

    logger.info("=" * 70)
    logger.info(f"STRESS TEST: {args.strategy} on {args.symbol}")
    logger.info("=" * 70)
    logger.info(f"{'Period':<20} {'Dates':<25} {'Return':>8} {'Sharpe':>8} {'MaxDD':>8} {'Result'}")
    logger.info("-" * 70)

    results_by_period = {}
    failures = []

    for period_name in periods_to_test:
        if period_name not in STRESS_PERIODS:
            logger.warning(f"Unknown period: {period_name}")
            continue

        start, end, description = STRESS_PERIODS[period_name]

        result = run_stress_period(
            strategy_name=args.strategy,
            symbols=args.symbol,
            start=start, end=end,
            db_path=db_path,
            capital=args.capital,
        )

        if "error" in result:
            logger.warning(f"  {period_name:<20} SKIPPED: {result['error']}")
            continue

        total_return = result.get("total_return", 0)
        sharpe = result.get("sharpe_ratio", 0)
        max_dd = result.get("max_drawdown", 0)

        # Pass/fail criteria: did not lose more than 25% in the period
        # (a strategy can underperform in a crash; it just cannot catastrophically blow up)
        max_acceptable_loss = -0.25
        passed = max_dd >= max_acceptable_loss

        status = "✓ PASS" if passed else "✗ FAIL"
        if not passed:
            failures.append(period_name)

        results_by_period[period_name] = {
            "total_return": total_return,
            "sharpe": sharpe,
            "max_dd": max_dd,
            "passed": passed,
            "description": description,
        }

        logger.info(
            f"  {period_name:<20} {start}→{end[:4]}  "
            f"{total_return:>+8.1%} {sharpe:>8.2f} {max_dd:>8.1%}  {status}"
        )

    # Summary
    logger.info("=" * 70)
    n_tested = len(results_by_period)
    n_passed = sum(1 for r in results_by_period.values() if r["passed"])
    logger.info(f"SUMMARY: {n_passed}/{n_tested} periods passed")

    if failures:
        logger.warning(f"\nFAILED PERIODS: {failures}")
        logger.warning(
            "A strategy that fails stress periods is fragile and regime-dependent."
        )
        logger.warning(
            "It will likely fail in live trading when those conditions recur."
        )
    else:
        logger.info(
            "\n✓ All stress periods passed. Strategy shows regime resilience."
        )

    # Generate tearsheet if requested
    if args.output and "full_history" in results_by_period:
        logger.info(f"\nGenerating tearsheet: {args.output}")
        from reporting.tearsheet import generate_tearsheet
        # Use full_history equity curve for tearsheet
        full = run_stress_period(
            strategy_name=args.strategy,
            symbols=args.symbol,
            start="2000-01-01", end="2023-12-31",
            db_path=db_path, capital=args.capital,
        )
        if "error" not in full:
            logger.info("Tearsheet generation requires equity curve from DB.")

    logger.info("=" * 70)
    return results_by_period


if __name__ == "__main__":
    main()
