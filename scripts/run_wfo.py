"""
Walk-Forward Optimization and Parameter Sweep Runner

Usage:
  # Walk-forward optimization on SMA crossover
  python scripts/run_wfo.py --signal sma --symbol SPY --train-years 3 --test-months 12

  # Parameter sweep (fast, vectorized)
  python scripts/run_wfo.py --mode sweep --signal sma --symbol SPY QQQ \
    --param fast 10 20 50 --param slow 100 150 200

  # Full WFO with regime analysis
  python scripts/run_wfo.py --signal momentum --symbol SPY QQQ GLD TLT IWM \
    --train-years 4 --test-months 12 --regime-analysis
"""

import sys
import os
import argparse
import json
import logging
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("wfo")


SIGNAL_REGISTRY = {
    "sma": ("backtesting.vectorized", "sma_crossover_signal", {"fast": [10, 20, 50], "slow": [100, 150, 200]}),
    "ema": ("backtesting.vectorized", "sma_crossover_signal", {"fast": [8, 12, 20], "slow": [26, 50, 100]}),
    "rsi": ("backtesting.vectorized", "rsi_signal", {"period": [10, 14, 21], "oversold": [25, 30], "overbought": [70, 75]}),
    "momentum": ("backtesting.vectorized", "momentum_signal", {"lookback": [126, 189, 252], "skip": [21]}),
    "bollinger": ("backtesting.vectorized", "bollinger_signal", {"window": [10, 15, 20, 25], "k": [1.5, 2.0, 2.5]}),
    "donchian": ("backtesting.vectorized", "donchian_signal", {"period": [15, 20, 30, 55]}),
}


def main():
    parser = argparse.ArgumentParser(description="QuantSim Walk-Forward Optimizer")
    parser.add_argument("--mode", default="wfo",
                        choices=["wfo", "sweep"],
                        help="wfo = rolling walk-forward; sweep = single-period parameter sweep")
    parser.add_argument("--signal", default="sma", choices=list(SIGNAL_REGISTRY.keys()),
                        help="Signal function to optimize")
    parser.add_argument("--symbol", "--symbols", nargs="+", default=["SPY"],
                        help="Symbol(s)")
    parser.add_argument("--start", default="2010-01-01", help="Historical data start")
    parser.add_argument("--end", default=None, help="Historical data end")
    parser.add_argument("--train-years", type=float, default=3.0,
                        help="Training window in years (WFO mode)")
    parser.add_argument("--test-months", type=int, default=12,
                        help="Test window in months (WFO mode)")
    parser.add_argument("--step-months", type=int, default=6,
                        help="Step size in months (WFO mode)")
    parser.add_argument("--capital", type=float, default=100_000)
    parser.add_argument("--commission", type=float, default=0.001,
                        help="Commission rate (0.001 = 0.1%%)")
    parser.add_argument("--slippage", type=float, default=0.001,
                        help="Slippage rate (0.001 = 0.1%%)")
    parser.add_argument("--param", nargs="+", action="append", metavar=("KEY", "VALUES"),
                        help="Custom parameter: --param fast 10 20 50")
    parser.add_argument("--regime-analysis", action="store_true",
                        help="Analyze performance across market regimes")
    parser.add_argument("--db", default=None, help="Database path override")
    parser.add_argument("--output", default=None, help="Save results to JSON file")
    args = parser.parse_args()

    from core.config import config
    from core.database_v2 import init_full_db
    db_path = args.db or config.db_path
    if args.db:
        os.environ["QUANTSIM_DB"] = args.db
    init_full_db(db_path)

    # Load data
    logger.info(f"Loading data for {args.symbol} from {args.start}...")
    from data.ingestion import get_bars
    import pandas as pd

    end = datetime.strptime(args.end, "%Y-%m-%d") if args.end else datetime.utcnow()
    start = datetime.strptime(args.start, "%Y-%m-%d")

    price_dict = {}
    for sym in args.symbol:
        df = get_bars(sym, start, end, adjusted=True, db_path=db_path)
        if not df.empty:
            col = "adj_close" if "adj_close" in df.columns else "close"
            price_dict[sym] = df[col]
        else:
            logger.warning(f"No data for {sym} - run bootstrap_data.py first")

    if not price_dict:
        logger.error("No price data loaded. Run: python scripts/bootstrap_data.py --symbols " + " ".join(args.symbol))
        sys.exit(1)

    prices = pd.DataFrame(price_dict).dropna(how="all")
    logger.info(f"Loaded {len(prices)} bars for {list(prices.columns)}")

    # Build signal function and param grid
    module_name, fn_name, default_grid = SIGNAL_REGISTRY[args.signal]
    import importlib
    module = importlib.import_module(module_name)
    signal_fn = getattr(module, fn_name)

    # Override param grid from CLI
    param_grid = default_grid.copy()
    if args.param:
        for param_spec in args.param:
            if len(param_spec) >= 2:
                key = param_spec[0]
                values = [float(v) if "." in str(v) else int(v) for v in param_spec[1:]]
                param_grid[key] = values

    logger.info(f"Parameter grid: {param_grid}")
    n_configs = 1
    for v in param_grid.values():
        n_configs *= len(v)
    logger.info(f"Total configurations: {n_configs}")

    # Benchmark
    benchmark = prices.get("SPY", prices.iloc[:, 0])

    if args.mode == "sweep":
        # Fast vectorized parameter sweep
        logger.info("Running vectorized parameter sweep...")
        from backtesting.vectorized import VectorizedBacktester
        bt = VectorizedBacktester(
            prices=prices,
            initial_capital=args.capital,
            commission_rate=args.commission,
            slippage_pct=args.slippage,
        )
        results_df = bt.parameter_sweep(signal_fn, param_grid, benchmark=benchmark)
        logger.info(f"\nTop 5 configurations (by Deflated Sharpe):")
        cols = [c for c in ["deflated_sharpe_corrected", "sharpe_ratio", "cagr",
                             "max_drawdown", "calmar_ratio"] if c in results_df.columns]
        cols = list(param_grid.keys()) + cols
        print(results_df[cols].head(5).to_string(index=False))

        results_dict = {"mode": "sweep", "results": results_df.to_dict("records")}

    else:
        # Walk-forward optimization
        logger.info(
            f"Running WFO: train={args.train_years}yr, "
            f"test={args.test_months}mo, step={args.step_months}mo"
        )
        from backtesting.walk_forward import WalkForwardOptimizer, RegimeAnalyzer
        wfo = WalkForwardOptimizer(
            prices=prices,
            train_years=args.train_years,
            test_months=args.test_months,
            step_months=args.step_months,
        )
        results = wfo.optimize_and_evaluate(
            signal_fn=signal_fn,
            param_grid=param_grid,
            initial_capital=args.capital,
            commission_rate=args.commission,
            slippage_pct=args.slippage,
            benchmark=benchmark,
        )

        logger.info("\n" + "=" * 60)
        logger.info("WALK-FORWARD RESULTS (OUT-OF-SAMPLE ONLY)")
        logger.info("=" * 60)
        logger.info(f"Windows tested: {results.get('n_windows', 0)}")
        logger.info(f"Total configs tested: {results.get('total_configs_tested', 0)}")
        logger.info(f"Average OOS Sharpe: {results.get('avg_oos_sharpe', 0):.3f}")
        logger.info(f"Average IS Sharpe: {results.get('avg_is_sharpe', 0):.3f}")
        logger.info(f"IS→OOS Degradation: {results.get('sharpe_degradation', 0):.3f}")
        logger.info(f"Deflated SR (corrected): {results.get('deflated_sharpe_corrected', 0):.3f}")
        logger.info(f"OOS Win Rate: {results.get('oos_win_rate', 0):.1%}")
        logger.info(f"Most stable params: {results.get('most_stable_params', 'N/A')}")
        logger.info(f"VERDICT: {results.get('summary', {}).get('message', 'N/A')}")
        logger.info("=" * 60)

        # Regime analysis
        if args.regime_analysis and len(prices.columns) > 0:
            logger.info("\nRunning regime analysis...")
            spy = prices.get("SPY", prices.iloc[:, 0])
            regimes = RegimeAnalyzer.classify_regimes(spy)
            regime_counts = regimes.value_counts()
            logger.info(f"Regime distribution:\n{regime_counts.to_string()}")

        results_dict = results

    # Save output
    # Persist WFO results to database
    if results_dict.get('window_results'):
        try:
            import uuid
            from core.database_v2 import log_wfo_result
            for w in results_dict.get('window_results', []):
                if isinstance(w, dict):
                    log_wfo_result(
                        db_path=db_path,
                        result_id=str(uuid.uuid4()),
                        strategy_id=args.signal,
                        window=w,
                        best_params=w.get('best_params', {}),
                        metrics=w,
                    )
        except Exception as e:
            logger.warning(f'WFO persistence failed: {e}')

    if args.output:
        def clean(obj):
            if isinstance(obj, dict):
                return {k: clean(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean(v) for v in obj]
            elif hasattr(obj, 'isoformat'):
                return obj.isoformat()
            elif isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            return str(obj)

        with open(args.output, "w") as f:
            json.dump(clean(results_dict), f, indent=2)
        logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
