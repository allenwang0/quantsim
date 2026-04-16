"""
QuantSim Backtest Runner - v2

Supports:
  Event-driven backtesting (all strategies)
  Vectorized backtesting (fast, parameter sweeps)
  Walk-forward optimization
  Portfolio optimization overlays (HRP, Risk Parity, Black-Litterman)
  Ensemble multi-strategy backtests

Usage:
  # Walking skeleton validation
  python scripts/run_backtest.py --validate

  # Event-driven backtest
  python scripts/run_backtest.py --strategy sma --symbol SPY
  python scripts/run_backtest.py --strategy donchian --symbol SPY QQQ GLD TLT IWM
  python scripts/run_backtest.py --strategy iron_condor --symbol SPY

  # Vectorized backtest (100x faster, good for research)
  python scripts/run_backtest.py --mode vectorized --strategy sma --symbol SPY QQQ

  # Parameter sweep
  python scripts/run_backtest.py --mode sweep --strategy sma --symbol SPY

  # Walk-forward optimization
  python scripts/run_backtest.py --mode wfo --strategy sma --symbol SPY

  # With portfolio optimization overlay
  python scripts/run_backtest.py --strategy tsmom --optimizer hrp --symbol SPY QQQ GLD TLT

  # Ensemble of strategies
  python scripts/run_backtest.py --strategy sma ema macd --ensemble --symbol SPY QQQ

Available strategies:
  buy_and_hold, sma, ema, macd, donchian, adx, tsmom,
  bollinger, rsi, xs_momentum, dual_momentum, low_vol,
  covered_call, iron_condor, long_straddle
"""

import sys, os, argparse, json, logging
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.logging_config import setup_logging
setup_logging()

logger = logging.getLogger("backtest")


def main():
    parser = argparse.ArgumentParser(description="QuantSim Backtest Runner v2")
    parser.add_argument("--mode", default="event",
                        choices=["event", "vectorized", "sweep", "wfo"],
                        help="Engine mode (default: event)")
    parser.add_argument("--strategy", nargs="+", default=["sma"])
    parser.add_argument("--symbol", "--symbols", nargs="+", default=["SPY"])
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default=datetime.now(timezone.utc).replace(tzinfo=None).strftime("%Y-%m-%d"))
    parser.add_argument("--capital", type=float, default=None)
    parser.add_argument("--slippage", default="volume",
                        choices=["none", "fixed", "volume"])
    parser.add_argument("--commission", default="zero",
                        choices=["zero", "per_share"])
    parser.add_argument("--optimizer", default=None,
                        choices=["hrp", "risk_parity", "mean_variance", "black_litterman"],
                        help="Portfolio optimization overlay")
    parser.add_argument("--ensemble", action="store_true",
                        help="Use signal ensemble aggregation for multiple strategies")
    parser.add_argument("--ensemble-method", default="confidence_weighted",
                        choices=["majority_vote", "confidence_weighted", "orthogonality_checked"])
    parser.add_argument("--train-years", type=float, default=3.0,
                        help="Training window years (WFO mode)")
    parser.add_argument("--test-months", type=int, default=12,
                        help="Test window months (WFO mode)")
    parser.add_argument("--db", default=None)
    parser.add_argument("--output", default=None, help="Save results to JSON")
    parser.add_argument("--validate", action="store_true",
                        help="Walking skeleton validation (SPY buy-and-hold)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    from core.config import config, QuantSimConfig
    from core.database_v2 import init_full_db

    db_path = args.db or config.db_path
    if args.db:
        os.environ["QUANTSIM_DB"] = args.db
    init_full_db(db_path)

    capital = args.capital or config.initial_capital

    # Walking skeleton validation
    if args.validate:
        args.strategy = ["buy_and_hold"]
        args.symbol = ["SPY"]
        args.start = "2020-01-02"
        args.end = "2023-12-29"
        args.mode = "event"
        logger.info("=" * 60)
        logger.info("WALKING SKELETON VALIDATION")
        logger.info("Expected: SPY total return 2020-2023 ≈ 50-80%")
        logger.info("=" * 60)

    start_dt = datetime.strptime(args.start, "%Y-%m-%d")
    end_dt = datetime.strptime(args.end, "%Y-%m-%d")

    # ── Vectorized / Sweep / WFO modes ───────────────────────────────────────
    if args.mode in ("vectorized", "sweep", "wfo"):
        import pandas as pd
        from data.ingestion import get_bars

        price_dict = {}
        for sym in args.symbol:
            df = get_bars(sym, start_dt, end_dt, adjusted=True, db_path=db_path)
            if not df.empty:
                col = "adj_close" if "adj_close" in df.columns else "close"
                price_dict[sym] = df[col]
            else:
                logger.warning(f"No data for {sym}. Run bootstrap_data.py first.")
        if not price_dict:
            logger.error("No price data. Run: python scripts/bootstrap_data.py")
            sys.exit(1)
        prices = pd.DataFrame(price_dict).dropna(how="all")

        SIGNAL_MAP = {
            "sma": "sma_crossover_signal",
            "rsi": "rsi_signal",
            "momentum": "momentum_signal",
            "tsmom": "momentum_signal",
            "bollinger": "bollinger_signal",
            "donchian": "donchian_signal",
        }
        signal_name = SIGNAL_MAP.get(args.strategy[0], "sma_crossover_signal")
        from backtesting import vectorized as vmod
        signal_fn = getattr(vmod, signal_name, vmod.sma_crossover_signal)

        DEFAULT_GRIDS = {
            "sma_crossover_signal": {"fast": [10, 20, 50], "slow": [100, 150, 200]},
            "rsi_signal": {"period": [10, 14, 21], "oversold": [25, 30]},
            "momentum_signal": {"lookback": [126, 189, 252], "skip": [21]},
            "bollinger_signal": {"window": [10, 20, 25], "k": [1.5, 2.0]},
            "donchian_signal": {"period": [15, 20, 30, 55]},
        }
        param_grid = DEFAULT_GRIDS.get(signal_name, {"fast": [20, 50], "slow": [100, 200]})

        bench = prices.get("SPY", prices.iloc[:, 0])

        if args.mode == "vectorized":
            from backtesting.vectorized import VectorizedBacktester
            bt = VectorizedBacktester(prices=prices, initial_capital=capital)
            results = bt.run(signal_fn, benchmark=bench)
            _print_results(results)
            _save_results(results, args.output)

        elif args.mode == "sweep":
            from backtesting.vectorized import VectorizedBacktester
            bt = VectorizedBacktester(prices=prices, initial_capital=capital)
            df = bt.parameter_sweep(signal_fn, param_grid, benchmark=bench)
            logger.info(f"\nTop 5 configurations (by Deflated Sharpe):\n{df.head().to_string()}")
            _save_results({"sweep_results": df.to_dict("records")}, args.output)

        elif args.mode == "wfo":
            from backtesting.walk_forward import WalkForwardOptimizer
            wfo = WalkForwardOptimizer(
                prices=prices,
                train_years=args.train_years,
                test_months=args.test_months,
            )
            results = wfo.optimize_and_evaluate(signal_fn, param_grid, benchmark=bench)
            logger.info("\n" + "=" * 60)
            logger.info(f"OOS Sharpe: {results['avg_oos_sharpe']:.3f}")
            logger.info(f"DSR corrected: {results['deflated_sharpe_corrected']:.3f}")
            logger.info(f"VERDICT: {results['summary']['message']}")
            _save_results(results, args.output)
        return

    # ── Event-Driven mode ────────────────────────────────────────────────────
    from strategies.registry import StrategyRegistry, EnsembleEngine
    from core.event_queue import EventQueue
    from backtesting.execution import (
        NoSlippage, FixedSpreadSlippage, VolumeProportionalSlippage,
        ZeroCommission, PerShareCommission, FillModel,
    )
    from backtesting.engine import BacktestEngine

    slippage_map = {
        "none": NoSlippage(),
        "fixed": FixedSpreadSlippage(),
        "volume": VolumeProportionalSlippage(),
    }
    commission_map = {"zero": ZeroCommission(), "per_share": PerShareCommission()}

    eq = EventQueue()
    strategies = []
    for sname in args.strategy:
        try:
            s = StrategyRegistry.build(sname, asset_ids=args.symbol, event_queue=eq)
            strategies.append(s)
        except ValueError as e:
            logger.error(str(e))
            sys.exit(1)

    sizer = None
    if args.optimizer:
        from portfolio.optimization import PortfolioOptimizationStrategy
        opt_strat = PortfolioOptimizationStrategy(optimizer_name=args.optimizer)
        logger.info(f"Portfolio optimizer: {args.optimizer}")

    engine = BacktestEngine(
        strategies=strategies,
        start=start_dt,
        end=end_dt,
        initial_capital=capital,
        slippage_model=slippage_map[args.slippage],
        commission_model=commission_map[args.commission],
        db_path=db_path,
        verbose=args.verbose,
    )

    results = engine.run()

    if args.validate:
        _validate_walking_skeleton(results)

    _print_results(results)
    _save_results(results, args.output)

    # Generate HTML tearsheet if requested
    if args.tearsheet:
        try:
            from reporting.analytics import load_equity_curve_from_db, load_trades_from_db
            from reporting.tearsheet import generate_tearsheet
            equity = load_equity_curve_from_db(db_path)
            trades = load_trades_from_db(db_path)
            strat_label = ' + '.join(args.strategy) if isinstance(args.strategy, list) else args.strategy
            generate_tearsheet(
                equity_curve=equity,
                strategy_name=f"{strat_label} on {' '.join(args.symbol)}",
                trades_df=trades if not trades.empty else None,
                output_path=args.tearsheet,
                n_strategies_tested=1,
            )
            logger.info(f'Tearsheet saved to {args.tearsheet}')
        except Exception as e:
            logger.warning(f'Tearsheet generation failed: {e}')


def _print_results(results):
    meta = results.get("_meta", {})
    logger.info("\n" + "=" * 50)
    logger.info("BACKTEST RESULTS")
    logger.info("=" * 50)
    if "total_return" in results:
        logger.info(f"  Total Return:  {results['total_return']:+.2%}")
    if "cagr" in results:
        logger.info(f"  CAGR:          {results['cagr']:+.2%}")
    if "sharpe_ratio" in results:
        logger.info(f"  Sharpe:        {results['sharpe_ratio']:.3f}")
    if "deflated_sharpe_ratio" in results:
        dsr = results['deflated_sharpe_ratio']
        sig = "✓ significant" if dsr > 0 else "✗ not significant"
        logger.info(f"  Deflated SR:   {dsr:.3f} ({sig})")
    if "max_drawdown" in results:
        logger.info(f"  Max Drawdown:  {results['max_drawdown']:.2%}")
    if "calmar_ratio" in results:
        logger.info(f"  Calmar:        {results['calmar_ratio']:.3f}")
    if "n_trades" in results:
        logger.info(f"  Trades:        {results['n_trades']}")
        if results.get("insufficient_sample_warning"):
            logger.warning(f"  ⚠ {results.get('sample_warning_message', '')}")
    logger.info("=" * 50)


def _validate_walking_skeleton(results):
    total = results.get("total_return", 0)
    logger.info("\n" + "=" * 60)
    logger.info("WALKING SKELETON VALIDATION RESULT")
    logger.info(f"  Your total return: {total:.2%}")
    logger.info(f"  Expected range:    50% – 80%")
    if 0.40 <= total <= 0.90:
        logger.info("  ✓ PASS: Proceed to Phase 2")
    else:
        logger.warning("  ✗ FAIL: Check adjustment factors and dividend handling")
    logger.info("=" * 60)


def _save_results(results, path):
    if not path:
        return
    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        if hasattr(obj, "isoformat"):
            return obj.isoformat()
        if isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        return str(obj)
    with open(path, "w") as f:
        json.dump(_clean(results), f, indent=2)
    logger.info(f"Results saved to {path}")


if __name__ == "__main__":
    main()
