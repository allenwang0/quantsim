"""
Run live paper trading from the command line.

Usage:
  quantsim paper --strategy sma --symbol SPY QQQ
  quantsim paper --strategy dual_momentum

  # Multi-strategy ensemble
  quantsim paper --strategy sma ema macd --symbol SPY --ensemble

The dashboard runs as a separate process:
  streamlit run src/quantsim/dashboard/app.py

Alpaca credentials must be set:
  export ALPACA_API_KEY=your_key
  export ALPACA_SECRET_KEY=your_secret
"""

import argparse
import asyncio
import logging
import os
import sys

from quantsim.core.logging_config import setup_logging

setup_logging()

logger = logging.getLogger("paper_trading")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="quantsim paper", description="QuantSim Paper Trading")
    parser.add_argument("--strategy", nargs="+", default=["sma"], help="Strategy name(s)")
    parser.add_argument(
        "--symbol", "--symbols", nargs="+", default=["SPY"], help="Symbol(s) to trade"
    )
    parser.add_argument(
        "--capital", type=float, default=None, help="Initial capital (default: from config)"
    )
    parser.add_argument(
        "--ensemble", action="store_true", help="Use ensemble aggregation for multiple strategies"
    )
    parser.add_argument(
        "--ensemble-method",
        default="confidence_weighted",
        choices=["majority_vote", "confidence_weighted", "orthogonality_checked"],
        help="Signal aggregation method",
    )
    parser.add_argument("--db", default=None, help="Database path override")
    args = parser.parse_args(argv)

    from quantsim.core.config import config
    from quantsim.core.database_v2 import init_full_db
    from quantsim.core.event_queue import EventQueue
    from quantsim.strategies.registry import StrategyRegistry

    db_path = args.db or config.db_path
    if args.db:
        os.environ["QUANTSIM_DB"] = args.db

    init_full_db(db_path)

    # Validate Alpaca config
    if not config.alpaca_configured:
        logger.warning(
            "ALPACA_API_KEY/ALPACA_SECRET_KEY not set. "
            "Running in simulation mode (fills simulated locally, no real paper account)."
        )

    # Build strategies
    # Each strategy gets its own event queue initially; the engine will coordinate them

    event_queue = EventQueue()

    strategies = []
    for strategy_name in args.strategy:
        try:
            s = StrategyRegistry.build(
                name=strategy_name,
                asset_ids=args.symbol,
                event_queue=event_queue,
            )
            strategies.append(s)
            logger.info(f"Built strategy: {strategy_name} on {args.symbol}")
        except Exception as e:
            logger.error(f"Failed to build strategy '{strategy_name}': {e}")
            sys.exit(1)

    if not strategies:
        logger.error("No strategies built. Exiting.")
        sys.exit(1)

    # Build engine
    from quantsim.paper_trading.engine import PaperEngine

    engine = PaperEngine(
        strategies=strategies,
        initial_capital=args.capital,
        use_ensemble=args.ensemble or len(strategies) > 1,
        ensemble_method=args.ensemble_method,
        db_path=db_path,
    )

    logger.info("=" * 60)
    logger.info("QUANTSIM PAPER TRADING")
    logger.info(f"Strategies: {[s.strategy_id for s in strategies]}")
    logger.info(f"Universe: {args.symbol}")
    logger.info(f"Capital: ${engine.initial_capital:,.0f}")
    logger.info(f"DB: {db_path}")
    logger.info(f"Alpaca: {'configured' if config.alpaca_configured else 'simulation'}")
    logger.info("=" * 60)
    logger.info("Press Ctrl+C to stop gracefully.")
    logger.info("Run dashboard in separate terminal: streamlit run src/quantsim/dashboard/app.py")
    logger.info("=" * 60)

    try:
        asyncio.run(engine.run())
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")


if __name__ == "__main__":
    main()
