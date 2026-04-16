# QuantSim v2

Production-grade paper trading and backtesting platform. Zero paid data dependencies.

## What's inside

```
quantsim/
├── core/               events, event queue, database (v1+v2 schemas), config system
├── data/               yfinance, FRED, Alpaca data; options BSM reconstruction
├── strategies/         strategy library, registry, ensemble engine, GARCH vol
├── backtesting/        event-driven engine, vectorized engine, walk-forward optimizer
├── portfolio/          portfolio tracking, HRP/Risk Parity/Black-Litterman/Mean-Variance
├── risk/               circuit breakers, Greeks limits, drawdown monitoring
├── reporting/          full analytics suite, Monte Carlo CI, factor decay, attribution
├── paper_trading/      Alpaca integration, paper trading engine (asyncio)
├── dashboard/          Streamlit 3-view dashboard (separate process)
├── tests/              86 tests (unit + integration + e2e)
└── scripts/            CLI runners for all modes
```

## Architecture

**Two backtesting engines, one strategy codebase:**

```
[Event-Driven Engine]          [Vectorized Engine]
  BarEvent → Strategy            NumPy price matrix
  → SignalEvent → Order          → signal function
  → FillEvent → Portfolio        → Numba JIT simulation
  
  ~2s/year-of-data              ~1ms/year-of-data
  Realistic fills/slippage      Best for parameter sweeps
  Live paper trading parity     100-1000x faster
```

Both engines use identical strategy code. Signal functions are pure NumPy for the vectorized engine; the same logic wraps into strategy classes for the event-driven engine.

## Quick Start

### Phase 0: Walking skeleton validation (always first)
```bash
pip install -r requirements.txt
python scripts/bootstrap_data.py --symbols SPY --start 2020-01-01
python scripts/run_backtest.py --validate
# Expected: SPY total return 2020-2023 ≈ 50-80%. If it fails, you have a data bug.
```

### Bootstrap full universe
```bash
python scripts/bootstrap_data.py --start 2010-01-01
# Downloads ~80 symbols + all FRED macro series. ~10 minutes.
```

### Run backtests
```bash
# Event-driven (default mode)
python scripts/run_backtest.py --strategy sma --symbol SPY --start 2015-01-01

# Vectorized (fast research mode)
python scripts/run_backtest.py --mode vectorized --strategy sma --symbol SPY QQQ

# Parameter sweep
python scripts/run_backtest.py --mode sweep --strategy sma --symbol SPY

# Walk-forward optimization (out-of-sample only)
python scripts/run_backtest.py --mode wfo --strategy sma --symbol SPY

# With portfolio optimization overlay
python scripts/run_backtest.py --strategy tsmom --optimizer hrp --symbol SPY QQQ GLD TLT IWM

# Multi-strategy ensemble
python scripts/run_backtest.py --strategy sma ema macd --ensemble --symbol SPY QQQ
```

### Run walk-forward optimizer directly
```bash
python scripts/run_wfo.py --signal sma --symbol SPY --train-years 3 --test-months 12
python scripts/run_wfo.py --mode sweep --signal momentum --symbol SPY QQQ GLD TLT IWM
```

### Start paper trading
```bash
export ALPACA_API_KEY=your_key
export ALPACA_SECRET_KEY=your_secret
python scripts/run_paper_trading.py --strategy sma --symbol SPY QQQ
# Without Alpaca credentials: simulates fills locally (still useful for testing)
```

### Dashboard (separate terminal)
```bash
streamlit run dashboard/app.py
# Opens at http://localhost:8501
# Reads from DB every 2s. Never blocks trading loop.
```

### Run all tests
```bash
pytest tests/ -v                    # 86 tests
pytest tests/ -m "not live" -v     # skip tests requiring live data
```

## Strategy Library

| Name | Type | Description | Validated Range |
|------|------|-------------|-----------------|
| `buy_and_hold` | passive | Walking skeleton baseline | — |
| `sma` | trend | SMA crossover (50/200) | Sharpe 0.4–0.7 on SPY |
| `ema` | trend | EMA crossover (12/26) | — |
| `macd` | trend | MACD signal line | — |
| `donchian` | trend | Donchian breakout (Turtle) | Use on diversified basket |
| `adx` | trend | ADX-filtered trend | — |
| `tsmom` | trend | Time-series momentum 12-1 | Moskowitz et al. 2012 |
| `bollinger` | mean-rev | Bollinger Band z-score | ADX < 20 filter required |
| `rsi` | mean-rev | RSI oversold/overbought | SMA 200 trend filter |
| `xs_momentum` | momentum | Jegadeesh-Titman | Sharpe 0.3–0.5 after costs |
| `dual_momentum` | momentum | Antonacci dual momentum | Monthly rebalance |
| `low_vol` | factor | Low volatility factor | Baker et al. 2011 |
| `covered_call` | options | Covered call income | IVR > 30, LOW_VOL |
| `iron_condor` | options | Iron condor theta decay | IVR > 50, LOW_VOL |
| `long_straddle` | options | Long vol event play | VIX > 25 |

## Portfolio Optimizers

All plug into the backtesting engine as position sizing overlays:

| Optimizer | When to use |
|-----------|-------------|
| `hrp` | Default. No matrix inversion. Robust OOS. |
| `risk_parity` | Equal risk contribution. No expected return needed. |
| `mean_variance` | When you have high-confidence return estimates. Fragile. |
| `black_litterman` | Incorporating specific market views into equilibrium. |
| `equal_weight` | Simplest baseline. Surprisingly hard to beat. |

## Data Sources

| Source | Data | Free tier |
|--------|------|-----------|
| yfinance | OHLCV, dividends, splits, current options chains | Unlimited (unofficial) |
| FRED | VIX, yields, HY spreads, CPI, fed funds | 120 req/min (set FRED_API_KEY) |
| Alpaca | Historical bars 2002+, paper trading, WebSocket | Free paper account |
| Stooq | Bulk historical CSV | No limit |

## Performance Analytics Output

Every backtest produces the full suite:
- Total return, CAGR, monthly distribution
- Sharpe, Sortino, Calmar, max drawdown, drawdown duration
- **Deflated Sharpe Ratio** (corrects for number of parameter configurations tested)
- **Monte Carlo 95% CI** on Sharpe (bootstrap)
- Win rate, profit factor, expectancy, average holding period
- Alpha, beta, information ratio vs SPY benchmark
- Rolling 12-month Sharpe (regime drift detection)
- Factor decay curves (ML strategies)
- Turnover and transaction cost drag

## Walk-Forward Optimization

The correct protocol per Bailey, Borwein, Lopez de Prado, Zhu (2014):

```
[──── TRAIN 3yr ────][TEST 1yr]
                 [──── TRAIN 3yr ────][TEST 1yr]
                              [──── TRAIN 3yr ────][TEST 1yr]

Reported Sharpe = average of TEST periods only (never train)
```

The **Deflated Sharpe Ratio** (DSR) adjusts for however many parameter combinations you tested. DSR > 0 is required before proceeding. A raw Sharpe of 1.2 after testing 100 configurations may have DSR < 0.

## Known Constraints

1. **Backtesting resolution is DAILY ONLY.** No free intraday historical data exists at scale.

2. **Options backtesting uses BSM-reconstructed chains.** IV reconstruction error ≈ ±20-30%. Results are indicative; validate forward in paper trading.

3. **Survivorship bias** in the default universe. Replace with historical Russell 3000 constituent lists for production research.

4. **Options transaction costs are material.** A 4-leg iron condor has 8 half-spreads (open + close). For illiquid options at 20% bid-ask spread, this alone consumes 80% of premium. The engine models this explicitly; many published retail results do not.

## Environment Variables

```bash
QUANTSIM_DB=/path/to/db.db       # database path (default: ~/.quantsim/quantsim.db)
QUANTSIM_CAPITAL=100000           # initial capital
QUANTSIM_SLIPPAGE=volume          # slippage model: none/fixed/volume
ALPACA_API_KEY=your_key           # paper trading (required for live mode)
ALPACA_SECRET_KEY=your_secret     # paper trading (required for live mode)
ALPACA_PAPER=true                 # use paper account (default: true)
FRED_API_KEY=your_key             # optional: improves FRED rate limits
QUANTSIM_ML=true                  # enable ML alpha features (requires lightgbm)
```

## References

- Faber (2007): "A Quantitative Approach to Tactical Asset Allocation"
- Jegadeesh & Titman (1993): "Returns to Buying Winners and Selling Losers"
- Moskowitz, Ooi, Pedersen (2012): "Time Series Momentum"
- Antonacci (2014): "Dual Momentum Investing"
- Baker, Bradley, Wurgler (2011): "Benchmarks as Limits to Arbitrage"
- Almgren & Chriss (2001): "Optimal Execution of Portfolio Transactions"
- Lopez de Prado (2016): "Building Diversified Portfolios that Outperform Out of Sample" (HRP)
- Bailey, Borwein, Lopez de Prado, Zhu (2014): "Pseudo-Mathematics and Financial Charlatanism" (DSR)
- Vidyamurthy (2004): "Pairs Trading: Quantitative Methods and Analysis"
- Black & Litterman (1992): "Global Portfolio Optimization"
