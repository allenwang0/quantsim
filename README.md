# quantsim

A daily-bar backtesting framework in Python with two engines: an event-driven one that models orders and fills, and a Numba-vectorized one for fast parameter sweeps. It is built to make lookahead and overfitting hard to get away with.

![Backtest report built from synthetic data](docs/img/report.png)

Live report: <https://allenwang0.github.io/quantsim/>, rebuilt from seeded synthetic data on every push to `main`.

## Highlights

- **Two independent engines that agree.** A parity test runs one SMA crossover through both and requires matching round-trip counts and total returns within 0.5 percentage points.
- **Up to 500x faster vectorized.** 50 assets over 10 years: 0.11 s vectorized versus 54.7 s event-driven.
- **No lookahead by default.** A signal on bar T fills at T+1's open. The report reruns each strategy with same-bar fills to measure the bias; for the SMA crossover it inflates total return by 1.3 points.
- **Overfitting guard.** Walk-forward parameter selection, a bootstrap Sharpe interval, a cost sweep and the deflated Sharpe ratio, which discounts the best Sharpe by how many parameter sets were tried.
- **Paper trading through Alpaca**, with live trading behind two explicit opt-ins.

## Quickstart

Needs [uv](https://docs.astral.sh/uv/) and Python 3.11 or 3.12.

```bash
git clone https://github.com/allenwang0/quantsim.git && cd quantsim
make setup
uv run python tests/fixtures/make_fixture.py
uv run quantsim report --prices tests/fixtures/synthetic_prices.csv --strategy sma momentum --out reports/demo
```

Open `reports/demo/index.html`. Tests: `uv run pytest -m "not live and not browser"`.

On real prices (yfinance, cached in SQLite at `~/.quantsim/quantsim.db`):

```bash
uv run quantsim bootstrap --symbols SPY --start 2015-01-01
uv run quantsim backtest --mode sweep --strategy sma --symbol SPY --start 2015-01-01
uv run quantsim wfo --signal sma --symbol SPY --start 2015-01-01
uv run quantsim paper --strategy sma --symbol SPY
```

`quantsim <command> --help` lists every option.

## How it works

```
event-driven engine (BacktestEngine)
  session T:   BarEvent -> Strategy (sees bars through T) -> SignalEvent
               -> PortfolioManager -> OrderEvent, held by the execution handler
  session T+1: order fills at T+1's open (plus slippage and commission)
               -> FillEvent -> Portfolio

vectorized engine (VectorizedBacktester)
  signal function -> position wanted at T's close, using data through T
  Numba simulation -> trades that position at T+1's price
```

- **Fill timing.** `FillModel.NEXT_BAR_OPEN` is the default; `IMMEDIATE_CLOSE` is opt-in. `SAME_BAR_OPEN_LOOKAHEAD` reproduces the pre-3.0 bug for the report's comparison only.
- **Vectorized simulation.** Positions are sized from equity on a direction change and held until the signal changes. No asset trades on a forward-filled price; `calendar="NYSE"` aligns rows to exchange sessions.
- **Walk-forward.** `WalkForwardOptimizer` fits on a training window and scores only the next test window, starting the run at the training start so indicators are warm.
- **Deflated Sharpe** (Bailey and Lopez de Prado, 2014): the probability that the true Sharpe beats the best Sharpe expected from that many unskilled trials. DSR > 0.95 counts as significant.

Version 3.0.0 fixed same-bar fills, per-bar position resizing, a two-bar signal delay, a DSR on the wrong scale and analytics computed on monthly snapshots, each with a test that fails on the old code. See [CHANGELOG.md](CHANGELOG.md).

## Results

Engine benchmark (`make bench`, [benchmarks/results.json](benchmarks/results.json)), median of 5 runs on 2520 daily bars, SMA 50/200, Apple M5 Pro:

| Assets | Vectorized (s) | Event-driven (s) | Event / vectorized |
|---:|---:|---:|---:|
| 1 | 0.1107 | 0.98 | 9x |
| 10 | 0.1059 | 11.96 | 113x |
| 50 | 0.1090 | 54.73 | 502x |

Walk-forward on the quickstart data (synthetic GBM, seven one-year test windows 2016-11-25 to 2023-08-30, 0 bps):

| Strategy | Trials | Out-of-sample return | Buy and hold | OOS Sharpe (95% interval) | DSR |
|---|---:|---:|---:|---:|---:|
| SMA crossover | 28 | -3.6% | -19.3% | -0.33 (-1.06 to 0.47) | 0.0 |
| Time-series momentum | 14 | -3.9% | -19.3% | -0.36 (-0.97 to 0.34) | 0.0 |

The synthetic data is a random walk with no edge to find, so this is a test of the guard: after 28 and 14 trials the deflated Sharpe correctly rejects both strategies.

## Limitations

- Daily bars only; no options backtesting.
- The default `bootstrap` universe is today's large caps, so long backtests carry survivorship bias.
- yfinance is unofficial; check its prices.
- Sharpe subtracts a 5% risk-free rate by default while cash earns nothing, so mostly-cash strategies show negative Sharpe.
- `quantsim paper` simulates fills locally without Alpaca keys and uses the Alpaca paper account with them. Live trading needs both `ALPACA_PAPER=false` and `QUANTSIM_ALLOW_LIVE=1`.

## Related

- [statarb](https://github.com/allenwang0/statarb): Kalman-filter pairs trading. quantsim's `PairsTradingStrategy` passes the same Kalman test cases.
- [volatilityscanner](https://github.com/allenwang0/volatilityscanner): option pricing and implied volatility.

MIT licensed. See [LICENSE](LICENSE).
