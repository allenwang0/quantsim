# quantsim

A Python backtesting framework for daily stock prices with two engines: an event-driven one that models orders and fills, and a fast vectorized one for testing many parameter settings. It is built to make two mistakes hard: trading on prices you could not have known yet, and overfitting to the past.

![Backtest report: Sharpe intervals include zero](docs/img/report.png)

The [report](https://allenwang0.github.io/quantsim/) is a control: on synthetic prices the checks correctly find no edge.

## Highlights

- **Two independent engines that agree:** a parity test requires matching trade counts and returns within 0.5 percentage points.
- **Vectorized engine** (Numba): 8.8x faster than event-driven on one asset over 10 years (0.11 s vs 0.98 s); 502x at 50 assets, since its time stays flat. One laptop run (Apple M5 Pro).
- **No lookahead by default:** signals fill at the next day's open; the report measures what same-day fills would change.
- **Overfitting checks:** walk-forward testing, a bootstrap Sharpe interval, and a deflated Sharpe check that discounts for how many variants were tried.

## Run it

```bash
git clone https://github.com/allenwang0/quantsim.git && cd quantsim
make setup
uv run python tests/fixtures/make_fixture.py
uv run quantsim report --prices tests/fixtures/synthetic_prices.csv --strategy sma momentum --out reports/demo
open reports/demo/index.html
```

## How it works

- Event-driven engine: each bar goes to the strategy; its signals become orders that fill at the next open, with slippage and commission.
- Vectorized engine: a signal sets the target position at each close; a Numba loop trades it at the next bar.
- Walk-forward testing picks parameters on a training window and scores them on the next, unseen one.
- Real prices come from yfinance and are cached locally; paper trading runs through Alpaca.

## Limitations

- Daily bars only; no options backtesting.
- yfinance data is unofficial, and the default stock list is today's large caps, so long backtests carry survivorship bias.
- Paper trading by default; live trading needs two explicit opt-ins. Not financial advice.
