# quantsim

A Python backtesting framework for daily stock prices with two engines: an event-driven one that models orders and fills, and a fast vectorized one for testing many parameter settings. It is built to make two mistakes hard: trading on prices you could not have known yet, and overfitting to the past.

![Backtest report built from synthetic data](docs/img/report.png)

## Highlights

- **Two independent engines that agree.** A parity test runs the same strategy through both and requires matching trade counts and returns within 0.5 percentage points.
- **Up to 500x faster vectorized engine** (Numba): 50 assets over 10 years in 0.11 s versus 54.7 s event-driven.
- **No lookahead by default.** A signal on one day fills at the next day's open; the report measures how much same-day fills would change returns.
- **Overfitting checks:** walk-forward testing, a bootstrap confidence interval on the Sharpe ratio, and a deflated Sharpe check that discounts the Sharpe ratio for how many strategy variants were tried.

## Run it

```bash
git clone https://github.com/allenwang0/quantsim.git && cd quantsim
make setup
uv run python tests/fixtures/make_fixture.py
uv run quantsim report --prices tests/fixtures/synthetic_prices.csv --strategy sma momentum --out reports/demo
open reports/demo/index.html
```

## How it works

- Event-driven engine: each day's bar goes to the strategy, which emits signals; the portfolio turns them into orders that fill at the next open with slippage and commission.
- Vectorized engine: a signal function sets the position wanted at each close, and a Numba loop trades it at the next bar's price.
- Walk-forward testing picks parameters on a training window and scores them only on the following unseen window.
- Real prices come from yfinance and are cached locally; paper trading runs through Alpaca.

## Limitations

- Daily bars only; no options backtesting.
- yfinance data is unofficial, and the default stock list is today's large caps, so long backtests carry survivorship bias.
- Paper trading by default; live trading needs two explicit opt-ins. Not financial advice.
