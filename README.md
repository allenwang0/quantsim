# quantsim

A Python backtester for daily stock prices with two engines that agree: an event-driven one that models orders and fills, and a vectorized one for sweeping parameters. It is built to make lookahead and overfitting hard to commit by accident.

**[Read the report →](https://allenwang0.github.io/quantsim/)** measurements, methodology and limits, built from committed results.

![quantsim's own HTML backtest report on synthetic data](docs/img/report.png)

## Highlights

<!-- results:start -->
- **Two engines that agree:** equal trade counts and total returns within 0.5 pp in the parity test.
- **Vectorized engine 460× faster:** 0.10 s vs 45.5 s for 50 assets × 10 years (median of 5 runs, synthetic data, one strategy, Apple M5 Pro).
- **No lookahead:** orders fill at the next bar's open; same-bar fills would move the demo's return by +1.3 pp.
- **Overfitting checks:** walk-forward, bootstrap Sharpe intervals and a deflated Sharpe that discounts for 28 trials (0.00 on synthetic data: no edge certified).
- **Costs on by default:** 0.1% commission + 0.1% slippage per trade.
<!-- results:end -->

## Run it

```bash
git clone https://github.com/allenwang0/quantsim.git && cd quantsim
make setup
uv run python tests/fixtures/make_fixture.py
uv run quantsim report --prices tests/fixtures/synthetic_prices.csv --strategy sma momentum --out reports/demo
open reports/demo/index.html
```

## How it works

- Event-driven: each bar goes to the strategy; its signal becomes an order that fills at the next open with slippage and commission.
- Vectorized: a signal function sets the target position at each close; a Numba loop trades it at the next bar.
- Walk-forward picks parameters on a training window and scores them only on the next unseen window.
- Prices come from yfinance, cached locally; paper trading runs through Alpaca.

## Limitations

- Published numbers come from seeded synthetic prices, one machine and one strategy.
- Bars are stored unadjusted; event-engine bars carry the raw open but the adjusted close.
- Daily bars only, no options. The default stock list is today's large caps (survivorship bias).
- Paper trading by default; live trading needs two opt-ins. Not financial advice.
