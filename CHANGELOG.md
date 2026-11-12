# Changelog

## 3.0.0 (unreleased)

Fixes in this section change reported backtest results.

### Fixed

- Event engine: orders created on bar T now fill at bar T+1's open. Before,
  they filled at bar T's open, a price from before the signal existed, which
  overstated returns for any strategy that reacts to the close.
  `IMMEDIATE_CLOSE` still fills at bar T's close as an explicit opt-in.
- Paper trading: `ALPACA_PAPER` accepts only `true` or `false`; unset means
  paper. Live mode also needs `QUANTSIM_ALLOW_LIVE=1`. Before, any value other
  than `true` (for example `1` or `yes`) selected a live account.
- Vectorized simulator: positions are sized from equity at entry and kept
  while the signal direction holds. Before, a held position was resized from
  remaining cash every bar, which inflated trade counts and cost drag.
- Vectorized signal functions no longer shift their output; the simulator
  applies the single one-bar delay. Before, signals were delayed two bars.
- Deflated Sharpe ratio follows Bailey and Lopez de Prado (2014): it uses the
  per-period Sharpe, scales the expected maximum Sharpe by the variance of
  the trial Sharpes, and returns a probability in [0, 1].
  `deflated_sharpe_significant` now means DSR > 0.95. Before, an annualized
  Sharpe was mixed with a daily sample size and the result was a z-score
  tested against 0.
- Sortino ratio is the mean excess return over the downside deviation
  across all periods, annualized by sqrt(252). Before, it divided CAGR by
  the standard deviation of negative days only.
- Pairs strategy: the Kalman hedge ratio updates once per date with both
  legs' prices from that date, and is invariant to the price level. Before,
  it updated twice per date, the first time pairing one leg's new price
  with the other's old one, which also halved the effective z-score window.
  The strategy no longer claims a cointegration test it does not run.
- Portfolio: a fill that flips a position from long to short (or back)
  now sets the new position's cost basis to the fill price. Before, the
  new position kept the old cost, so its unrealized P&L was wrong and
  equity no longer equalled initial capital plus realized and unrealized
  P&L.
- Vectorized engine: no trade happens on a forward-filled (stale) price;
  the position is taken on the asset's next real bar. `calendar="NYSE"`
  aligns rows to exchange sessions.
