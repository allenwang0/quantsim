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
