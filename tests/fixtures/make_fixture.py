"""
Write the synthetic price fixture used by the demo report and the README quickstart.

    python tests/fixtures/make_fixture.py [--out tests/fixtures/synthetic_prices.csv]

Seeded GBM with regime shifts (quantsim.data.synthetic). The CSV is generated,
not committed; a JSON sidecar records the seed and generator so every page
built from it can say where the numbers came from.
"""

import argparse
import json
from pathlib import Path

from quantsim.data.synthetic import regime_gbm_prices

SEED = 20240611
N_ASSETS = 5
N_BARS = 2520


def main(argv=None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--out", default=str(Path(__file__).with_name("synthetic_prices.csv")))
    args = parser.parse_args(argv)

    prices, regimes = regime_gbm_prices(n_assets=N_ASSETS, n_bars=N_BARS, seed=SEED)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prices.join(regimes).to_csv(out, index_label="date")
    out.with_suffix(".json").write_text(
        json.dumps(
            {
                "synthetic": True,
                "generator": "make_fixture.py",
                "seed": SEED,
                "description": f"Seeded synthetic GBM with regime shifts, seed {SEED}",
                "n_assets": N_ASSETS,
                "n_bars": N_BARS,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Wrote {out} ({N_BARS} bars x {N_ASSETS} assets, synthetic, seed {SEED})")
    return out


if __name__ == "__main__":
    main()
