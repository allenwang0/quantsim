# report-kit v1

Shared tokens, components and SVG charts for the project reports. One
self-contained HTML page per repo: the generator inlines `report-kit.css`.

| File | What |
| --- | --- |
| `report-kit.css` | Tokens (light/dark, `data-theme` override), bento grid, components, chart classes, print |
| `report_kit.py` | Python 3.10+ stdlib: `page()`, components, `bar_chart`, `ladder_chart`, `line_chart` |
| `check_kit.py` | Fails if any file differs from the sha256 pinned in `KIT_VERSION` |
| `KIT_VERSION` | `v1`, then `sha256  filename` per file |

## Vendor it

Copy the whole directory, unchanged, to `report-kit/` at the repo root. Never
edit a vendored copy; change the canonical kit, bump `KIT_VERSION`, re-vendor.

```python
sys.path.insert(0, str(ROOT / "report-kit"))
import report_kit as rk
html = rk.page(title=..., description=..., canonical=..., header=..., main=..., footer=...)
```

Text arguments are escaped; pass HTML as `rk.Raw`. Every helper returns `Raw`.

## Check it

```sh
python3 report-kit/check_kit.py                 # CI: hashes match KIT_VERSION
python3 report-kit/check_kit.py --canonical DIR  # also identical to the canonical copy
```
