# report-kit v1.1

Shared tokens, components and SVG charts for the project reports. One
self-contained HTML page per repo: the generator inlines `report-kit.css`.

| File | What |
| --- | --- |
| `report-kit.css` | Tokens (light/dark, `data-theme` override), bento grid, components, chart classes, print |
| `report_kit.py` | Python 3.10+ stdlib: `page()`, components, `bar_chart`, `ladder_chart`, `line_chart` |
| `check_kit.py` | Fails if any file differs from the sha256 pinned in `KIT_VERSION` |
| `KIT_VERSION` | `v1.1`, then `sha256  filename` per file |

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

## API (v1.1 additions marked *)

| Call | Use |
| --- | --- |
| `page`, `report_header`, `section`, `bento`, `metric`, `comparison`, `note`, `code` | Shell and components |
| `hero(eyebrow, title, subtitle, value=None, label=None, support, context, actions, figure)` * | `value=None` drops the metric (compact briefs); `figure` fills the right column. Content is top-aligned |
| `report_nav(items)` | Sticky section nav, Tier A reports |
| `toc(items)` * | Inline, non-sticky contents list, Tier B briefs; same type and divider as `report_nav` |
| `num(value, unit)` * | `"55.6", "M/s"` -> `55.6\u00a0M/s`; the unit never wraps away. Format the number first |
| `figure`, `variants`, `table`, `data_table` | Figure panel; per-breakpoint charts; tables (caption * now sits above the scroll wrapper) |
| `methodology`, `limitations`, `reproduce`, `ref`, `footer` | Back matter |
| `bar_chart`, `ladder_chart`, `line_chart` | Charts. `bar_chart` below 420 px stacks labels; * its zero line no longer runs through them |
| `legend(items, width=, x=, y=)` * | Wrapping key: items `(name, "line"/"dot", accent[, dashed])`; returns `(svg, height)` |
| `svg`, `svg_text`, `text_width`, `nice_ticks`, `log_ticks` | Low-level SVG |

Screenshots and GIFs: put an `<img>` or `<picture>` inside a `figure(...)` chart slot or the hero
`figure`; the kit sizes it (max-width 100%, small radius, border token). No inline styles.

Project-specific charts (dumbbell, strip, ECDF, histogram, interval) stay in the generator.

### `toc` or `report_nav`?

Tier A report (long, many sections, readers jump around): `report_nav`, sticky. Tier B brief
(short page, one screen of contents): `toc`, placed right after the hero. Never both.

### Upgrading from v1

Drop-in: no signature removed. Visible changes: hero content top-aligns (was bottom), table
captions render as `<p class="table-caption">` above the scroll area, stacked `bar_chart` draws
one short zero tick per bar.
