"""Build docs/index.html (the published project report) and the README results
block from committed results.

Usage:
  python3 scripts/build_report.py [--results results/demo/results.json]
      [--bench benchmarks/results.json] [--tests results/tests.json]
      [--out docs/index.html] [--readme README.md]

Python 3.10+ standard library only. Every number on the page is computed here
from the three committed result files or read from constants in the committed
sources (parity tolerance, default costs, walk-forward windows); none is typed
by hand. Page components come from the vendored report kit in report-kit/.

This page is separate from `quantsim report`, whose interactive Plotly page is
the tool's own output for any price series. This one tells the project's story
from one seeded run of that command, committed under results/demo/.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "report-kit"))
import report_kit as rk  # noqa: E402
from report_kit import Raw, esc  # noqa: E402

REPO = "allenwang0/quantsim"
REPO_URL = f"https://github.com/{REPO}"
PAGE_URL = "https://allenwang0.github.io/quantsim/"
HOME_URL = "https://allenwang0.github.io/"
PRIMARY, SECONDARY = "sma", "momentum"
README_START = "<!-- results:start -->"
README_END = "<!-- results:end -->"
THIN = " "  # narrow no-break space: "0.5 pp" never wraps
MINUS = "−"


# ---------------------------------------------------------------- formatting


def signed(value: float, digits: int = 1, unit: str = "") -> str:
    text = f"{abs(value):,.{digits}f}{unit}"
    if round(value, digits) == 0:
        return f"{0:.{digits}f}{unit}"
    return ("+" if value > 0 else MINUS) + text


def plain(value: float, digits: int = 2) -> str:
    """Number with a typographic minus, no plus sign."""
    text = f"{value:,.{digits}f}"
    return text.replace("-", MINUS)


def secs(value: float) -> str:
    if value >= 10:
        return f"{value:,.1f}{THIN}s"
    return f"{value:.2f}{THIN}s"


def ratio(value: float) -> str:
    if value >= 100:
        return f"{value:,.0f}×"
    if value >= 10:
        return f"{value:.1f}×"
    return f"{value:.2f}×"


def pct_pp(value: float) -> str:
    return f"{value * 100:g}{THIN}pp"


def params_text(params: dict) -> str:
    return ", ".join(f"{k} {v}" for k, v in params.items())


def nowrap(text: str) -> Raw:
    return Raw(f'<span class="nowrap">{esc(text)}</span>')


def assets_text(n: int) -> str:
    return "1 asset" if n == 1 else f"{n} assets"


# ---------------------------------------------------------------- inputs


def load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def source_text(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


def line_of(rel, needle, start_at=1):
    for number, line in enumerate(source_text(rel).splitlines(), 1):
        if number >= start_at and needle in line:
            return number
    raise ValueError(f"{needle!r} not found in {rel}")


def facts_from_sources():
    """Constants the page quotes, read from the code that uses them."""
    parity = source_text("tests/test_engine_parity.py")
    tol = re.search(r"abs\(event_return - vector_return\) < ([\d.]+)", parity)
    consts = re.search(r"FAST, SLOW, SIZE, CAPITAL = (\d+), (\d+), ([\d.]+), ([\d_.]+)", parity)
    bars = re.search(r"n_bars = (\d+)", parity)
    vec = source_text("src/quantsim/backtesting/vectorized.py")
    commission = re.search(r"commission_rate: float = ([\d.]+)", vec)
    slippage = re.search(r"slippage_pct: float = ([\d.]+)", vec)
    page = source_text("src/quantsim/reporting/report_page.py")
    wfo = re.search(
        r"WalkForwardOptimizer\(frame, train_years=(\d+), test_months=(\d+), step_months=(\d+)\)",
        page,
    )
    per_window = re.search(r"returns, n_simulations=(\d+), risk_free_rate", page)
    sims = re.search(r"BOOTSTRAP_SIMULATIONS = (\d+)", page)
    size = re.search(r"POSITION_SIZE = ([\d.]+)", page)
    engine = source_text("src/quantsim/backtesting/engine.py")
    k = re.search(
        r"def __init__\(self, k: float = ([\d.]+)\)",
        source_text("src/quantsim/backtesting/execution.py"),
    )
    missing = [
        name
        for name, m in (
            ("parity tolerance", tol),
            ("parity constants", consts),
            ("parity bars", bars),
            ("commission", commission),
            ("slippage", slippage),
            ("walk-forward", wfo),
            ("bootstrap", per_window),
            ("simulations", sims),
            ("position size", size),
            ("impact k", k),
        )
        if not m
    ]
    if missing or "VolumeProportionalSlippage()" not in engine:
        raise ValueError(f"source constants not found: {missing}")
    return {
        "parity_tol": float(tol.group(1)),
        "parity_fast": int(consts.group(1)),
        "parity_slow": int(consts.group(2)),
        "parity_bars": int(bars.group(1)),
        "commission": float(commission.group(1)),
        "slippage": float(slippage.group(1)),
        "train_years": int(wfo.group(1)),
        "test_months": int(wfo.group(2)),
        "step_months": int(wfo.group(3)),
        "window_sims": int(per_window.group(1)),
        "pooled_sims": int(sims.group(1)),
        "position_size": float(size.group(1)),
        "impact_k": float(k.group(1)),
    }


def git_last_commit(rel):
    try:
        out = subprocess.run(
            ["git", "log", "-1", "--format=%H", "--", rel],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return out or None


# ---------------------------------------------------------------- derived numbers


def derive(results, bench, tests):
    """Every number the page and README show, computed from the result files."""
    rows = bench["rows"]
    big = max(rows, key=lambda r: r["assets"])
    small = min(rows, key=lambda r: r["assets"])
    s = results["strategies"][PRIMARY]
    wf = s["walk_forward"]
    windows = wf["windows"]
    fill = s["fill_timing"]
    cost = s["cost_sensitivity"]
    default_i = (
        cost["cost_bps"].index(cost["default_bps"])
        if cost["default_bps"] in cost["cost_bps"]
        else None
    )
    span = lambda r, e: (min(r[f"{e}_runs_s"]), max(r[f"{e}_runs_s"]))  # noqa: E731
    return {
        "big": big,
        "small": small,
        "speedup": big["event_median_s"] / big["vectorized_median_s"],
        "speedup_floor": span(big, "event")[0] / span(big, "vectorized")[1],
        "vec_span": span(big, "vectorized"),
        "event_span": span(big, "event"),
        "event_growth": big["event_median_s"] / small["event_median_s"],
        "vec_growth": big["vectorized_median_s"] / small["vectorized_median_s"],
        "asset_growth": big["assets"] / small["assets"],
        "runs": bench["runs"],
        "fill": fill,
        "gap": fill["same_bar_open_return_pct"] - fill["next_bar_open_return_pct"],
        "windows": windows,
        "zero_in_ci": sum(1 for w in windows if w["ci_low"] <= 0 <= w["ci_high"]),
        "positive_windows": sum(1 for w in windows if w["oos_sharpe"] > 0),
        "dsr": wf["dsr"],
        "trials": wf["total_trials"],
        "configs": wf["n_configs"],
        "oos_return": wf["oos_return_pct"],
        "hold_return": wf["buy_and_hold_return_pct"],
        "cost": cost,
        "default_i": default_i,
        "tests": tests,
    }


# ---------------------------------------------------------------- charts

T = rk.svg_text
f = rk.coord


def year_frac(date: str) -> float:
    d = datetime.date.fromisoformat(date)
    start = datetime.date(d.year, 1, 1)
    days = (datetime.date(d.year + 1, 1, 1) - start).days
    return d.year + (d - start).days / days


def spread_labels(items, min_gap=15.0):
    """items: [(y, ...)], returns y positions pushed apart so labels do not overlap."""
    order = sorted(range(len(items)), key=lambda i: items[i])
    ys = [items[i] for i in order]
    for k in range(1, len(ys)):
        if ys[k] - ys[k - 1] < min_gap:
            ys[k] = ys[k - 1] + min_gap
    out = [0.0] * len(items)
    for k, i in enumerate(order):
        out[i] = ys[k]
    return out


def xy_chart(
    uid,
    width,
    height,
    series,
    *,
    title,
    desc,
    x_ticks,
    y_ticks,
    fmt_x,
    fmt_y,
    x_title,
    y_title,
    log_x=False,
    log_y=False,
    zero=False,
    vlines=(),
    legend=False,
    markers=False,
):
    """Line chart with optional markers, min-max whiskers, zero line and vertical markers.

    series: dicts name, points [(x, y)], accent, optional whiskers [(x, lo, hi)],
    optional end (label text at the last point). Direct end labels unless legend=True.
    """
    fx = (lambda v: math.log10(v)) if log_x else (lambda v: v)
    fy = (lambda v: math.log10(v)) if log_y else (lambda v: v)
    tick_w = max(rk.text_width(fmt_y(t), 12) for t in y_ticks)
    ml = tick_w + 14
    end_texts = [s.get("end", s["name"]) for s in series]
    mr = 16 if legend else max(rk.text_width(t) for t in end_texts) + 18
    rows_needed = 1
    if legend:
        kx = 0.0
        for s in series:
            w = 24 + rk.text_width(s["name"])
            if kx and kx + w > width:
                rows_needed, kx = rows_needed + 1, 0.0
            kx += w + 16
    mt = (32 + 20 * rows_needed) if legend else 34
    mb = 46
    x0, x1 = fx(x_ticks[0]), fx(x_ticks[-1])
    y0, y1 = fy(y_ticks[0]), fy(y_ticks[-1])
    px = lambda v: ml + (width - ml - mr) * (fx(v) - x0) / ((x1 - x0) or 1)  # noqa: E731
    py = lambda v: height - mb - (height - mt - mb) * (fy(v) - y0) / ((y1 - y0) or 1)  # noqa: E731
    parts = []
    for t in y_ticks:
        parts.append(f'<path class="c-grid" d="M{f(ml)} {f(py(t))}H{f(width - mr)}"/>')
        parts.append(T(ml - 8, py(t) + 4, fmt_y(t), "c-tick", "end"))
    for t in x_ticks:
        parts.append(T(px(t), height - mb + 18, fmt_x(t), "c-tick", "middle"))
    parts.append(f'<path class="c-axis" d="M{f(ml)} {f(height - mb)}H{f(width - mr)}"/>')
    if zero:
        parts.append(f'<path class="c-floor" d="M{f(ml)} {f(py(0))}H{f(width - mr)}"/>')
    for vx, label in vlines:
        parts.append(f'<path class="c-floor" d="M{f(px(vx))} {f(mt - 6)}V{f(height - mb)}"/>')
        parts.append(T(px(vx) + 6, height - mb - 8, label, "c-tick"))
    ordered = sorted(series, key=lambda s: bool(s.get("accent")))
    for s in ordered:
        stroke = "c-stroke-accent" if s.get("accent") else "c-stroke-muted"
        fill = "c-accent" if s.get("accent") else "c-muted"
        for x, lo, hi in s.get("whiskers", ()):
            parts.append(
                f'<path class="{stroke}" stroke-width="1.5" d="M{f(px(x))} {f(py(lo))}V{f(py(hi))}'
                f'M{f(px(x) - 4)} {f(py(lo))}H{f(px(x) + 4)}M{f(px(x) - 4)} {f(py(hi))}H{f(px(x) + 4)}"/>'
            )
        d = "".join(
            ("M" if k == 0 else "L") + f"{f(px(a))} {f(py(b))}"
            for k, (a, b) in enumerate(s["points"])
        )
        dash = ' stroke-dasharray="5 4"' if s.get("dash") else ""
        parts.append(
            f'<path class="{stroke}" stroke-width="{2 if s.get("accent") else 1.75}"{dash} d="{d}"/>'
        )
        if markers:
            for a, b in s["points"]:
                parts.append(
                    f'<circle cx="{f(px(a))}" cy="{f(py(b))}" r="{4 if s.get("accent") else 3.5}" class="{fill}"/>'
                )
    if legend:
        kx, ky = 0.0, 10.0
        for s in series:
            if kx and kx + 24 + rk.text_width(s["name"]) > width:
                kx, ky = 0.0, ky + 20
            stroke = "c-stroke-accent" if s.get("accent") else "c-stroke-muted"
            dash = ' stroke-dasharray="5 4"' if s.get("dash") else ""
            parts.append(
                f'<path class="{stroke}" stroke-width="2"{dash} d="M{f(kx)} {f(ky)}H{f(kx + 18)}"/>'
            )
            parts.append(
                T(kx + 24, ky + 4, s["name"], "c-label c-hi" if s.get("accent") else "c-label")
            )
            kx += 24 + rk.text_width(s["name"]) + 16
    else:
        ends = [py(s["points"][-1][1]) + 4 for s in series]
        placed = spread_labels(ends)
        for s, y in zip(series, placed):
            lx = s["points"][-1][0]
            parts.append(
                T(
                    px(lx) + 8,
                    y,
                    s.get("end", s["name"]),
                    "c-label c-hi" if s.get("accent") else "c-label",
                )
            )
    parts.append(T(0, mt - 18, y_title, "c-title"))
    parts.append(T(width - mr, height - 6, x_title, "c-title", "end"))
    return rk.svg(uid, width, height, title, desc, "".join(parts))


def interval_chart(uid, width, rows, *, title, desc, axis_title, ticks):
    """Forest plot: one row per window, point estimate with 95% interval, zero line."""
    stacked = width < 520
    label_w = 0 if stacked else max(rk.text_width(r["label"]) for r in rows) + 18
    value_w = 0 if stacked else max(rk.text_width(r["value"], 13) for r in rows) + 16
    row_h = 48 if stacked else 34
    top = 34 if stacked else 18
    plot_l, plot_r = label_w + 8, width - value_w - 8
    lo, hi = ticks[0], ticks[-1]
    x = lambda v: plot_l + (plot_r - plot_l) * (v - lo) / (hi - lo)  # noqa: E731
    bottom = top + row_h * len(rows)
    parts = []
    for t in ticks:
        parts.append(f'<path class="c-grid" d="M{f(x(t))} {top - 8}V{f(bottom)}"/>')
        parts.append(
            T(x(t), bottom + 16, plain(t, 0) if t == int(t) else plain(t, 1), "c-tick", "middle")
        )
    parts.append(f'<path class="c-floor" d="M{f(x(0))} {top - 14}V{f(bottom)}"/>')
    parts.append(T(x(0) + 6, top - 18 if stacked else top - 4, "Sharpe 0", "c-tick"))
    for i, r in enumerate(rows):
        cy = top + i * row_h + (row_h - 14 if stacked else row_h / 2)
        if stacked:
            parts.append(T(0, cy - 14, r["label"], "c-label"))
            parts.append(T(width, cy - 14, r["value"], "c-value", "end"))
        else:
            parts.append(T(label_w - 4, cy + 4.5, r["label"], "c-label", "end"))
            parts.append(T(plot_r + 16, cy + 4.5, r["value"], "c-value"))
        parts.append(
            f'<path class="c-whisker" d="M{f(x(r["lo"]))} {f(cy)}H{f(x(r["hi"]))}'
            f'M{f(x(r["lo"]))} {f(cy - 5)}V{f(cy + 5)}M{f(x(r["hi"]))} {f(cy - 5)}V{f(cy + 5)}"/>'
        )
        parts.append(f'<circle cx="{f(x(r["mid"]))}" cy="{f(cy)}" r="5" class="c-accent"/>')
    parts.append(T(plot_r if not stacked else width, bottom + 36, axis_title, "c-title", "end"))
    return rk.svg(uid, width, bottom + 44, title, desc, "".join(parts))


def plot_step(fill, target=300):
    """Plot every n-th trading day (plus the last) so each curve has about `target` points."""
    return max(1, len(fill["dates"]) // target)


def fill_chart(d, uid, width, height, legend):
    fill = d["fill"]
    step = plot_step(fill)
    idx = list(range(0, len(fill["dates"]), step))
    if idx[-1] != len(fill["dates"]) - 1:
        idx.append(len(fill["dates"]) - 1)
    xs = [year_frac(fill["dates"][i]) for i in idx]
    pct = lambda v: (v / fill["next_bar_open"][0] - 1) * 100  # noqa: E731
    nxt = [(xs[k], pct(fill["next_bar_open"][i])) for k, i in enumerate(idx)]
    same = [(xs[k], pct(fill["same_bar_open"][i])) for k, i in enumerate(idx)]
    ys = [pct(v) for v in fill["next_bar_open"] + fill["same_bar_open"]] + [0.0]
    y_ticks = rk.nice_ticks(min(ys), max(ys), 5)
    first, last = int(math.ceil(xs[0])), int(math.floor(xs[-1]))
    every = 2 if width < 700 else 1
    x_ticks = [xs[0]] + [y for y in range(first, last + 1) if (y - first) % every == 0] + [xs[-1]]
    x_ticks = sorted(set(x_ticks))
    gap = d["gap"]
    series = [
        {
            "name": "Same-bar (lookahead)",
            "end": f"Same-bar {signed(fill['same_bar_open_return_pct'], 1, '%')}",
            "points": same,
            "accent": False,
            "dash": True,
        },
        {
            "name": "Next-open (default)",
            "end": f"Next-open {signed(fill['next_bar_open_return_pct'], 1, '%')}",
            "points": nxt,
            "accent": True,
        },
    ]
    title = f"Same-bar fills change SMA crossover's total return by {signed(gap, 1)} pp"
    desc = (
        f"Total return of the SMA crossover from {fill['dates'][0]} to {fill['dates'][-1]} on one synthetic asset. "
        f"Next-bar-open fills end at {signed(fill['next_bar_open_return_pct'], 1, '%')} total return; "
        f"same-bar fills, which trade before the signal is known, end at "
        f"{signed(fill['same_bar_open_return_pct'], 1, '%')}."
    )
    return xy_chart(
        uid,
        width,
        height,
        series,
        title=title,
        desc=desc,
        x_ticks=x_ticks,
        y_ticks=y_ticks,
        fmt_x=lambda v: str(int(round(v))) if abs(v - round(v)) < 1e-6 else "",
        fmt_y=lambda v: signed(v, 0, "%"),
        x_title="Date",
        y_title="Total return since start (%)",
        zero=True,
        legend=legend,
    )


def sharpe_chart(d, uid, width):
    rows = []
    for w in d["windows"]:
        start, end = w["test_start"][:7], w["test_end"][:7]
        rows.append(
            {
                "label": f"{start} to {end}",
                "lo": w["ci_low"],
                "mid": w["oos_sharpe"],
                "hi": w["ci_high"],
                "value": f"{plain(w['oos_sharpe'])}",
            }
        )
    lo = min(r["lo"] for r in rows)
    hi = max(r["hi"] for r in rows)
    ticks = rk.nice_ticks(math.floor(lo), math.ceil(hi), 6)
    desc = (
        "; ".join(
            f"{r['label']}: Sharpe {plain(r['mid'])}, 95% interval {plain(r['lo'])} to {plain(r['hi'])}"
            for r in rows
        )
        + "."
    )
    return interval_chart(
        uid,
        width,
        rows,
        title="Out-of-sample Sharpe ratio per one-year test window, with 95% bootstrap intervals",
        desc=desc,
        axis_title="Annualized Sharpe ratio, test window only",
        ticks=ticks,
    )


def runtime_chart(bench, uid, width, height, legend):
    rows = bench["rows"]
    series = []
    for engine, name, accent in (
        ("event", "Event-driven", False),
        ("vectorized", "Vectorized", True),
    ):
        series.append(
            {
                "name": name,
                "end": name,
                "accent": accent,
                "points": [(r["assets"], r[f"{engine}_median_s"]) for r in rows],
                "whiskers": [
                    (r["assets"], min(r[f"{engine}_runs_s"]), max(r[f"{engine}_runs_s"]))
                    for r in rows
                ],
            }
        )
    ys = [w for s in series for _, lo, hi in s["whiskers"] for w in (lo, hi)]
    steps = [m * 10**e for e in range(-3, 4) for m in (1, 2, 5)]
    y_ticks = rk.log_ticks(
        max(v for v in steps if v <= min(ys)), min(v for v in steps if v >= max(ys))
    )
    x_ticks = rk.log_ticks(1, max(r["assets"] for r in rows))
    desc = (
        "; ".join(
            f"{assets_text(r['assets'])}: event-driven {secs(r['event_median_s'])}, vectorized "
            f"{secs(r['vectorized_median_s'])}"
            for r in rows
        )
        + f"; median of {bench['runs']} runs."
    )
    return xy_chart(
        uid,
        width,
        height,
        series,
        title="Seconds per 10-year backtest against number of assets, log-log",
        desc=desc,
        x_ticks=x_ticks,
        y_ticks=y_ticks,
        fmt_x=lambda v: f"{v:g}",
        fmt_y=lambda v: f"{v:g}",
        x_title="Assets (log scale)",
        y_title="Seconds per backtest (log scale)",
        log_x=True,
        log_y=True,
        legend=legend,
        markers=True,
    )


def cost_chart(results, d, uid, width, height, legend):
    series = []
    for key, accent in ((SECONDARY, False), (PRIMARY, True)):
        s = results["strategies"][key]
        c = s["cost_sensitivity"]
        series.append(
            {
                "name": s["label"],
                "end": s["label"],
                "accent": accent,
                "points": list(zip(c["cost_bps"], c["sharpe"])),
            }
        )
    xs = d["cost"]["cost_bps"]
    ys = [p[1] for s in series for p in s["points"]] + [0]
    y_ticks = rk.nice_ticks(min(ys), max(ys), 5)
    x_ticks = rk.nice_ticks(min(xs), max(xs), 6)
    desc = " ".join(
        f"{s['name']}: " + ", ".join(f"{x:g} bps Sharpe {plain(y)}" for x, y in s["points"]) + "."
        for s in series
    )
    default = d["cost"]["default_bps"]
    return xy_chart(
        uid,
        width,
        height,
        series,
        title="Full-sample Sharpe ratio against cost per trade",
        desc=desc,
        x_ticks=x_ticks,
        y_ticks=y_ticks,
        fmt_x=lambda v: f"{v:g}",
        fmt_y=lambda v: plain(v, 1),
        x_title="Commission and slippage, each (bps per trade)",
        y_title="Annualized Sharpe ratio",
        zero=True,
        vlines=[(default, "default")],
        legend=legend,
        markers=True,
    )


def engines_svg(uid, width):
    """Event-driven and vectorized pipelines, side by side in lanes; stacked boxes on phones."""
    lanes = [
        (
            "Event-driven engine",
            False,
            [
                ("Bar T", "one asset, one day"),
                ("Strategy", "emits a signal"),
                ("Portfolio + risk", "size, limits → order"),
                ("Execution", "fill at T+1 open"),
                ("Equity", "marked each bar"),
            ],
        ),
        (
            "Vectorized engine",
            True,
            [
                ("Price matrix", "bars × assets"),
                ("Signal function", "target at each close"),
                ("Numba loop", "trade at T+1 price"),
                ("Equity", "+ analytics"),
            ],
        ),
    ]
    parts = []
    if width >= 800:
        n = max(len(steps) for _, _, steps in lanes)
        gap = 28
        bw = (width - gap * (n - 1)) / n
        y = 0
        for name, accent, steps in lanes:
            parts.append(T(0, y + 14, name, "c-value c-hi" if accent else "c-value"))
            by = y + 28
            for i, (head, sub) in enumerate(steps):
                bx = i * (bw + gap)
                parts.append(
                    f'<rect class="c-node" x="{f(bx)}" y="{f(by)}" width="{f(bw)}" height="58" rx="10"/>'
                )
                parts.append(T(bx + 12, by + 24, head, "c-label"))
                parts.append(T(bx + 12, by + 44, sub, "c-tick"))
                if i:
                    ax = bx - gap
                    parts.append(
                        f'<path class="{"c-line-accent" if accent else "c-line"}" '
                        f'd="M{f(ax + 4)} {f(by + 29)}H{f(bx - 6)}M{f(bx - 11)} {f(by + 24)}L{f(bx - 6)} '
                        f'{f(by + 29)}L{f(bx - 11)} {f(by + 34)}"/>'
                    )
            y = by + 58 + 36
        height = y - 36
    else:
        y = 0
        for name, accent, steps in lanes:
            parts.append(T(0, y + 14, name, "c-value c-hi" if accent else "c-value"))
            by = y + 26
            for i, (head, sub) in enumerate(steps):
                parts.append(
                    f'<rect class="c-node" x="0" y="{f(by)}" width="{f(width)}" height="40" rx="10"/>'
                )
                parts.append(T(12, by + 25, head, "c-label"))
                parts.append(T(width - 12, by + 25, sub, "c-tick", "end"))
                if i < len(steps) - 1:
                    parts.append(
                        f'<path class="{"c-line-accent" if accent else "c-line"}" '
                        f'd="M24 {f(by + 41)}V{f(by + 53)}M20 {f(by + 49)}L24 {f(by + 53)}L28 {f(by + 49)}"/>'
                    )
                by += 54
            y = by + 20
        height = y - 20
    desc = (
        "Event-driven: each bar goes to the strategy, which emits a signal; the portfolio and risk manager turn "
        "it into an order, the execution handler fills it at the next bar's open, and equity is marked each "
        "bar. Vectorized: a signal function computes the target position at every close for the whole price "
        "matrix, and one compiled loop trades it at the next bar's price."
    )
    return rk.svg(
        uid, width, height, "How each engine turns prices into trades", desc, "".join(parts)
    )


def hero_chart(d, uid, width):
    big = d["big"]
    rows = [
        {"label": "Event-driven", "value": big["event_median_s"], "accent": False},
        {"label": "Vectorized", "value": big["vectorized_median_s"], "accent": True},
    ]
    return rk.bar_chart(
        rows,
        uid=uid,
        width=width,
        title=f"Seconds per backtest, {assets_text(big['assets'])} × 10 years",
        desc=f"Event-driven {secs(big['event_median_s'])}, vectorized "
        f"{secs(big['vectorized_median_s'])}; median of {d['runs']} runs.",
        axis_title="Seconds per backtest, bars from zero",
        fmt=secs,
    )


# ---------------------------------------------------------------- README


def readme_block(results, bench, tests, facts):
    d = derive(results, bench, tests)
    big = d["big"]
    return "\n".join(
        [
            f"- **Two engines that agree.** A parity test runs one strategy through both and requires equal trade "
            f"counts and total returns within {pct_pp(facts['parity_tol'])}.",
            f"- **Vectorized engine {ratio(d['speedup'])} faster** on {big['assets']} assets × 10 years: "
            f"{secs(big['vectorized_median_s'])} vs {secs(big['event_median_s'])} event-driven (median of {d['runs']} "
            f"runs, synthetic data, one strategy, {bench['cpu']}).",
            "- **No lookahead by default.** A signal on one day fills at the next day's open; on the seeded demo, "
            f"same-bar fills would change the SMA crossover's return by {signed(d['gap'], 1)} pp.",
            f"- **Overfitting checks.** Walk-forward testing, bootstrap Sharpe intervals and a deflated Sharpe ratio "
            f"that discounts for {d['trials']} trials. On synthetic data the deflated Sharpe is {d['dsr']:.2f}: no "
            f"edge is certified.",
            f"- **Default costs:** {facts['commission'] * 100:g}% commission + {facts['slippage'] * 100:g}% slippage "
            "per trade in the vectorized engine.",
        ]
    )


def update_readme(path, results, bench, tests, facts):
    text = pathlib.Path(path).read_text(encoding="utf-8")
    if README_START not in text or README_END not in text:
        sys.exit(f"{path} has no {README_START} / {README_END} markers")
    head, rest = text.split(README_START, 1)
    _, tail = rest.split(README_END, 1)
    pathlib.Path(path).write_text(
        f"{head}{README_START}\n{readme_block(results, bench, tests, facts)}\n{README_END}{tail}",
        encoding="utf-8",
    )


# ---------------------------------------------------------------- page


def evidence_rows(d, facts, code_sha, res_sha, bench_sha, tests_sha):
    def blob(rel, sha, start=None, end=None):
        anchor = f"#L{start}" + (f"-L{end}" if end and end != start else "") if start else ""
        return f"{REPO_URL}/blob/{sha}/{rel}{anchor}"

    big = d["big"]
    par = (
        line_of("tests/test_engine_parity.py", "assert event_round_trips >= 3"),
        line_of("tests/test_engine_parity.py", "abs(event_return - vector_return)"),
    )
    look = line_of("tests/test_no_lookahead.py", "def test_signal_on_bar_2_fills_at_bar_3_open")
    vec = (
        line_of("src/quantsim/backtesting/vectorized.py", "commission_rate: float ="),
        line_of("src/quantsim/backtesting/vectorized.py", "slippage_pct: float ="),
    )
    eng = (
        line_of("src/quantsim/backtesting/engine.py", "slippage_model=slippage_model or"),
        line_of("src/quantsim/backtesting/engine.py", "commission_model=commission_model or"),
    )
    raw = line_of("src/quantsim/data/ingestion.py", "store raw (unadjusted) bars")
    mix = line_of("src/quantsim/data/data_handler.py", 'open=float(row.get("open"')
    results_rel = "results/demo/results.json"
    t = d["tests"]
    return [
        (
            "E1",
            f"{assets_text(big['assets'])} × 10 years: vectorized {secs(big['vectorized_median_s'])}, "
            f"event-driven {secs(big['event_median_s'])}, median of {d['runs']} runs",
            "benchmarks/results.json",
            blob("benchmarks/results.json", bench_sha),
            "Verified (re-run)",
        ),
        (
            "E2",
            f"Vectorized engine {ratio(d['speedup'])} faster at {big['assets']} assets",
            "ratio of the E1 medians",
            blob("benchmarks/results.json", bench_sha),
            "Derived",
        ),
        (
            "E3",
            f"Engines agree on trade count and on total return within {pct_pp(facts['parity_tol'])}",
            f"tests/test_engine_parity.py:{par[0]}–{par[1]}",
            blob("tests/test_engine_parity.py", code_sha, *par),
            "Verified (test)",
        ),
        (
            "E4",
            "A signal on bar T fills at bar T+1's open",
            f"tests/test_no_lookahead.py:{look}",
            blob("tests/test_no_lookahead.py", code_sha, look),
            "Verified (test)",
        ),
        (
            "E5",
            f"{t['passed']} tests passed, {t['failed']} failed; {t['deselected']} live or browser tests deselected",
            "results/tests.json",
            blob("results/tests.json", tests_sha),
            "Verified (run)",
        ),
        (
            "E6",
            f"Vectorized default costs {facts['commission'] * 100:g}% commission + {facts['slippage'] * 100:g}% "
            "slippage per trade",
            f"vectorized.py:{vec[0]}–{vec[1]}",
            blob("src/quantsim/backtesting/vectorized.py", code_sha, *vec),
            "Verified (code)",
        ),
        (
            "E7",
            f"Same-bar fills change SMA total return by {signed(d['gap'], 1)} pp "
            f"({signed(d['fill']['same_bar_open_return_pct'], 1, '%')} vs "
            f"{signed(d['fill']['next_bar_open_return_pct'], 1, '%')})",
            results_rel,
            blob(results_rel, res_sha),
            "Verified (run)",
        ),
        (
            "E8",
            f"Deflated Sharpe {d['dsr']:.2f} after {d['trials']} trials; out-of-sample return "
            f"{signed(d['oos_return'], 1, '%')}",
            results_rel,
            blob(results_rel, res_sha),
            "Verified (run)",
        ),
        (
            "E9",
            f"{d['zero_in_ci']} of {len(d['windows'])} window Sharpe intervals include zero",
            results_rel,
            blob(results_rel, res_sha),
            "Derived",
        ),
        (
            "E10",
            "Cost sensitivity: Sharpe at each commission + slippage level",
            results_rel,
            blob(results_rel, res_sha),
            "Verified (run)",
        ),
        (
            "E11",
            "Event engine defaults: volume-proportional slippage, no commission",
            f"engine.py:{eng[0]}–{eng[1]}",
            blob("src/quantsim/backtesting/engine.py", code_sha, *eng),
            "Verified (code)",
        ),
        (
            "E12",
            "Bars stored unadjusted; bar events mix raw open with adjusted close",
            f"ingestion.py:{raw}; data_handler.py:{mix}",
            blob("src/quantsim/data/data_handler.py", code_sha, mix),
            "Verified (code)",
        ),
    ]


def build_page(results, bench, tests, facts, shas):
    d = derive(results, bench, tests)
    code_sha = results["meta"]["sha"]
    ev = {
        row[0]: row
        for row in evidence_rows(d, facts, code_sha, shas["results"], shas["bench"], shas["tests"])
    }
    R = rk.ref
    big, small = d["big"], d["small"]
    data = results["meta"]["data"]
    fill = d["fill"]
    sma = results["strategies"][PRIMARY]
    mom = results["strategies"][SECONDARY]
    runs = f"median of {d['runs']} runs"
    cpu = bench["cpu"]
    load_b = " / ".join(f"{v:.2f}" for v in bench.get("loadavg_before", []))
    load_a = " / ".join(f"{v:.2f}" for v in bench.get("loadavg_after", []))
    costs_vec = f"{facts['commission'] * 100:g}% commission + {facts['slippage'] * 100:g}% slippage"
    tol = pct_pp(facts["parity_tol"])

    hero = rk.hero(
        eyebrow="Quant research · Python · Backtesting",
        title="quantsim",
        subtitle=Raw(
            "A backtester for daily stock prices with two engines that agree, built so that trading on "
            "tomorrow's prices and overfitting to the past are hard to do by accident."
        ),
        value=ratio(d["speedup"]),
        label=Raw(
            f"speed of the vectorized engine over the event-driven one, {big['assets']} assets × 10 years"
            f"{R('E2')}"
        ),
        support=Raw(
            f"{secs(big['vectorized_median_s'])} vs {secs(big['event_median_s'])} per backtest, {runs}"
            f"{R('E1')}. Both engines agree within {tol} on the same strategy{R('E3')}."
        ),
        context=f"Synthetic data (seeded random prices), one strategy (SMA {bench['strategy'].split()[-1]}), "
        f"{cpu}.",
        actions=[("Source", REPO_URL, True), ("Reproduce", "#reproduce", False)],
        figure=rk.tag(
            "p", f"Seconds per backtest, {big['assets']} assets, {runs}", class_="card-label"
        )
        + rk.variants(
            hero_chart(d, "hero-d", 398), hero_chart(d, "hero-t", 640), hero_chart(d, "hero-p", 280)
        ),
    )

    nav = rk.report_nav(
        [
            ("findings", "Findings"),
            ("lookahead", "Lookahead"),
            ("overfitting", "Overfitting"),
            ("costs", "Costs"),
            ("engines", "Engines"),
            ("speed", "Speed"),
            ("methodology", "Method"),
            ("limitations", "Limits"),
            ("reproduce", "Reproduce"),
            ("evidence", "Evidence"),
        ]
    )

    # Findings bento
    t = d["tests"]
    findings = rk.bento(
        [
            (
                8,
                1,
                rk.join(
                    [
                        rk.tag(
                            "h3",
                            "Filling on the signal's own bar changes the result",
                            class_="card-heading",
                        ),
                        Raw('<div class="mini">'),
                        rk.variants(
                            fill_chart(d, "mini-d", 640, 220, False),
                            fill_chart(d, "mini-t", 600, 220, False),
                            fill_chart(d, "mini-p", 300, 220, True),
                        ),
                        Raw("</div>"),
                        rk.tag(
                            "p",
                            Raw(
                                f"SMA crossover on one synthetic asset: {signed(fill['next_bar_open_return_pct'], 1, '%')} "
                                f"with next-open fills, {signed(fill['same_bar_open_return_pct'], 1, '%')} if it could "
                                f"trade at the open of the bar that produced the signal.{R('E7')} "
                                f'<a href="#lookahead">Fill timing</a>'
                            ),
                        ),
                    ]
                ),
            ),
            (
                4,
                2,
                rk.join(
                    [
                        rk.tag("h3", "Two engines, one answer", class_="card-heading"),
                        rk.metric(
                            "Largest allowed return gap", f"{facts['parity_tol'] * 100:g}", "pp"
                        ),
                        rk.tag(
                            "p",
                            Raw(
                                f"The parity test runs an SMA {facts['parity_fast']}/{facts['parity_slow']} crossover "
                                f"over {facts['parity_bars']} bars through both engines and requires the same number of "
                                f"round trips and total returns within {tol}.{R('E3')}"
                            ),
                        ),
                    ]
                ),
            ),
            (
                4,
                3,
                rk.join(
                    [
                        rk.tag("h3", "Vectorized time barely moves", class_="card-heading"),
                        rk.comparison(
                            [
                                (
                                    f"Event, {small['assets']} → {big['assets']} assets",
                                    d["event_growth"],
                                    ratio(d["event_growth"]),
                                    False,
                                ),
                                (
                                    f"Vectorized, {small['assets']} → {big['assets']} assets",
                                    d["vec_growth"],
                                    ratio(d["vec_growth"]),
                                    True,
                                ),
                            ],
                            max(d["event_growth"], d["vec_growth"]),
                        ),
                        rk.tag(
                            "p",
                            Raw(
                                f"Growth in run time from {assets_text(small['assets'])} to {big['assets']}.{R('E1')} "
                                f'<a href="#speed">Scaling</a>'
                            ),
                        ),
                    ]
                ),
            ),
            (
                4,
                4,
                rk.join(
                    [
                        rk.tag("h3", "Tested", class_="card-heading"),
                        rk.metric("Tests passed", f"{t['passed']:,}"),
                        rk.tag(
                            "p",
                            Raw(
                                f"{t['failed']} failed, {t['deselected']} live-trading and browser tests deselected; "
                                f"includes next-open fill, parity and timestamp tests.{R('E5')}{R('E4')}"
                            ),
                        ),
                    ]
                ),
            ),
            (
                4,
                5,
                rk.join(
                    [
                        rk.tag("h3", "Costs on by default", class_="card-heading"),
                        rk.metric(
                            "Per trade, vectorized engine",
                            f"{(facts['commission'] + facts['slippage']) * 100:g}",
                            "%",
                        ),
                        rk.tag(
                            "p",
                            Raw(
                                f"{costs_vec}.{R('E6')} The event engine defaults to volume-based slippage and no "
                                f"commission.{R('E11')}"
                            ),
                        ),
                    ]
                ),
            ),
        ]
    )
    findings_sec = rk.section(
        "findings", "What the measurements show", findings, eyebrow="Findings"
    )

    # Lookahead
    gap_word = "raises" if d["gap"] > 0 else "lowers"
    other_gap = (
        mom["fill_timing"]["same_bar_open_return_pct"]
        - mom["fill_timing"]["next_bar_open_return_pct"]
    )
    fill_fig = rk.figure(
        "fig-fill",
        f"Trading at the signal bar's own open {gap_word} SMA's return by {abs(d['gap']):.1f} pp",
        rk.variants(
            fill_chart(d, "fill-d", 1000, 400, False),
            fill_chart(d, "fill-t", 640, 340, False),
            fill_chart(d, "fill-p", 300, 300, True),
        ),
        rk.tag(
            "p",
            Raw(
                f"Event engine, SMA crossover ({esc(params_text(sma['params']))}), one synthetic asset "
                f"({esc(data['symbol'])}, seed {esc(data['seed'])}), {nowrap(fill['dates'][0])} to "
                f"{nowrap(fill['dates'][-1])}, "
                f"{fill['trade_count']} fills with next-open timing; single run per mode. Costs: {esc(fill['costs'])}, "
                f"identical in both runs. The synthetic bars set each open to the previous close, so a same-bar fill "
                f"trades at the price before the signal day's move. Every {plot_step(fill)}th trading day "
                f"plotted, plus the last.{R('E7')}"
            ),
        ),
        question="How much would results change if an order could fill on the bar that generated it?",
        table=rk.data_table(
            "Total return by fill timing (event engine, synthetic data)",
            ["Strategy", "Next-open fills", "Same-bar fills", "Difference"],
            [
                [
                    s["label"],
                    signed(s["fill_timing"]["next_bar_open_return_pct"], 1, "%"),
                    signed(s["fill_timing"]["same_bar_open_return_pct"], 1, "%"),
                    signed(
                        s["fill_timing"]["same_bar_open_return_pct"]
                        - s["fill_timing"]["next_bar_open_return_pct"],
                        1,
                        " pp",
                    ),
                ]
                for s in (sma, mom)
            ],
            numeric={1, 2, 3},
        ),
        data_href=ev["E7"][3],
    )
    look_body = rk.join(
        [
            Raw('<div class="prose">'),
            rk.tag(
                "p",
                Raw(
                    "<em>Lookahead</em> means a backtest uses a price it could not have known when it decided to trade. "
                    "In quantsim a strategy sees bar T's close, so its order waits for bar T+1 and fills at that bar's open."
                    f"{R('E4')} A flag, <code>SAME_BAR_OPEN_LOOKAHEAD</code>, reproduces the old bug so its effect can be "
                    "measured."
                ),
            ),
            Raw("</div>"),
            fill_fig,
            Raw('<div class="prose" style="margin-top:32px">'),
            rk.tag(
                "p",
                Raw(
                    f"The gap has no fixed sign: same-bar fills move the momentum strategy by {signed(other_gap, 1)} pp on "
                    f"the same series. A test over 50 seeded random walks checks that the next-open default gives a "
                    f"close-peeking strategy no edge."
                ),
            ),
            Raw("</div>"),
        ]
    )
    look_sec = rk.section(
        "lookahead",
        "Orders fill at the next bar's open, so signals cannot see their own fill",
        look_body,
        eyebrow="Lookahead",
    )

    # Overfitting
    windows = d["windows"]
    sharpe_fig = rk.figure(
        "fig-sharpe",
        f"{d['zero_in_ci']} of {len(windows)} one-year Sharpe intervals include zero",
        rk.variants(
            sharpe_chart(d, "sh-d", 1000),
            sharpe_chart(d, "sh-t", 640),
            sharpe_chart(d, "sh-p", 300),
        ),
        rk.tag(
            "p",
            Raw(
                f"SMA crossover, walk-forward: parameters chosen on the {facts['train_years']} preceding years, scored on "
                f"the next {facts['test_months']} months, stepping {facts['step_months']} months. Dots: annualized "
                f"Sharpe of the test window; bars: 95% stationary-bootstrap interval, {facts['window_sims']:,} resamples "
                f"per window. Costs: {esc(sma['walk_forward']['costs'])}. Synthetic data, seed {esc(data['seed'])}."
                f"{R('E9')}"
            ),
        ),
        question="Is any one-year out-of-sample result distinguishable from zero?",
        table=rk.data_table(
            "Walk-forward test windows (SMA crossover)",
            ["Test window", "Chosen parameters", "Sharpe", "95% interval"],
            [
                [
                    w["label"],
                    ", ".join(f"{k} {v}" for k, v in w["params"].items()),
                    plain(w["oos_sharpe"]),
                    f"{plain(w['ci_low'])} to {plain(w['ci_high'])}",
                ]
                for w in windows
            ],
            numeric={2, 3},
        ),
        data_href=ev["E8"][3],
    )
    verdict = "above" if d["dsr"] > 0.95 else "below"
    over_body = rk.join(
        [
            Raw('<div class="prose">'),
            rk.tag(
                "p",
                Raw(
                    f"Trying many settings and keeping the best inflates the Sharpe ratio. The <em>deflated Sharpe ratio</em> "
                    f"is the probability that the chosen setting beats the best of the same number of unskilled ones. Here "
                    f"the walk-forward tried {d['configs']} settings in each of {len(windows)} windows, {d['trials']} trials "
                    f"in all, and the deflated Sharpe is {d['dsr']:.2f}, {verdict} the usual 0.95 bar.{R('E8')}"
                ),
            ),
            rk.tag(
                "p",
                Raw(
                    f"Stitched together, the out-of-sample windows returned {signed(d['oos_return'], 1, '%')} "
                    f"after {esc(sma['walk_forward']['costs'])}; buy-and-hold, with no costs, returned "
                    f"{signed(d['hold_return'], 1, '%')}. The check is designed to withhold certification when the "
                    f"best of many settings could be luck, and here it does."
                ),
            ),
            Raw("</div>"),
            sharpe_fig,
        ]
    )
    over_sec = rk.section(
        "overfitting",
        f"After {d['trials']} trials, the deflated Sharpe certifies no edge",
        over_body,
        eyebrow="Overfitting",
    )

    # Costs
    cost = d["cost"]
    di = d["default_i"]
    default_sharpe = cost["sharpe"][di] if di is not None else None
    cost_fig = rk.figure(
        "fig-costs",
        (
            f"At the default {cost['default_bps']:g} bps each, SMA's full-sample Sharpe is {plain(default_sharpe)}"
            if default_sharpe is not None
            else "Sharpe falls as costs rise"
        ),
        rk.variants(
            cost_chart(results, d, "co-d", 1000, 360, False),
            cost_chart(results, d, "co-t", 640, 320, False),
            cost_chart(results, d, "co-p", 300, 300, True),
        ),
        rk.tag(
            "p",
            Raw(
                f"Vectorized engine, full sample ({esc(data['start'])} to {esc(data['end'])}), the parameters chosen most "
                f"often in walk-forward, {facts['position_size'] * 100:g}% of equity per position. Each level charges that "
                f"many bps of commission and the same again as slippage per trade; the dashed line marks the default. "
                f"Single run per level; parameters were chosen on this data, so these are in-sample.{R('E10')}"
            ),
        ),
        question="How fast does the result decay as trading costs rise?",
        table=rk.data_table(
            "Annualized Sharpe by cost level (bps of commission and of slippage, per trade)",
            ["Strategy"] + [f"{c:g} bps" for c in cost["cost_bps"]],
            [
                [s["label"]] + [plain(v) for v in s["cost_sensitivity"]["sharpe"]]
                for s in (sma, mom)
            ],
            numeric=set(range(1, len(cost["cost_bps"]) + 1)),
        ),
        data_href=ev["E10"][3],
    )
    cost_sec = rk.section(
        "costs",
        "Every published backtest runs with its engine's default costs",
        rk.join(
            [
                Raw('<div class="prose">'),
                rk.tag(
                    "p",
                    Raw(
                        f"The vectorized engine charges {costs_vec} on every trade by default.{R('E6')} Walk-forward results "
                        f"above use those defaults; the fill-timing runs use the event engine's own defaults, volume-proportional "
                        f"slippage (k = {facts['impact_k']:g}) and no commission.{R('E11')}"
                    ),
                ),
                Raw("</div>"),
                cost_fig,
            ]
        ),
        eyebrow="Costs",
    )

    # Engines
    eng_fig = rk.figure(
        "fig-engines",
        "Both engines see bar T and trade at bar T+1",
        rk.variants(engines_svg("en-d", 1000), engines_svg("en-t", 640), engines_svg("en-p", 300)),
        rk.tag(
            "p",
            Raw(
                "Schematic. The event engine passes objects between components one bar at a time; the "
                "vectorized engine computes a whole signal matrix, then runs one compiled loop."
            ),
        ),
        question="Where does each engine decide, and where does it fill?",
    )
    eng_body = rk.join(
        [
            eng_fig,
            Raw('<div class="prose" style="margin-top:32px">'),
            rk.tag(
                "p",
                Raw(
                    f"The parity test feeds both engines one seeded series of {facts['parity_bars']} bars whose opens equal "
                    f"their closes, so a fill at bar T+1's open and at bar T+1's price are the same trade. It requires equal "
                    f"round-trip counts and total returns within {tol}, with costs off in both.{R('E3')} The event engine is "
                    f"the reference for order handling; the vectorized engine is for sweeping parameters."
                ),
            ),
            Raw("</div>"),
        ]
    )
    eng_sec = rk.section(
        "engines",
        f"Two engines trade the same signal one bar later and agree within {tol}",
        eng_body,
        eyebrow="Design",
    )

    # Speed
    speed_fig = rk.figure(
        "fig-speed",
        f"From {small['assets']} to {big['assets']} assets, event time grows {ratio(d['event_growth'])}; "
        f"vectorized {ratio(d['vec_growth'])}",
        rk.variants(
            runtime_chart(bench, "sp-d", 1000, 380, False),
            runtime_chart(bench, "sp-t", 640, 340, False),
            runtime_chart(bench, "sp-p", 300, 300, True),
        ),
        rk.tag(
            "p",
            Raw(
                f"{esc(bench['data'])}; {esc(bench['strategy'])}. Points: {runs} after one warm-up run; whiskers: "
                f"fastest and slowest run. Timed: the vectorized run including its analytics, and the event engine's "
                f"run loop (loading bars from SQLite is not timed). {esc(cpu)}.{R('E1')}"
            ),
        ),
        question="How does run time grow with the number of assets?",
        table=rk.data_table(
            "Seconds per 10-year backtest",
            [
                "Assets",
                "Vectorized median",
                "Vectorized range",
                "Event median",
                "Event range",
                "Event ÷ vectorized",
            ],
            [
                [
                    str(r["assets"]),
                    secs(r["vectorized_median_s"]),
                    f"{min(r['vectorized_runs_s']):.3f}–{max(r['vectorized_runs_s']):.3f}",
                    secs(r["event_median_s"]),
                    f"{min(r['event_runs_s']):.1f}–{max(r['event_runs_s']):.1f}",
                    ratio(r["event_median_s"] / r["vectorized_median_s"]),
                ]
                for r in bench["rows"]
            ],
            numeric={1, 2, 3, 4, 5},
        ),
        data_href=ev["E1"][3],
    )
    speed_body = rk.join(
        [
            speed_fig,
            Raw('<div class="prose" style="margin-top:32px">'),
            rk.tag(
                "p",
                Raw(
                    f"Vectorized time stays near {secs(small['vectorized_median_s'])} at every size, consistent with fixed "
                    f"costs (signal setup and analytics) dominating rather than the per-asset loop; this benchmark does not "
                    f"separate them. Even the slowest vectorized run beat the fastest event run by "
                    f"{ratio(d['speedup_floor'])} at {big['assets']} assets."
                ),
            ),
            rk.tag(
                "p",
                Raw(
                    f"Speed is what makes the robustness checks routine: the walk-forward above runs {d['trials']} vectorized "
                    f"backtests plus one per window for scoring, and the cost sweep one per level."
                ),
            ),
            Raw("</div>"),
        ]
    )
    speed_sec = rk.section(
        "speed",
        f"The vectorized engine runs {big['assets']} assets in "
        f"{secs(big['vectorized_median_s'])}, event-driven in "
        f"{secs(big['event_median_s'])}",
        speed_body,
        eyebrow="Speed",
    )

    # Methodology
    method = rk.methodology(
        [
            ("Hardware", f"{cpu}; single process"),
            ("Operating system", None),
            (
                "Software",
                f"Python {bench['python']}, NumPy {bench['numpy']}, Numba {bench['numba']}, quantsim "
                f"{bench['quantsim']}",
            ),
            (
                "Code measured",
                Raw(
                    f'<a class="src" href="{REPO_URL}/tree/{esc(code_sha)}">{esc(code_sha[:7])}</a>'
                ),
            ),
            ("Benchmark data", f"{bench['data']}; {bench['strategy']}; 1, 10 and 50 assets"),
            (
                "Benchmark runs",
                f"{d['runs']} timed runs per engine and size after one warm-up run (which compiles "
                f"Numba); time.perf_counter; median, whiskers min–max",
            ),
            (
                "Load average",
                f"{load_b} before, {load_a} after (1 / 5 / 15 min); shared machine, benchmark run "
                f"under an exclusive lock"
                if load_b
                else None,
            ),
            ("Benchmark date", bench.get("date", "")[:10]),
            (
                "Report data",
                f"{data['description']}; {data['n_bars']:,} business days, {data['n_assets']} assets "
                f"generated, {data['symbol']} used ({data['start']} to {data['end']})",
            ),
            (
                "Walk-forward",
                f"{facts['train_years']}-year train, {facts['test_months']}-month test, "
                f"{facts['step_months']}-month step; objective Sharpe; {d['configs']} settings per window, "
                f"{d['trials']} trials",
            ),
            (
                "Costs",
                f"Walk-forward and cost sweep: vectorized engine; fill timing: {fill['costs']}",
            ),
            (
                "Bootstrap",
                f"Stationary bootstrap, mean block √n; {facts['window_sims']:,} resamples per window, "
                f"95% percentile interval, seed {results['meta']['seed']}",
            ),
            (
                "Report run",
                f"{results['meta']['command']} at {results['meta']['generated_at'][:10]}",
            ),
            (
                "Tests",
                f"{t['command']}, {t['summary']}, at {t['sha'][:7]} on {t['date'][:10]}, "
                f"Python {t['python']}",
            ),
        ]
    )
    method_sec = rk.section("methodology", "Methodology", method, eyebrow="Conditions")

    limits = rk.limitations(
        [
            (
                "Synthetic data only.",
                f"Every number here comes from seeded random prices with regime shifts "
                f"(benchmark seed {bench['data'].split('seed ')[1].split(',')[0]}, report seed "
                f"{data['seed']}); no market data is committed, so nothing here measures a real edge.",
            ),
            (
                "One machine, one strategy.",
                f"Timings are from one {cpu} and the SMA crossover only; the "
                f"{ratio(d['speedup'])} ratio depends on both.",
            ),
            (
                "Shared machine.",
                f"Load average {load_b} at the start of the benchmark; whiskers show spread within the "
                f"session only.",
            ),
            (
                "Unadjusted bars.",
                Raw(
                    f"Real prices are stored unadjusted with split and dividend factors applied when "
                    f"read; the event engine's bar events carry the raw open but the adjusted close, so "
                    f"on real data a split day can show a false gap. Synthetic data has no adjustments."
                    f"{R('E12')}"
                ),
            ),
            (
                "Survivorship bias.",
                "The default download list is today's large caps, so long real-data backtests "
                "favour survivors.",
            ),
            (
                "In-sample cost sweep.",
                "The cost-sensitivity parameters were chosen on the same data they are scored on.",
            ),
            (
                "Daily bars, equities.",
                "No intraday data and no options backtesting; paper trading by default, live "
                "trading needs two explicit opt-ins.",
            ),
        ]
    )
    limits_sec = rk.section(
        "limitations",
        "Results cover synthetic prices, one strategy and one machine",
        limits,
        eyebrow="Limitations",
    )

    steps_html = rk.reproduce(
        [
            ("Install with every extra (Python 3.11+, uv).", "make setup"),
            (
                "Write the seeded synthetic prices and rebuild the demo results with the default costs.",
                "uv run python tests/fixtures/make_fixture.py\n"
                "uv run quantsim report --prices tests/fixtures/synthetic_prices.csv \\\n"
                "    --strategy sma momentum --out results/demo --card",
            ),
            (
                Raw(
                    f"Time both engines ({d['runs']} runs × 3 sizes; about "
                    f"{sum(sum(r['event_runs_s']) * (1 + 1 / d['runs']) for r in bench['rows']) / 60:.0f} minutes here)."
                ),
                "make bench",
            ),
            (
                Raw(
                    f"Run the offline tests and record the count; expect “{t['passed']} passed, {t['deselected']} deselected”."
                ),
                "uv run python scripts/record_tests.py",
            ),
            (
                "Rebuild this page and the README block from the committed results, then check them.",
                "python3 scripts/build_report.py --readme README.md\n"
                "uv run pytest tests/test_build_report.py -q",
            ),
        ]
    )
    repro_sec = rk.section(
        "reproduce",
        "Reproduce it in five steps",
        steps_html,
        eyebrow="Reproduce",
        intro=Raw(f'Commands run from a clone of <a class="src" href="{REPO_URL}">{REPO}</a>.'),
    )

    ev_table = rk.table(
        "Evidence ledger",
        ["ID", "Claim", "Source", "Class"],
        [
            [
                Raw(f'<span id="evidence-{eid.lower()}">{eid}</span>'),
                claim,
                Raw(f'<a class="src" href="{esc(href)}">{esc(src)}</a>'),
                cls,
            ]
            for eid, claim, src, href, cls in ev.values()
        ],
        wrap_cols={1, 2},
    )
    ev_sec = rk.section(
        "evidence", "Every number traces to a committed file", ev_table, eyebrow="Sources"
    )

    main = rk.join(
        [
            hero,
            nav,
            findings_sec,
            look_sec,
            over_sec,
            cost_sec,
            eng_sec,
            speed_sec,
            method_sec,
            limits_sec,
            repro_sec,
            ev_sec,
        ]
    )
    footer = rk.footer(
        Raw(
            f"Generated by <code>scripts/build_report.py</code> from <code>results/demo/results.json</code>, "
            f"<code>benchmarks/results.json</code> and <code>results/tests.json</code> · code {esc(code_sha[:7])} · "
            f"benchmark {esc(bench.get('date', '')[:10])} on {esc(cpu)}"
        ),
        Raw(f'<a href="{HOME_URL}">Allen Wang</a> · <a href="{REPO_URL}">Source</a>'),
    )
    return rk.page(
        title="quantsim · Allen Wang",
        description=(
            f"quantsim, a Python backtester with two engines that agree within {tol}; the vectorized engine "
            f"runs {big['assets']} assets × 10 years {ratio(d['speedup'])} faster ({runs}, synthetic data, "
            f"{cpu})."
        ),
        canonical=PAGE_URL,
        header=rk.report_header(HOME_URL, "← Allen Wang", [("Source", REPO_URL)]),
        main=main,
        footer=footer,
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--results", default="results/demo/results.json")
    parser.add_argument("--bench", default="benchmarks/results.json")
    parser.add_argument("--tests", default="results/tests.json")
    parser.add_argument("--out", default="docs/index.html")
    parser.add_argument("--readme", help="rewrite the results block between the README markers")
    args = parser.parse_args()

    results, bench, tests = load_json(args.results), load_json(args.bench), load_json(args.tests)
    facts = facts_from_sources()
    rel = lambda p: os.path.relpath(os.path.abspath(p), ROOT)  # noqa: E731
    fallback = results["meta"]["sha"]
    shas = {
        "results": git_last_commit(rel(args.results)) or fallback,
        "bench": git_last_commit(rel(args.bench)) or fallback,
        "tests": git_last_commit(rel(args.tests)) or fallback,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    pathlib.Path(args.out).write_text(
        build_page(results, bench, tests, facts, shas), encoding="utf-8"
    )
    if args.readme:
        update_readme(args.readme, results, bench, tests, facts)


if __name__ == "__main__":
    main()
