"""report-kit v1.1: page shell, HTML components and SVG charts for project reports.

Python 3.10+, standard library only, deterministic output. Text arguments are
escaped; pass already-built HTML as Raw (every helper here returns Raw).
CSS lives in report-kit.css next to this file and is inlined into the page.
"""

from __future__ import annotations

import html
import math
import pathlib

KIT_VERSION = "1.1"
CSS_PATH = pathlib.Path(__file__).with_name("report-kit.css")


class Raw(str):
    """Trusted HTML. esc() leaves it untouched; Raw + text escapes the text."""

    def __add__(self, other):
        return Raw(str.__add__(self, esc(other)))


def esc(value) -> Raw:
    if isinstance(value, Raw):
        return value
    return Raw(html.escape("" if value is None else str(value), quote=True))


def join(parts) -> Raw:
    return Raw("".join(esc(p) for p in parts if p is not None))


def attrs(**kw) -> str:
    """class_="x" -> class="x"; data_op="a" -> data-op="a"; None/False dropped, True bare."""
    out = []
    for key, value in kw.items():
        if value is None or value is False:
            continue
        name = key.rstrip("_").replace("_", "-")
        out.append(f" {name}" if value is True else f' {name}="{html.escape(str(value), quote=True)}"')
    return "".join(out)


def tag(name: str, content="", **kw) -> Raw:
    return Raw(f"<{name}{attrs(**kw)}>{esc(content)}</{name}>")


def fnum(value: float, digits: int = 1) -> str:
    """Fixed-precision number with thousands separators (deterministic)."""
    return f"{value:,.{digits}f}"


NBSP = "\u00a0"


def num(value, unit) -> str:
    """Number and unit joined by a no-break space so the unit never wraps onto its own line.

    value is used as given (format it first, e.g. fnum(x, 1)); returns plain text, escaped on output.
    """
    return f"{value}{NBSP}{unit}"


# ---------------------------------------------------------------- page shell

THEME_BOOTSTRAP = (
    'try{var t=localStorage.getItem("theme");if(t==="light"||t==="dark")'
    'document.documentElement.setAttribute("data-theme",t)}catch(e){}'
    'addEventListener("beforeprint",function(){document.querySelectorAll("details")'
    '.forEach(function(d){d.open=true})});'
)


def page(*, title: str, description: str, canonical: str, header: str, main: str, footer: str,
         og_image: str | None = None, lang: str = "en", site_name: str = "Allen Wang") -> str:
    """Full HTML document with landmarks, skip link, meta/og tags and the inlined kit CSS."""
    if not title or not description:
        raise ValueError("page() needs a title and a description")
    css = CSS_PATH.read_text(encoding="utf-8")
    meta = [
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{esc(title)}</title>",
        f'<meta name="description"{attrs(content=description)}>',
        f'<link rel="canonical"{attrs(href=canonical)}>',
        '<meta name="color-scheme" content="light dark">',
        '<meta property="og:type" content="article">',
        f'<meta property="og:title"{attrs(content=title)}>',
        f'<meta property="og:description"{attrs(content=description)}>',
        f'<meta property="og:url"{attrs(content=canonical)}>',
        f'<meta property="og:site_name"{attrs(content=site_name)}>',
    ]
    if og_image:
        meta.append(f'<meta property="og:image"{attrs(content=og_image)}>')
    return (
        f'<!doctype html>\n<html lang="{esc(lang)}">\n<head>\n' + "\n".join(meta)
        + f"\n<script>{THEME_BOOTSTRAP}</script>\n<style>\n{css}</style>\n</head>\n<body>\n"
        + '<a class="skip" href="#main">Skip to content</a>\n'
        + f'<header class="wrap">{esc(header)}</header>\n'
        + f'<main id="main">\n{esc(main)}\n</main>\n'
        + f'<footer class="wrap">{esc(footer)}</footer>\n</body>\n</html>\n'
    )


# ---------------------------------------------------------------- components

def report_header(back_href: str, back_label: str, links) -> Raw:
    """Identity row: back link left, (label, href) links right."""
    right = join(tag("a", label, href=href) for label, href in links)
    return Raw(f'<nav class="report-header" aria-label="Site">{tag("a", back_label, href=back_href)}'
               f"<span>{right}</span></nav>")


def ref(eid: str, href: str = "#evidence") -> Raw:
    """Evidence reference such as [E2], linking to the ledger row."""
    return tag("a", f"[{eid}]", class_="ref", href=f"{href}-{eid.lower()}" if href == "#evidence" else href,
               aria_label=f"Evidence {eid}")


def hero(*, eyebrow, title, subtitle, value=None, label=None, support=None, context=None, actions=(),
         figure=None) -> Raw:
    """Hero with at most one metric. actions: (label, href, primary).

    value=None drops the metric (compact briefs). figure: Raw HTML for the right column (chart,
    img or picture); top-aligned with the text. Without a figure the text keeps its 7/12 measure.
    """
    acts = join(tag("a", text, href=href, class_="btn btn-primary" if primary else "btn")
                for text, href, primary in actions)
    text = join([
        tag("p", eyebrow, class_="eyebrow"),
        tag("h1", title, class_="hero-title", id="title"),
        tag("p", subtitle, class_="hero-sub"),
        Raw(f'<p class="hero-metric"><span class="value">{esc(value)}</span>'
            f'<span class="label">{esc(label)}</span></p>') if value is not None else None,
        tag("p", support, class_="hero-support") if support else None,
        tag("p", context, class_="hero-context") if context else None,
        tag("div", acts, class_="actions") if actions else None,
    ])
    fig = tag("div", figure, class_="hero-figure") if figure else Raw("")
    return Raw(f'<section class="hero wrap" aria-labelledby="title"><div class="hero-grid"><div>'
               f"{text}</div>{fig}</div></section>")


def report_nav(items) -> Raw:
    """Sticky section nav (Tier A): (id, label) pairs."""
    lis = join(Raw(f'<li>{tag("a", label, href="#" + sid)}</li>') for sid, label in items)
    return Raw(f'<nav class="report-nav" aria-label="Sections"><ul>{lis}</ul></nav>')


def toc(items) -> Raw:
    """Inline contents list (Tier B): report_nav's look, not sticky, wraps instead of scrolling."""
    lis = join(Raw(f'<li>{tag("a", label, href="#" + sid)}</li>') for sid, label in items)
    return Raw(f'<nav class="report-toc" aria-label="Contents"><ul>{lis}</ul></nav>')


def section(sid: str, headline, body, *, eyebrow=None, intro=None) -> Raw:
    head = join([
        tag("p", eyebrow, class_="eyebrow") if eyebrow else None,
        Raw(f'<h2 id="{esc(sid)}">{esc(headline)}<a class="anchor" href="#{esc(sid)}" aria-label="Link to this section">#</a></h2>'),
        tag("p", intro) if intro else None,
    ])
    return Raw(f'<section class="section wrap" aria-labelledby="{esc(sid)}">'
               f'<div class="section-head">{head}</div>{esc(body)}</section>')


def bento(cards) -> Raw:
    """cards: (span, priority, inner_html). Spans per row must be 12, 8+4, 4+8, 6+6 or 4+4+4."""
    out = []
    for span, priority, inner in cards:
        if span not in (4, 6, 8, 12):
            raise ValueError(f"bento span {span} not allowed")
        out.append(f'<div class="card span-{span}" style="--p:{int(priority)}">{esc(inner)}</div>')
    return Raw(f'<div class="bento">{"".join(out)}</div>')


def metric(label, value, unit=None, *, reported=False) -> Raw:
    unit_html = f'<span class="unit">{esc(unit)}</span>' if unit else ""
    flag = '<span class="reported">as reported</span>' if reported else ""
    return Raw(f'<dl class="metric"><dt>{esc(label)}</dt><dd>{esc(value)}{unit_html}{flag}</dd></dl>')


def comparison(rows, maximum: float) -> Raw:
    """HTML bar pair from zero. rows: (label, value, display_text, accent)."""
    out = []
    for label, value, shown, accent in rows:
        pct = 0 if maximum <= 0 else max(0.0, min(100.0, value / maximum * 100))
        out.append(f'<div class="cmp-row"><div class="cmp-head"><span>{esc(label)}</span><b>{esc(shown)}</b></div>'
                   f'<div class="cmp-track" aria-hidden="true"><span class="cmp-fill{" is-accent" if accent else ""}" '
                   f'style="width:{pct:.1f}%"></span></div></div>')
    return Raw(f'<div class="cmp">{"".join(out)}</div>')


def figure(fid: str, takeaway, chart, caption, *, question=None, table=None, data_href=None) -> Raw:
    """Figure panel: takeaway headline above, chart, caption below, optional data table + link."""
    path = str(data_href or "").split("#")[0]
    name = path.split("/blob/", 1)[1].split("/", 1)[-1] if "/blob/" in path else path.rsplit("/", 1)[-1]
    data_link = f' <a class="src" href="{esc(data_href)}">Data: {esc(name)}</a>' if data_href else ""
    return Raw(
        f'<figure class="figure" id="{esc(fid)}" aria-labelledby="{esc(fid)}-t">'
        f'<h3 class="fig-takeaway" id="{esc(fid)}-t">{esc(takeaway)}</h3>'
        + (tag("p", question, class_="fig-question") if question else "")
        + f'<div class="fig-chart">{esc(chart)}</div>'
        f"<figcaption>{esc(caption)}{data_link}</figcaption>{esc(table) if table else ''}</figure>"
    )


def variants(desktop, tablet=None, phone=None) -> Raw:
    """Width-specific SVGs: desktop >= 1068 px, tablet 735-1067 px, phone < 735 px."""
    if tablet is None and phone is None:
        return esc(desktop)
    return Raw(f'<div class="v-d">{esc(desktop)}</div><div class="v-t">{esc(tablet or desktop)}</div>'
               f'<div class="v-p">{esc(phone or tablet)}</div>')


def note(title, body, *, nid=None) -> Raw:
    hid = f"{nid}-h" if nid else None
    return Raw(f'<aside class="note"{attrs(id=nid, aria_labelledby=hid)}>{tag("h3", title, id=hid)}{esc(body)}</aside>')


def code(text: str, label: str | None = None) -> Raw:
    return Raw(f'<pre class="code" tabindex="0"{attrs(aria_label=label)}><code>{esc(text)}</code></pre>')


def table(caption, head, rows, *, numeric=(), row_headers=True, wrap_cols=()) -> Raw:
    """Benchmark table in a scroll wrapper; the caption sits above the wrapper so it never scrolls.

    Cells are text or Raw. The region and table also carry the caption as aria-label
    (no ids, so the same table can appear in several variants()).
    """
    ths = "".join(f'<th scope="col"{attrs(class_="num" if i in numeric else None)}>{esc(h)}</th>' for i, h in enumerate(head))
    body = []
    for row in rows:
        cells = []
        for i, cell in enumerate(row):
            cls = " ".join(c for c in ("num" if i in numeric else "", "wrap-ok" if i in wrap_cols else "") if c) or None
            if i == 0 and row_headers:
                cells.append(f'<th scope="row"{attrs(class_=cls)}>{esc(cell)}</th>')
            else:
                cells.append(f"<td{attrs(class_=cls)}>{esc(cell)}</td>")
        body.append(f"<tr>{''.join(cells)}</tr>")
    return Raw(f'<p class="table-caption">{esc(caption)}</p>'
               f'<div class="table-wrap" tabindex="0" role="region" aria-label="{esc(caption)}">'
               f'<table class="bench" aria-label="{esc(caption)}"><thead><tr>{ths}</tr></thead>'
               f"<tbody>{''.join(body)}</tbody></table></div>")


def data_table(caption, head, rows, *, numeric=()) -> Raw:
    """Chart fallback: the plotted numbers in a disclosure."""
    return Raw(f'<details class="data"><summary>Show data table</summary>{table(caption, head, rows, numeric=numeric)}</details>')


def methodology(items) -> Raw:
    """(term, value) pairs; a missing value renders as 'not recorded'."""
    rows = "".join(f"<dt>{esc(k)}</dt><dd>{esc(v) if v not in (None, '') else 'not recorded'}</dd>" for k, v in items)
    return Raw(f'<dl class="method">{rows}</dl>')


def limitations(items) -> Raw:
    """(limit, effect) pairs."""
    if not items:
        raise ValueError("state 'No material limits identified' with a reason instead of an empty list")
    lis = "".join(f"<li><b>{esc(limit)}</b> {esc(effect)}</li>" for limit, effect in items)
    return Raw(f'<ul class="limits">{lis}</ul>')


def reproduce(steps, expected=None) -> Raw:
    """steps: (description, command or None)."""
    lis = "".join(f"<li>{tag('p', desc)}{code(cmd) if cmd else ''}</li>" for desc, cmd in steps)
    tail = tag("p", expected) if expected else ""
    return Raw(f'<ol class="steps">{lis}</ol>{tail}')


def footer(left, right=None) -> Raw:
    return Raw(f'<div class="report-footer"><span>{esc(left)}</span>{tag("span", right) if right else ""}</div>')


# ---------------------------------------------------------------- SVG charts
# All charts: role="img", <title> + <desc>, classes from report-kit.css so they
# flip with the theme, coordinates rounded to 0.1 px for deterministic output.

CHAR_W = 7.4  # average advance of a 13 px label in the system font


def coord(x: float) -> str:
    s = f"{x:.1f}"
    return s[:-2] if s.endswith(".0") else s


def svg(uid, width, height, title, desc, inner) -> Raw:
    return Raw(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {coord(width)} {coord(height)}" width="{coord(width)}" '
        f'height="{coord(height)}" role="img" aria-labelledby="{uid}-title {uid}-desc">'
        f'<title id="{uid}-title">{esc(title)}</title><desc id="{uid}-desc">{esc(desc)}</desc>{inner}</svg>'
    )


def svg_text(x, y, s, cls="c-label", anchor=None, **kw) -> str:
    return f'<text x="{coord(x)}" y="{coord(y)}" class="{cls}"{attrs(text_anchor=anchor, **kw)}>{esc(s)}</text>'


def text_width(s: str, size: float = 13) -> float:
    return len(s) * CHAR_W * size / 13


def nice_ticks(lo: float, hi: float, count: int = 5) -> list[float]:
    """Linear ticks on 1-2-5 steps covering [lo, hi]."""
    span = max(hi - lo, 1e-12)
    raw = span / max(count - 1, 1)
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 5, 10) if m * mag >= raw)
    start = math.floor(lo / step) * step
    ticks, t = [], start
    while t <= hi + step * 1e-9:
        ticks.append(round(t, 10))
        t += step
    if ticks[-1] < hi:
        ticks.append(round(ticks[-1] + step, 10))
    return ticks


def log_ticks(lo: float, hi: float) -> list[float]:
    """1-2-5 ticks between lo and hi on a log axis."""
    out = []
    for e in range(math.floor(math.log10(lo)), math.ceil(math.log10(hi)) + 1):
        for m in (1, 2, 5):
            v = m * 10 ** e
            if lo - 1e-9 <= v <= hi + 1e-9:
                out.append(v)
    return out


def legend(items, *, width, x=0.0, y=12.0) -> tuple[Raw, float]:
    """Chart key that wraps onto new rows inside `width`. Place it above the plot.

    items: (name, kind, accent[, dashed]) with kind "line" (short stroke) or "dot" (filled circle).
    Returns (svg fragment, height used) so the caller can push the plot down by that height.
    """
    parts, kx, ky = [], x, y
    for item in items:
        name, kind, accent = item[:3]
        dashed = len(item) > 3 and item[3]
        w = 24 + text_width(name) + 16
        if kx > x and kx + w - 16 > x + width:
            kx, ky = x, ky + 20
        if kind == "dot":
            parts.append(f'<circle cx="{coord(kx + 9)}" cy="{coord(ky)}" r="4.5" class="{"c-accent" if accent else "c-muted"}"/>')
        else:
            dash = ' stroke-dasharray="5 4"' if dashed else ""
            parts.append(f'<path class="{"c-stroke-accent" if accent else "c-stroke-muted"}" stroke-width="2"{dash} '
                         f'd="M{coord(kx)} {coord(ky)}H{coord(kx + 18)}"/>')
        parts.append(svg_text(kx + 24, ky + 4, name, "c-label c-hi" if accent else "c-label"))
        kx += w
    return Raw("".join(parts)), ky - y + 20


def bar_chart(rows, *, uid, width, title, desc, axis_title, fmt, domain_max=None) -> Raw:
    """Horizontal bars from zero with direct value labels and optional min-max whiskers.

    rows: dicts with label, value, accent (bool), optional lo/hi (whisker).
    Below 420 px wide, labels sit above their bars.
    """
    stacked = width < 420
    labels_w = 0 if stacked else max(text_width(r["label"]) for r in rows) + 16
    value_w = max(text_width(fmt(r["value"]), 13) for r in rows) + 18
    top, bar_h = 8, 22
    row_h = bar_h + (40 if stacked else 22)
    plot_l, plot_r = labels_w, width - value_w - 8
    vmax = domain_max or max(r.get("hi") or r["value"] for r in rows)
    x = lambda v: plot_l + (plot_r - plot_l) * v / vmax
    parts = []
    for i, r in enumerate(rows):
        y0 = top + i * row_h + (20 if stacked else 0)
        if stacked:
            parts.append(svg_text(0, y0 - 7, r["label"], "c-label c-hi" if r.get("accent") else "c-label"))
        else:
            parts.append(svg_text(labels_w - 12, y0 + bar_h / 2 + 4.5, r["label"], "c-label", "end"))
        parts.append(f'<rect x="{coord(plot_l)}" y="{coord(y0)}" width="{coord(max(x(r["value"]) - plot_l, 1))}" height="{bar_h}" rx="3" '
                     f'class="{"c-accent" if r.get("accent") else "c-muted"}"/>')
        end = x(r["value"])
        if r.get("lo") is not None and r.get("hi") is not None:
            lo, hi, cy = x(r["lo"]), x(r["hi"]), y0 + bar_h / 2
            parts.append(f'<path class="c-whisker" d="M{coord(lo)} {coord(cy)}H{coord(hi)}M{coord(lo)} {coord(cy - 5)}V{coord(cy + 5)}'
                         f'M{coord(hi)} {coord(cy - 5)}V{coord(cy + 5)}"/>')
            end = max(end, hi)
        parts.append(svg_text(end + 8, y0 + bar_h / 2 + 4.5, fmt(r["value"]), "c-value"))
    base_y = top + len(rows) * row_h + (20 if stacked else 0) - (40 if stacked else 22) + bar_h + 6
    if stacked:  # one zero-line tick per bar; a full-height axis would cut through the labels above each bar
        ys = [top + i * row_h + 20 for i in range(len(rows))]
        parts.append(f'<path class="c-axis" d="{"".join(f"M{coord(plot_l)} {coord(y - 3)}V{coord(y + bar_h + 3)}" for y in ys)}"/>')
    else:
        parts.append(f'<path class="c-axis" d="M{coord(plot_l)} {top - 4}V{coord(base_y)}"/>')
    parts.append(svg_text(plot_l, base_y + 18, axis_title, "c-title"))
    return svg(uid, width, base_y + 26, title, desc, "".join(parts))


def ladder_chart(rows, *, uid, width, title, desc, series, points, domain, axis_title, fmt,
                 floor=None, floor_label=None, zero_label=None) -> Raw:
    """Percentile ladder / dot plot on a log x axis.

    rows: dicts with label and values {series_key: {point: value}}.
    series: (key, name, accent). points: ordered keys, e.g. ("p50", "p90", "p99", "p99.9");
    the first is drawn as a filled dot, the last as a ring with its value labelled.
    domain: (lo, hi) in data units. Values <= 0 (below resolution) are drawn in a
    shaded band left of `floor`, which is a labelled dashed line.
    """
    lo, hi = domain
    name_w = max(text_width(name, 12) for _, name, _ in series) + 14
    plot_l, plot_r = name_w, width - 54
    zero_w = 30 if zero_label else 0
    lx = lambda v: plot_l + zero_w + (plot_r - plot_l - zero_w) * (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))
    x = lambda v: plot_l + zero_w / 2 if v <= 0 else lx(max(v, lo))
    head_h, sub_h, gap, top = 20, 22, 14, 70
    row_h = head_h + sub_h * len(series) + gap
    bottom = top + row_h * len(rows) - gap + 4
    parts = []
    if zero_label:
        parts.append(f'<rect class="c-band" x="{coord(plot_l)}" y="{top - 6}" width="{coord(zero_w)}" height="{coord(bottom - top + 6)}"/>')
    for t in [t for t in log_ticks(lo, hi) if not (zero_label and t == lo)]:
        parts.append(f'<path class="c-grid" d="M{coord(lx(t))} {top - 6}V{coord(bottom)}"/>')
        parts.append(svg_text(lx(t), bottom + 16, fmt(t), "c-tick", "middle"))
    if zero_label:
        parts.append(svg_text(plot_l + zero_w / 2, bottom + 16, "0", "c-tick", "middle"))
    if floor:
        fx = lx(floor)
        parts.append(f'<path class="c-floor" d="M{coord(fx)} {top - 14}V{coord(bottom)}"/>')
        if floor_label:
            parts.append(svg_text(fx + 6, top - 12, floor_label, "c-label"))
    # key: every marker drawn is named, top-right, in plotting order
    kx = plot_r
    key = [(points[-1], "ring")] + ([(", ".join(points[1:-1]), "small")] if len(points) > 2 else []) + [(points[0], "dot")]
    marker = {"ring": 'r="4" class="c-stroke-muted" stroke-width="1.5"', "small": 'r="3" class="c-muted"',
              "dot": 'r="4.5" class="c-muted"'}
    for label, shape in key:
        tw = text_width(label, 12)
        parts.append(svg_text(kx, 16, label, "c-title", "end"))
        cx = kx - tw - 12
        parts.append(f'<circle cx="{coord(cx)}" cy="12" {marker[shape]}/>')
        kx = cx - 16
    for i, row in enumerate(rows):
        y0 = top + i * row_h
        parts.append(f'<g{attrs(data_row=row.get("key", row["label"]))}>')
        parts.append(svg_text(0, y0 + 13, row["label"], "c-value"))
        for j, (key_s, name, accent) in enumerate(series):
            vals = row["values"].get(key_s)
            if not vals:
                continue
            cy = y0 + head_h + j * sub_h + sub_h / 2
            parts.append(svg_text(0, cy + 4, name, "c-tick c-hi" if accent else "c-tick"))
            xs = [x(vals[p]) for p in points]
            stroke = "c-stroke-accent" if accent else "c-stroke-muted"
            fill = "c-accent" if accent else "c-muted"
            parts.append(f'<path class="{stroke}" stroke-width="2" d="M{coord(min(xs))} {coord(cy)}H{coord(max(xs))}"/>')
            for k, p in enumerate(points[1:-1], start=1):
                parts.append(f'<circle cx="{coord(xs[k])}" cy="{coord(cy)}" r="3" class="{fill}"/>')
            parts.append(f'<circle cx="{coord(xs[0])}" cy="{coord(cy)}" r="5" class="{fill}"/>')
            parts.append(f'<circle cx="{coord(xs[-1])}" cy="{coord(cy)}" r="4.5" class="{stroke}" stroke-width="2" style="fill:var(--surface)"/>')
            parts.append(svg_text(xs[-1] + 9, cy + 4, fmt(vals[points[-1]]), "c-tick"))
        parts.append("</g>")
    if zero_label:
        parts.append(svg_text(0, 36, zero_label, "c-title"))
    parts.append(svg_text(plot_r, bottom + 36, axis_title, "c-title", "end"))
    return svg(uid, width, bottom + 44, title, desc, "".join(parts))


def line_chart(series, *, uid, width, height, title, desc, x_title, y_title, fmt_x=str, fmt_y=str,
               log_x=False, log_y=False, y_zero=True) -> Raw:
    """Line chart; series: dicts with name, points [(x, y)], accent. Direct end labels, no legend."""
    xs = [p[0] for s in series for p in s["points"]]
    ys = [p[1] for s in series for p in s["points"]]
    name_w = max(text_width(s["name"]) for s in series) + 12
    ml, mr, mt, mb = 48, name_w, 28, 44
    if log_x:
        xt = log_ticks(min(xs), max(xs))
        fx = lambda v: math.log10(v)
    else:
        xt = nice_ticks(min(xs), max(xs))
        fx = lambda v: v
    if log_y:
        yt = log_ticks(min(ys), max(ys))
        fy = lambda v: math.log10(v)
    else:
        yt = nice_ticks(0 if y_zero else min(ys), max(ys))
        fy = lambda v: v
    x0, x1, y0, y1 = fx(xt[0]), fx(xt[-1]), fy(yt[0]), fy(yt[-1])
    px = lambda v: ml + (width - ml - mr) * (fx(v) - x0) / ((x1 - x0) or 1)
    py = lambda v: height - mb - (height - mt - mb) * (fy(v) - y0) / ((y1 - y0) or 1)
    parts = []
    for t in yt:
        parts.append(f'<path class="c-grid" d="M{ml} {coord(py(t))}H{coord(width - mr)}"/>')
        parts.append(svg_text(ml - 8, py(t) + 4, fmt_y(t), "c-tick", "end"))
    for t in xt:
        parts.append(svg_text(px(t), height - mb + 18, fmt_x(t), "c-tick", "middle"))
    parts.append(f'<path class="c-axis" d="M{ml} {coord(height - mb)}H{coord(width - mr)}"/>')
    for s in sorted(series, key=lambda s: bool(s.get("accent"))):
        d = "".join(("M" if k == 0 else "L") + f"{coord(px(a))} {coord(py(b))}" for k, (a, b) in enumerate(s["points"]))
        parts.append(f'<path class="{"c-line-accent" if s.get("accent") else "c-stroke-muted"}" stroke-width="1.5" d="{d}"/>')
        lx, ly = s["points"][-1]
        parts.append(svg_text(px(lx) + 6, py(ly) + 4, s["name"], "c-label c-hi" if s.get("accent") else "c-label"))
    parts.append(svg_text(ml, 14, y_title, "c-title"))
    parts.append(svg_text(width - mr, height - 8, x_title, "c-title", "end"))
    return svg(uid, width, height, title, desc, "".join(parts))
