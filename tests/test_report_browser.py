"""
The report page must render in a real browser at phone and desktop widths in
light and dark mode, without console errors or horizontal scrolling.

Needs Playwright's Chromium (python -m playwright install chromium). The page
is self-contained (report-kit is inlined), so no network is needed. Set
REPORT_SCREENSHOTS=<dir> to keep the screenshots; docs/img/report.png in the
README is one of them.
"""

import os
from pathlib import Path

import pytest

from tests.test_report import built  # noqa: F401  (module-scoped report fixture)

playwright = pytest.importorskip("playwright.sync_api")

pytestmark = pytest.mark.browser


@pytest.fixture(scope="module")
def page_path(built, tmp_path_factory):  # noqa: F811
    from quantsim.reporting.report_page import render_html

    results, _ = built
    path = tmp_path_factory.mktemp("page") / "index.html"
    path.write_text(render_html(results))
    return path


@pytest.mark.parametrize("width", [375, 1280])
@pytest.mark.parametrize("scheme", ["light", "dark"])
def test_report_renders_without_errors_or_overflow(page_path, width, scheme, tmp_path):
    errors = []
    with playwright.sync_playwright() as p:
        # --single-process: in a sandboxed macOS shell Chromium's GPU process cannot
        # start and screenshots hang; one process renders the same page.
        browser = p.chromium.launch(args=["--disable-gpu", "--single-process"])
        page = browser.new_page(viewport={"width": width, "height": 900}, color_scheme=scheme)
        page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)
        page.on("pageerror", lambda exc: errors.append(str(exc)))
        page.goto(page_path.as_uri())
        # Hero interval chart, two charts per strategy (both strategies drawn) and the benchmark.
        page.wait_for_function("document.querySelectorAll('svg.chart').length >= 6")

        overflow = page.evaluate(
            "document.documentElement.scrollWidth - document.documentElement.clientWidth"
        )
        # Every evidence section of the visible strategy states its claim as a heading.
        headings = page.locator("[data-strategy]:not([hidden]) h2").all_inner_texts()
        assert len(headings) == 3 and all(h.strip() for h in headings), headings

        out_dir = Path(os.environ.get("REPORT_SCREENSHOTS", tmp_path))
        out_dir.mkdir(parents=True, exist_ok=True)
        height = page.evaluate("document.documentElement.scrollHeight")
        page.set_viewport_size({"width": width, "height": height})
        page.wait_for_timeout(500)
        page.screenshot(path=str(out_dir / f"report-{width}-{scheme}.png"), animations="disabled")
        browser.close()

    assert errors == []
    assert overflow <= 0
