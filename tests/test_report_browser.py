"""
The report page must render in a real browser at phone and desktop widths in
light and dark mode, without console errors or horizontal scrolling.

Needs Playwright's Chromium (python -m playwright install chromium) and
network access for the pinned Plotly script. Set REPORT_SCREENSHOTS=<dir> to
keep the screenshots; docs/img/report.png in the README is one of them.
"""

import os
from pathlib import Path

import pytest

from tests.test_report import built  # noqa: F401  (module-scoped report fixture)

playwright = pytest.importorskip("playwright.sync_api")

pytestmark = pytest.mark.browser
VIEW_IDS = ["view-fill", "view-wfo", "view-sharpe", "view-cost", "view-bench"]


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
        page.wait_for_function("document.querySelectorAll('.chart .main-svg').length >= 4")

        overflow = page.evaluate(
            "document.documentElement.scrollWidth - document.documentElement.clientWidth"
        )
        failed_views = page.locator(".error").all_inner_texts()
        for view in VIEW_IDS[:4]:
            assert page.locator(f"#{view} .takeaway").inner_text().strip(), view

        out_dir = Path(os.environ.get("REPORT_SCREENSHOTS", tmp_path))
        out_dir.mkdir(parents=True, exist_ok=True)
        height = page.evaluate("document.documentElement.scrollHeight")
        page.set_viewport_size({"width": width, "height": height})
        page.wait_for_timeout(500)
        page.screenshot(path=str(out_dir / f"report-{width}-{scheme}.png"), animations="disabled")
        browser.close()

    assert errors == []
    assert failed_views == []
    assert overflow <= 0
