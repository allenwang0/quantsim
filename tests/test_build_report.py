"""
The published report (docs/index.html) and the README results block must say
what the committed result files say, because readers take those numbers as-is.

Expected values are recomputed here from the JSON and the test sources,
independently of scripts/build_report.py, so a generator bug that shifts a
number (wrong row, ratio of rounded values, a zero-cost run) fails here instead
of reaching the page. Standard library only; no market data.
"""

import html
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "demo" / "results.json"
BENCH = ROOT / "benchmarks" / "results.json"
TESTS = ROOT / "results" / "tests.json"
PAGE = ROOT / "docs" / "index.html"


def text_of(page: str) -> str:
    text = html.unescape(re.sub(r"<[^>]+>", "", page))
    return text.replace("\u2212", "-")


@pytest.fixture(scope="module")
def fresh(tmp_path_factory):
    out = tmp_path_factory.mktemp("report") / "index.html"
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_report.py"), "--out", str(out)],
        check=True,
        cwd=ROOT,
    )
    return out.read_text()


@pytest.fixture(scope="module")
def data():
    return (
        json.loads(RESULTS.read_text()),
        json.loads(BENCH.read_text()),
        json.loads(TESTS.read_text()),
    )


def test_hero_is_the_speed_ratio_of_the_largest_benchmark(fresh, data):
    # The hero is what a 5-second reader keeps: the ratio of the two medians at
    # the largest asset count, never a single run or a rounded value.
    _, bench, _ = data
    big = max(bench["rows"], key=lambda r: r["assets"])
    speedup = big["event_median_s"] / big["vectorized_median_s"]
    hero = re.search(r'<p class="hero-metric"><span class="value">([^<]+)</span>', fresh)
    assert hero, "hero metric missing"
    assert hero.group(1) == f"{speedup:,.0f}×"
    assert speedup > 1, "owner rule: the hero never states a zero or negative result"
    text = text_of(fresh)
    assert (
        f"{big['vectorized_median_s']:.2f}\u00a0s vs {big['event_median_s']:.1f}\u00a0s per backtest, "
        f"median of {bench['runs']} runs" in text
    )
    assert "Synthetic data" in text.split("Findings")[0], (
        "hero context must say the data is synthetic"
    )
    assert "one strategy" in text.split("Findings")[0]


def test_hero_parity_matches_the_tolerance_the_test_enforces(fresh):
    # "Agree within 0.5 pp" is only true while the parity test asserts it.
    source = (ROOT / "tests" / "test_engine_parity.py").read_text()
    tol = float(re.search(r"abs\(event_return - vector_return\) < ([\d.]+)", source).group(1))
    assert f"Both engines agree within {tol * 100:g}\u00a0pp" in text_of(fresh)


def test_benchmark_samples_support_the_reported_medians(data):
    # Medians on the page must be the medians of the saved samples, and the
    # demo results must embed the same benchmark the page reads.
    results, bench, _ = data
    import statistics

    for row in bench["rows"]:
        for engine in ("vectorized", "event"):
            runs = row[f"{engine}_runs_s"]
            assert len(runs) == bench["runs"]
            assert row[f"{engine}_median_s"] == statistics.median(runs)
    assert results["benchmark"]["rows"] == bench["rows"]
    assert len(bench["loadavg_before"]) == 3 and len(bench["loadavg_after"]) == 3


def test_published_backtests_use_nonzero_default_costs(data):
    # The audit found the report views running with zero slippage; published
    # numbers must carry the documented defaults.
    results, _, _ = data
    vec = (ROOT / "src" / "quantsim" / "backtesting" / "vectorized.py").read_text()
    commission = float(re.search(r"commission_rate: float = ([\d.]+)", vec).group(1))
    slippage = float(re.search(r"slippage_pct: float = ([\d.]+)", vec).group(1))
    assert commission > 0 and slippage > 0
    for strategy in results["strategies"].values():
        wf_costs = strategy["walk_forward"]["costs"]
        assert (
            wf_costs == f"{commission * 100:g}% commission + {slippage * 100:g}% slippage per trade"
        )
        assert "no commission" in strategy["fill_timing"]["costs"]
        assert strategy["cost_sensitivity"]["default_bps"] == commission * 10_000
        assert (
            strategy["cost_sensitivity"]["default_bps"] in strategy["cost_sensitivity"]["cost_bps"]
        )


def test_overfitting_numbers_match_results(fresh, data):
    results, _, tests = data
    wf = results["strategies"]["sma"]["walk_forward"]
    text = text_of(fresh)
    assert f"After {wf['total_trials']} trials, the deflated Sharpe certifies no edge" in text
    assert f"the deflated Sharpe is {wf['dsr']:.2f}" in text
    inside = sum(1 for w in wf["windows"] if w["ci_low"] <= 0 <= w["ci_high"])
    assert f"{inside} of {len(wf['windows'])} one-year Sharpe intervals include zero" in text
    assert f"{tests['passed']:,}" in text and tests["failed"] == 0


def test_fill_timing_gap_matches_results(fresh, data):
    results, _, _ = data
    fill = results["strategies"]["sma"]["fill_timing"]
    gap = fill["same_bar_open_return_pct"] - fill["next_bar_open_return_pct"]
    assert f"SMA's return by {abs(gap):.1f}\u00a0pp" in text_of(fresh)


def test_every_evidence_reference_resolves(fresh):
    refs = set(re.findall(r'href="#evidence-(e\d+)"', fresh))
    rows = set(re.findall(r'id="evidence-(e\d+)"', fresh))
    assert refs and refs <= rows, refs - rows


def test_committed_page_and_readme_are_current(fresh, tmp_path):
    # Pages publishes docs/ as committed; a stale page would show numbers that
    # are no longer in the result files.
    assert fresh == PAGE.read_text(), (
        "docs/index.html is stale: run python3 scripts/build_report.py --readme README.md"
    )
    readme = tmp_path / "README.md"
    readme.write_text((ROOT / "README.md").read_text())
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_report.py"),
            "--out",
            str(tmp_path / "x.html"),
            "--readme",
            str(readme),
        ],
        check=True,
        cwd=ROOT,
    )
    assert readme.read_text() == (ROOT / "README.md").read_text(), "README results block is stale"
