"""
site/card.json feeds the portfolio site, which rejects files that fail its
schema (copied here from allenwang0.github.io scripts/card.schema.json,
schema_version 1).

The card is the project's headline on the portfolio, so it follows the same
rule as the report's hero: it states a positive, measured result with its
conditions, never a zero or negative one. That result is the vectorized
engine's speedup over the event engine: the ratio of the two median run times
at the largest asset count in the committed benchmark. The deflated-Sharpe
check (0 on synthetic data, by design) belongs in the report body, not here.
"""

import json
import re
from pathlib import Path

import jsonschema
import pytest

from quantsim.cli.report import card, parity_tolerance
from tests.test_report import built  # noqa: F401  (module-scoped report fixture)

ROOT = Path(__file__).parent.parent
SCHEMA = json.loads((Path(__file__).parent / "fixtures" / "card.schema.json").read_text())
COMMITTED_CARD = ROOT / "site" / "card.json"
BENCHMARK = ROOT / "benchmarks" / "results.json"


def check(document):
    jsonschema.Draft202012Validator(
        SCHEMA, format_checker=jsonschema.Draft202012Validator.FORMAT_CHECKER
    ).validate(document)
    for row in document["chart"]["rows"]:
        assert row["low"] <= row["mid"] <= row["high"], row


def expected_speedup(bench):
    big = max(bench["rows"], key=lambda r: r["assets"])
    return big["event_median_s"] / big["vectorized_median_s"]


def test_card_headlines_the_benchmark_speedup(built):  # noqa: F811
    results, _ = built
    document = card(results, parity_tolerance(ROOT / "tests" / "test_engine_parity.py"))
    check(document)
    speedup = expected_speedup(results["benchmark"])
    # Ratio of the medians, not of one run or of rounded values.
    assert document["metric"]["value"] == round(speedup, 1)
    # Owner rule: the portfolio headline is never zero or negative.
    assert document["metric"]["value"] > 1 and document["metric"]["higher_is_better"]
    assert "Deflated Sharpe" not in document["metric"]["label"]
    # The conditions travel with the number.
    assert "synthetic data" in document["takeaway"]
    assert "within 0.5 pp" in document["takeaway"]


def test_parity_tolerance_is_read_from_the_parity_test():
    # The card's "agree within" claim is only true while the test asserts it.
    source = (ROOT / "tests" / "test_engine_parity.py").read_text()
    tol = float(re.search(r"abs\(event_return - vector_return\) < ([\d.]+)", source).group(1))
    assert parity_tolerance(ROOT / "tests" / "test_engine_parity.py") == tol


@pytest.mark.skipif(not COMMITTED_CARD.exists(), reason="site/card.json not generated yet")
def test_committed_card_is_the_committed_median_speedup():
    document = json.loads(COMMITTED_CARD.read_text())
    check(document)
    speedup = expected_speedup(json.loads(BENCHMARK.read_text()))
    assert document["metric"]["value"] == round(speedup, 1)
    assert document["metric"]["value"] > 1
