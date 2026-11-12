"""
site/card.json feeds the portfolio site, which rejects files that fail its
schema (copied here from allenwang0.github.io scripts/card.schema.json,
schema_version 1). The card's numbers must be the report's numbers.
"""

import json
from pathlib import Path

import jsonschema
import pytest

from quantsim.cli.report import card
from tests.test_report import built  # noqa: F401  (module-scoped report fixture)

SCHEMA = json.loads((Path(__file__).parent / "fixtures" / "card.schema.json").read_text())
COMMITTED_CARD = Path(__file__).parent.parent / "site" / "card.json"


def check(document):
    jsonschema.Draft202012Validator(
        SCHEMA, format_checker=jsonschema.Draft202012Validator.FORMAT_CHECKER
    ).validate(document)
    for row in document["chart"]["rows"]:
        assert row["low"] <= row["mid"] <= row["high"], row


def test_card_from_report_validates_and_matches_results(built):  # noqa: F811
    results, _ = built
    document = card(results, "sma")
    check(document)
    assert document["metric"]["value"] == results["strategies"]["sma"]["walk_forward"]["dsr"]
    windows = results["strategies"]["sma"]["walk_forward"]["windows"]
    assert [r["mid"] for r in document["chart"]["rows"]] == [w["oos_sharpe"] for w in windows][-12:]


@pytest.mark.skipif(not COMMITTED_CARD.exists(), reason="site/card.json not generated yet")
def test_committed_card_validates():
    check(json.loads(COMMITTED_CARD.read_text()))
