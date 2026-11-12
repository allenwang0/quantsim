"""
The paper_trading package must never reach a live Alpaca account by accident.

A mistyped ALPACA_PAPER value ("1", "yes", "") used to be read as "not true",
which meant live trading. These tests pin the contract: unset means paper,
only "true"/"false" are accepted, and live mode needs QUANTSIM_ALLOW_LIVE=1 too.
No network: TradingClient is replaced by a recorder.
"""
import importlib

import pytest

import quantsim.paper_trading.alpaca_handler as handler
class RecordingTradingClient:
    calls = []

    def __init__(self, **kwargs):
        RecordingTradingClient.calls.append(kwargs)


@pytest.fixture
def recorder(monkeypatch):
    for name in ("ALPACA_PAPER", "ALPACA_API_KEY", "ALPACA_SECRET_KEY", "QUANTSIM_ALLOW_LIVE"):
        monkeypatch.delenv(name, raising=False)
    RecordingTradingClient.calls = []
    yield RecordingTradingClient
    importlib.reload(handler)


def build_client(monkeypatch):
    """Re-import the handler under the current environment, as a fresh process would."""
    importlib.reload(handler)
    monkeypatch.setattr(handler, "ALPACA_AVAILABLE", True)
    monkeypatch.setattr(handler, "TradingClient", RecordingTradingClient, raising=False)
    return handler.AlpacaTradingClient()


def test_unset_environment_constructs_paper_client(recorder, monkeypatch):
    build_client(monkeypatch)
    assert recorder.calls == [{"api_key": "", "secret_key": "", "paper": True}]


@pytest.mark.parametrize("value", ["1", "yes", "", "TRUE", "false", "ture"])
@pytest.mark.parametrize("allow_live", [None, "1"])
def test_only_explicit_false_plus_allow_live_reaches_live(recorder, monkeypatch, value, allow_live):
    monkeypatch.setenv("ALPACA_PAPER", value)
    if allow_live is not None:
        monkeypatch.setenv("QUANTSIM_ALLOW_LIVE", allow_live)

    try:
        build_client(monkeypatch)
    except ValueError:
        assert recorder.calls == []
        return

    assert len(recorder.calls) == 1
    expected_paper = not (value == "false" and allow_live == "1")
    assert recorder.calls[0]["paper"] is expected_paper


@pytest.mark.parametrize("value", ["1", "yes", "", "ture"])
def test_unrecognized_value_raises(recorder, monkeypatch, value):
    monkeypatch.setenv("ALPACA_PAPER", value)
    monkeypatch.setenv("QUANTSIM_ALLOW_LIVE", "1")
    with pytest.raises(ValueError):
        build_client(monkeypatch)


def test_false_without_allow_live_raises(recorder, monkeypatch):
    monkeypatch.setenv("ALPACA_PAPER", "false")
    with pytest.raises(ValueError, match="QUANTSIM_ALLOW_LIVE"):
        build_client(monkeypatch)
