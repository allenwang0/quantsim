"""Configuration, logging setup and package metadata."""

import os
import tempfile

import pytest


class TestConfig:
    def test_default_config_loads(self):
        from quantsim.core.config import QuantSimConfig

        cfg = QuantSimConfig()
        assert cfg.initial_capital > 0
        assert cfg.db_path.endswith(".db")

    def test_config_env_override(self):

        os.environ["QUANTSIM_CAPITAL"] = "250000"
        from quantsim.core.config import QuantSimConfig

        cfg = QuantSimConfig()
        assert cfg.initial_capital == 250000
        del os.environ["QUANTSIM_CAPITAL"]

    def test_config_save_load(self):

        from quantsim.core.config import QuantSimConfig

        cfg = QuantSimConfig()
        cfg.backtest.initial_capital = 75000

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name

        try:
            cfg.save(path)
            loaded = QuantSimConfig.load(path)
            assert loaded.backtest.initial_capital == 75000
        finally:
            os.unlink(path)

    def test_config_validation(self):
        from quantsim.core.config import QuantSimConfig

        cfg = QuantSimConfig()
        warnings = cfg.validate()
        # Should warn about missing Alpaca credentials
        assert isinstance(warnings, list)


class TestLoggingConfig:
    def test_setup_logging_runs(self):
        from quantsim.core.logging_config import setup_logging

        setup_logging(level="WARNING")  # quiet for tests
        import logging

        assert logging.getLogger().level == logging.WARNING

    def test_json_formatter_produces_valid_json(self):
        import json
        import logging

        from quantsim.core.logging_config import JSONFormatter

        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=1,
            msg="hello %s",
            args=("world",),
            exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["message"] == "hello world"
        assert "timestamp" in parsed
        assert "level" in parsed

    def test_get_logger_returns_logger(self):
        from quantsim.core.logging_config import get_logger

        logger = get_logger("test.module")
        import logging

        assert isinstance(logger, logging.Logger)


class TestPackageConfiguration:
    def test_pyproject_exists(self):
        assert os.path.exists("pyproject.toml"), "pyproject.toml not found"

    def test_pyproject_parseable(self):
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                pytest.skip("tomllib not available; skip toml parse test")
        with open("pyproject.toml", "rb") as f:
            data = tomllib.load(f)
        assert "project" in data
        assert data["project"]["name"] == "quantsim"
        assert "version" in data["project"]

    def test_lock_file_exists(self):
        assert os.path.exists("uv.lock")
