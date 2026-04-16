"""
Structured logging configuration for QuantSim.

Two modes:
  Development (default): colored human-readable output to stdout
  Production (QUANTSIM_LOG_JSON=true): JSON lines to stdout/file for log aggregation

Structured JSON fields on every log record:
  timestamp, level, logger, message, module, function, line
  + any extra kwargs passed via logger.info("msg", extra={"order_id": "..."})

Usage:
    from core.logging_config import setup_logging
    setup_logging()  # call once at startup

    import logging
    logger = logging.getLogger(__name__)
    logger.info("Order submitted", extra={"order_id": "abc", "symbol": "SPY"})
"""

import os
import sys
import json
import logging
import logging.handlers
from datetime import datetime, timezone
from pathlib import Path


class JSONFormatter(logging.Formatter):
    """
    Formats log records as newline-delimited JSON.
    Suitable for log aggregation systems (Datadog, CloudWatch, Splunk).
    """

    RESERVED_ATTRS = {
        "args", "created", "exc_info", "exc_text", "filename",
        "funcName", "id", "levelname", "levelno", "lineno", "module",
        "msecs", "message", "msg", "name", "pathname", "process",
        "processName", "relativeCreated", "stack_info", "thread",
        "threadName",
    }

    def format(self, record: logging.LogRecord) -> str:
        record.message = record.getMessage()
        if record.exc_info:
            record.exc_text = self.formatException(record.exc_info)

        log_record = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.message,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if record.exc_text:
            log_record["exception"] = record.exc_text

        # Add any extra fields
        for key, val in record.__dict__.items():
            if key not in self.RESERVED_ATTRS and not key.startswith("_"):
                try:
                    json.dumps(val)  # test serializability
                    log_record[key] = val
                except (TypeError, ValueError):
                    log_record[key] = str(val)

        return json.dumps(log_record)


class ColoredFormatter(logging.Formatter):
    """
    Human-readable colored formatter for development.
    Colors: DEBUG=cyan, INFO=green, WARNING=yellow, ERROR=red, CRITICAL=red+bold
    """

    COLORS = {
        "DEBUG":    "\033[36m",    # cyan
        "INFO":     "\033[32m",    # green
        "WARNING":  "\033[33m",    # yellow
        "ERROR":    "\033[31m",    # red
        "CRITICAL": "\033[1;31m",  # bold red
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        reset = self.RESET

        # Short logger name (last two components)
        parts = record.name.split(".")
        short_name = ".".join(parts[-2:]) if len(parts) > 1 else record.name

        ts = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        level = f"{color}{record.levelname:8s}{reset}"
        name = f"\033[34m{short_name:20s}{reset}"  # blue
        msg = record.getMessage()

        if record.exc_info:
            msg += "\n" + self.formatException(record.exc_info)

        return f"{ts} {level} {name} {msg}"


def setup_logging(
    level: str = None,
    log_file: str = None,
    json_mode: bool = None,
) -> None:
    """
    Configure logging for the entire quantsim application.

    Args:
        level:    Log level string. Defaults to QUANTSIM_LOG_LEVEL env var or INFO.
        log_file: Optional path to write logs. Defaults to QUANTSIM_LOG_FILE env var.
        json_mode: Use JSON formatter. Defaults to QUANTSIM_LOG_JSON env var.
    """
    level = level or os.environ.get("QUANTSIM_LOG_LEVEL", "INFO").upper()
    log_file = log_file or os.environ.get("QUANTSIM_LOG_FILE", "")
    json_mode = json_mode if json_mode is not None else \
        os.environ.get("QUANTSIM_LOG_JSON", "false").lower() == "true"

    numeric_level = getattr(logging, level, logging.INFO)

    # Root logger
    root = logging.getLogger()
    root.setLevel(numeric_level)

    # Clear existing handlers
    root.handlers.clear()

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(numeric_level)
    if json_mode:
        console.setFormatter(JSONFormatter())
    else:
        console.setFormatter(ColoredFormatter())
    root.addHandler(console)

    # File handler (if configured)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        # Rotating: 10MB per file, keep 5 backups
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(JSONFormatter())  # always JSON in files
        root.addHandler(file_handler)

    # Silence noisy third-party loggers
    for noisy in ("yfinance", "urllib3", "requests", "websocket", "asyncio",
                  "numba", "cvxpy", "arch", "matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.getLogger("quantsim").info(
        "Logging configured",
        extra={"level": level, "json_mode": json_mode, "log_file": log_file or "none"},
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger. Prefer this over logging.getLogger() directly so we
    can intercept and add context (run_id, strategy_id) globally.
    """
    return logging.getLogger(name)
