"""Structured logging.

Log lines are `key=value` pairs, one event per line, so they are readable
in a terminal and parseable by a log aggregator without a JSON parser.
The level comes from the ``LOG_LEVEL`` env var (via Settings).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

# LogRecord attributes that are part of the record itself, not event data.
_RESERVED = frozenset(
    {
        "args", "asctime", "created", "exc_info", "exc_text", "filename",
        "funcName", "levelname", "levelno", "lineno", "module", "msecs",
        "msg", "name", "pathname", "process", "processName",
        "relativeCreated", "stack_info", "thread", "threadName", "taskName",
    }
)


class KeyValueFormatter(logging.Formatter):
    """Format a log record as `ts=... level=... logger=... msg=... k=v ...`."""

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created, tz=UTC).isoformat()
        parts = [
            f"ts={ts}",
            f"level={record.levelname}",
            f"logger={record.name}",
            f"msg={record.getMessage()!r}",
        ]
        for key, value in sorted(record.__dict__.items()):
            if key in _RESERVED or key.startswith("_"):
                continue
            parts.append(f"{key}={value!r}")
        if record.exc_info:
            parts.append(f"exc_info={self.formatException(record.exc_info)!r}")
        return " ".join(parts)


def configure_logging(level: str = "INFO") -> None:
    """Install the structured handler on the root logger (idempotent)."""
    root = logging.getLogger()
    if any(isinstance(h, logging.StreamHandler) and
           isinstance(h.formatter, KeyValueFormatter) for h in root.handlers):
        root.setLevel(level)
        return
    handler = logging.StreamHandler()
    handler.setFormatter(KeyValueFormatter())
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """Return a module logger (configure_logging must have run first)."""
    return logging.getLogger(name)
