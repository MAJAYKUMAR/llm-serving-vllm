"""Application configuration, driven by environment variables.

Every setting has a sane default so the service runs with no env vars
set at all (CPU-only, no model loaded). No secrets live here — model
names and ports are not sensitive, and there is nothing else to
configure.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    return os.environ.get(name, "1" if default else "0") == "1"


@dataclass(frozen=True)
class Settings:
    """All runtime configuration for the service."""

    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    host: str = "0.0.0.0"
    port: int = 8000
    max_tokens_default: int = 128
    max_tokens_limit: int = 1024
    max_concurrent_requests: int = 8
    log_level: str = "INFO"
    load_model_on_startup: bool = False

    @classmethod
    def from_env(cls) -> Settings:
        """Build settings from the environment, falling back to defaults."""
        return cls(
            model_name=os.environ.get("MODEL_NAME", cls.model_name),
            host=os.environ.get("HOST", cls.host),
            port=_env_int("PORT", cls.port),
            max_tokens_default=_env_int("MAX_TOKENS", cls.max_tokens_default),
            max_tokens_limit=_env_int("MAX_TOKENS_LIMIT", cls.max_tokens_limit),
            max_concurrent_requests=_env_int(
                "MAX_CONCURRENT_REQUESTS", cls.max_concurrent_requests
            ),
            log_level=os.environ.get("LOG_LEVEL", cls.log_level).upper(),
            load_model_on_startup=_env_bool(
                "LOAD_MODEL_ON_STARTUP", cls.load_model_on_startup
            ),
        )
