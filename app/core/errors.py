"""Domain error types.

Services and infrastructure raise these; the API layer translates them
into HTTP responses with a clean error body. Nothing here knows about
HTTP status codes beyond the mapping the API layer needs — the status
code lives on the exception so the translation is a one-liner.
"""

from __future__ import annotations


class ServiceError(Exception):
    """Base class for errors that map to a clean API error response."""

    status_code: int = 500
    code: str = "internal_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ModelNotLoadedError(ServiceError):
    """The model engine has not been loaded yet (no GPU / vLLM)."""

    status_code = 503
    code = "model_not_loaded"


class ServerBusyError(ServiceError):
    """All generation slots are in use; the caller should retry later."""

    status_code = 503
    code = "server_busy"


class GenerationError(ServiceError):
    """The engine failed while generating a completion."""

    status_code = 500
    code = "generation_failed"
