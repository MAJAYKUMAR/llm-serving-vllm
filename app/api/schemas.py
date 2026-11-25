"""Request/response schemas for the HTTP API.

``GenerateRequest.max_tokens`` is optional: when omitted, the service
fills in the ``MAX_TOKENS`` default from Settings, so the default stays
configurable via environment instead of being baked into the schema.

``GenerateRequest.prompt`` is capped at ``MAX_PROMPT_CHARS`` characters
(default 4096) so an oversize prompt is rejected with a 422 instead of
blowing the model's context window and failing deep inside vLLM.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from app.core.config import Settings


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Input prompt")
    max_tokens: int | None = Field(
        default=None, ge=1, le=1024,
        description="Maximum new tokens to generate (default: MAX_TOKENS env)",
    )
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)

    @field_validator("prompt")
    @classmethod
    def _check_prompt_length(cls, value: str) -> str:
        limit = Settings.from_env().max_prompt_chars
        if len(value) > limit:
            raise ValueError(f"Prompt exceeds the {limit}-character limit")
        return value


class GenerateResponse(BaseModel):
    text: str
    model: str
    prompt_tokens: int
    generated_tokens: int
    latency_seconds: float


class HealthResponse(BaseModel):
    status: str
    model: str
    model_loaded: bool


class ErrorBody(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorBody
