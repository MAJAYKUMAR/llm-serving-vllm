"""Tests for the LLM serving API.

These run with the light dependencies only (no vLLM, no GPU):
the model engine is not loaded, so /health reports degraded and
/generate returns 503 -- both are the intended behaviour when no
model is loaded.

The service layer is also tested directly with a stub engine, so
generation behaviour is covered without a model.
"""

import threading

import pytest
from fastapi.testclient import TestClient

from app.api.schemas import GenerateRequest
from app.core.config import Settings
from app.core.errors import ServerBusyError
from app.infrastructure.vllm_engine import GenerationResult, ModelEngine
from app.main import app
from app.services.generation import GenerationService

client = TestClient(app)


@pytest.fixture(autouse=True)
def _restore_generation_service():
    """Restore app.state.generation_service after each test.

    Needed because pytest's monkeypatch deletes the attribute on
    teardown instead of restoring it: Starlette's State stores values
    in an internal dict, not the instance __dict__, so monkeypatch
    records "not set" and deletes it on undo.
    """
    original = app.state.generation_service
    yield
    app.state.generation_service = original


class StubEngine(ModelEngine):
    """Fake engine that pretends a model is loaded."""

    def __init__(self) -> None:
        super().__init__(model_name="stub-model")
        self.loaded = True
        self.last_call: dict | None = None

    def generate(
        self, prompt: str, max_tokens: int, temperature: float, top_p: float
    ) -> GenerationResult:
        self.last_call = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        return GenerationResult(
            text="stub reply",
            prompt_tokens=3,
            generated_tokens=2,
            latency_seconds=0.01,
        )


# ---------------------------------------------------------------------------
# Original smoke tests (unchanged behaviour)
# ---------------------------------------------------------------------------


def test_root_lists_endpoints():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "/generate" in resp.json()["endpoints"]


def test_health_without_model():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "degraded"
    assert body["model_loaded"] is False


def test_generate_without_model_is_503():
    resp = client.post("/generate", json={"prompt": "Hello"})
    assert resp.status_code == 503


def test_generate_rejects_empty_prompt():
    resp = client.post("/generate", json={"prompt": ""})
    assert resp.status_code == 422


def test_metrics_endpoint():
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "llm_requests_total" in resp.text


# ---------------------------------------------------------------------------
# New: error bodies, validation, config, service layer
# ---------------------------------------------------------------------------


def test_generate_without_model_has_error_body():
    resp = client.post("/generate", json={"prompt": "Hello"})
    assert resp.status_code == 503
    body = resp.json()
    assert body["error"]["code"] == "model_not_loaded"
    assert "message" in body["error"]
    assert "Traceback" not in resp.text


def test_generate_rejects_out_of_range_temperature():
    resp = client.post("/generate", json={"prompt": "Hello", "temperature": 5.0})
    assert resp.status_code == 422


def test_health_reports_default_model_name():
    resp = client.get("/health")
    assert resp.json()["model"] == "Qwen/Qwen2.5-0.5B-Instruct"


def test_config_from_env(monkeypatch):
    monkeypatch.setenv("MODEL_NAME", "Some/Model-1B")
    monkeypatch.setenv("MAX_TOKENS", "256")
    monkeypatch.setenv("PORT", "9000")
    monkeypatch.setenv("LOG_LEVEL", "debug")
    settings = Settings.from_env()
    assert settings.model_name == "Some/Model-1B"
    assert settings.max_tokens_default == 256
    assert settings.port == 9000
    assert settings.log_level == "DEBUG"


def test_config_invalid_int_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("PORT", "not-a-port")
    assert Settings.from_env().port == 8000


def test_service_generate_with_stub_engine():
    stub = StubEngine()
    service = GenerationService(engine=stub, settings=Settings())
    response = service.generate(GenerateRequest(prompt="Hello"))
    assert response.text == "stub reply"
    assert response.model == "stub-model"
    assert response.prompt_tokens == 3
    assert response.generated_tokens == 2
    # No max_tokens in the request -> service fills the env default.
    assert stub.last_call["max_tokens"] == Settings().max_tokens_default


def test_generate_with_stub_engine_over_http(monkeypatch):
    stub = StubEngine()
    monkeypatch.setattr(
        app.state,
        "generation_service",
        GenerationService(engine=stub, settings=Settings()),
    )
    resp = client.post(
        "/generate",
        json={"prompt": "Hello", "max_tokens": 16, "temperature": 0.0},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["text"] == "stub reply"
    assert body["generated_tokens"] == 2
    assert stub.last_call["max_tokens"] == 16


# ---------------------------------------------------------------------------
# Concurrency limit + request ids (commit: limit concurrent generations)
# ---------------------------------------------------------------------------


class BlockingStubEngine(ModelEngine):
    """Stub engine that blocks until released, to test saturation."""

    def __init__(self) -> None:
        super().__init__(model_name="blocking-stub")
        self.loaded = True
        self.entered = threading.Event()
        self.release = threading.Event()

    def generate(
        self, prompt: str, max_tokens: int, temperature: float, top_p: float
    ) -> GenerationResult:
        self.entered.set()
        self.release.wait(timeout=10)
        return GenerationResult(
            text="stub reply",
            prompt_tokens=3,
            generated_tokens=2,
            latency_seconds=0.01,
        )


def test_response_carries_request_id_header():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.headers.get("x-request-id")


def test_client_supplied_request_id_is_echoed():
    resp = client.get("/health", headers={"x-request-id": "test-123"})
    assert resp.status_code == 200
    assert resp.headers.get("x-request-id") == "test-123"


def test_generate_returns_503_when_saturated():
    stub = BlockingStubEngine()
    service = GenerationService(engine=stub, settings=Settings(max_concurrent_requests=1))
    errors: list = []

    def run_first() -> None:
        try:
            service.generate(GenerateRequest(prompt="Hello"))
        except Exception as exc:  # noqa: BLE001  (collected for the assert)
            errors.append(exc)

    thread = threading.Thread(target=run_first)
    thread.start()
    assert stub.entered.wait(timeout=5)
    try:
        with pytest.raises(ServerBusyError) as excinfo:
            service.generate(GenerateRequest(prompt="Hello again"))
        assert excinfo.value.status_code == 503
        assert excinfo.value.code == "server_busy"
    finally:
        stub.release.set()
        thread.join(timeout=5)
    assert not errors


def test_generate_http_503_when_saturated(monkeypatch):
    stub = StubEngine()
    service = GenerationService(engine=stub, settings=Settings(max_concurrent_requests=1))
    monkeypatch.setattr(app.state, "generation_service", service)
    # Hold the only slot, so the request hits the limit before the engine.
    assert service._inflight.acquire(blocking=False)
    try:
        resp = client.post("/generate", json={"prompt": "Hello"})
    finally:
        service._inflight.release()
    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "server_busy"


def test_config_max_concurrent_requests_from_env(monkeypatch):
    monkeypatch.setenv("MAX_CONCURRENT_REQUESTS", "3")
    assert Settings.from_env().max_concurrent_requests == 3
    assert Settings().max_concurrent_requests == 8
