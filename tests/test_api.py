"""Tests for the LLM serving API.

These run with the light dependencies only (no vLLM, no GPU):
the model engine is not loaded, so /health reports degraded and
/generate returns 503 -- both are the intended behaviour when no
model is loaded.

The service layer is also tested directly with a stub engine, so
generation behaviour is covered without a model.
"""

from fastapi.testclient import TestClient

from app.api.schemas import GenerateRequest
from app.core.config import Settings
from app.infrastructure.vllm_engine import GenerationResult, ModelEngine
from app.main import app
from app.services.generation import GenerationService

client = TestClient(app)


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
