# llm-serving-vllm

Production-style LLM inference service: **FastAPI + vLLM**, packaged with
Docker and Kubernetes manifests and instrumented with Prometheus metrics.

It serves a small open instruction model
([Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct))
through a clean HTTP API. Swap `MODEL_NAME` for any vLLM-supported model.

## Why

Running an LLM in production is mostly not about the model — it is about
the serving layer around it: request validation, health/readiness probes,
latency and token metrics, resource limits, and a repeatable deploy.
This repo is a minimal, working example of that layer.

## Architecture

Layered, so each concern can change independently:

```
app/
  main.py                  composition root: builds the app, wires layers,
                           registers error handlers (thin, no logic)
  __main__.py              `python -m app` entrypoint
  api/                     HTTP only: routes + Pydantic schemas
    routes.py              /generate, /health, /metrics, /
    schemas.py             GenerateRequest/Response, HealthResponse,
                           ErrorResponse
  services/                business logic (no FastAPI, no vLLM)
    generation.py          GenerationService: defaults, timing, metrics,
                           logging, error translation
  core/                    innermost layer, no external dependencies
    config.py              Settings from env vars (MODEL_NAME, HOST, PORT,
                           MAX_TOKENS, MAX_TOKENS_LIMIT, LOG_LEVEL,
                           LOAD_MODEL_ON_STARTUP)
    logging.py             structured key=value log lines
    errors.py              ModelNotLoadedError (503), GenerationError (500)
    metrics.py             Prometheus counters/histogram
  infrastructure/          adapters to outside systems
    vllm_engine.py         ModelEngine: lazy vLLM import, plain-dataclass
                           results

                  +------------------+
  curl / client   |   FastAPI app    |
  ───────────────▶|  app/main.py     |
                  |  /generate       |──▶ GenerationService ──▶ vLLM engine
                  |  /health         |    (lazy-loaded)
                  |  /metrics        |    Qwen2.5-0.5B-Instruct
                  +--------+---------+
                           | Prometheus scrape
                           ▼
                    llm_requests_total
                    llm_generation_latency_seconds
                    llm_tokens_generated_total
```

- **Lazy vLLM import** — `app/infrastructure/vllm_engine.py` imports vLLM
  only when the model is loaded, so the API (health, metrics, validation)
  can be tested on a CPU-only machine with the light dependencies.
- **Error handling** — a missing model returns `503` and a generation
  failure returns `500`, both as a clean
  `{"error": {"code": ..., "message": ...}}` body; tracebacks never leak
  to clients. Startup model-load failure leaves the service up in a
  degraded state instead of crashing it.
- **Structured logging** — one `key=value` line per event
  (`ts=... level=INFO logger=... msg=...`), level via `LOG_LEVEL`.
- **Metrics** — request counts, generation latency histogram, and tokens
  generated, exposed on `/metrics` for Prometheus.
- **Kubernetes** — `k8s/` has a Deployment (health/readiness probes,
  CPU/memory requests and limits, GPU request commented as optional)
  and a ClusterIP Service.

## Quickstart (CPU-only, no model)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn app.main:app --reload
```

`/health` reports `degraded` and `/generate` returns `503` until a model
is loaded — that is the intended behaviour without a GPU/vLLM.

## Run with a model (GPU)

```bash
pip install -r requirements.txt
LOAD_MODEL_ON_STARTUP=1 uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Or pick a different model:

```bash
MODEL_NAME=Qwen/Qwen2.5-7B-Instruct LOAD_MODEL_ON_STARTUP=1 \
  uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Example

```bash
curl -s http://localhost:8000/health
# {"status":"ok","model":"Qwen/Qwen2.5-0.5B-Instruct","model_loaded":true}

curl -s -X POST http://localhost:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Explain what a transformer model is in one sentence.","max_tokens":64,"temperature":0.7}'
# {
#   "text": "A transformer model is a neural network architecture that ...",
#   "model": "Qwen/Qwen2.5-0.5B-Instruct",
#   "prompt_tokens": 14,
#   "generated_tokens": 38,
#   "latency_seconds": 0.421
# }

curl -s http://localhost:8000/metrics | grep llm_requests_total
```

## Docker

```bash
docker build -t llm-serving-vllm:latest .
docker run --gpus all -p 8000:8000 llm-serving-vllm:latest
```

For a GPU image, base the Dockerfile on `nvidia/cuda:12.4.1-runtime-ubuntu22.04`
(see the comment at the top of `Dockerfile`).

## Kubernetes

```bash
kubectl apply -f k8s/
kubectl port-forward svc/llm-serving 8000:80
```

Uncomment the `nvidia.com/gpu` resource lines in `k8s/deployment.yaml`
on a GPU node pool.

## Load test

```bash
python scripts/load_test.py --url http://localhost:8000 --requests 50 --concurrency 5
# requests:     50
# concurrency:  5
# status codes: {200: 50}
# throughput:   12.34 req/s
# latency p50:  0.401s
# latency p90:  0.512s
# latency p99:  0.640s
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

The tests run against the light dependencies only — no vLLM, no GPU —
and assert the no-model behaviour (`/health` degraded, `/generate` 503
with a clean error body), plus config-from-env and the service layer
against a stub engine. Lint with `ruff check .`.

## License

MIT — see [LICENSE](LICENSE).
