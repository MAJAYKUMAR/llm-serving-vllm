"""Prometheus metrics.

Defined once at module level: Prometheus client registries are global,
so re-creating these per app instance would raise on duplicate
registration. Metric names are part of the public /metrics contract.
"""

from __future__ import annotations

from prometheus_client import Counter, Histogram

REQUESTS_TOTAL = Counter(
    "llm_requests_total",
    "Total generation requests",
    ["endpoint", "status"],
)
GENERATION_LATENCY = Histogram(
    "llm_generation_latency_seconds",
    "Time spent generating a completion",
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)
TOKENS_GENERATED = Counter(
    "llm_tokens_generated_total",
    "Total tokens generated",
)
