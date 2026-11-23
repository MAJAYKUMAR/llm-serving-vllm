#!/usr/bin/env python3
"""Simple load test for the LLM serving API.

Hammers POST /generate with a fixed prompt and reports latency
percentiles plus requests/sec. The model must already be loaded
(or the run will report 503s, which are counted separately).

Usage:
    python scripts/load_test.py --url http://localhost:8000 --requests 50 --concurrency 5
"""

import argparse
import concurrent.futures
import statistics
import time

import httpx


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(pct / 100 * len(ordered))))
    return ordered[idx]


def one_request(url: str, prompt: str) -> tuple[float, int]:
    start = time.monotonic()
    with httpx.Client(timeout=120.0) as client:
        resp = client.post(
            f"{url}/generate",
            json={"prompt": prompt, "max_tokens": 32, "temperature": 0.0},
        )
    return time.monotonic() - start, resp.status_code


def main() -> None:
    parser = argparse.ArgumentParser(description="Load test the LLM serving API")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--prompt", default="Explain what a transformer model is in one sentence."
    )
    args = parser.parse_args()

    latencies: list[float] = []
    statuses: dict[int, int] = {}
    start = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(one_request, args.url, args.prompt)
            for _ in range(args.requests)
        ]
        for fut in concurrent.futures.as_completed(futures):
            latency, status = fut.result()
            latencies.append(latency)
            statuses[status] = statuses.get(status, 0) + 1
    total = time.monotonic() - start

    print(f"requests:     {args.requests}")
    print(f"concurrency:  {args.concurrency}")
    print(f"status codes: {statuses}")
    print(f"total time:   {total:.2f}s")
    print(f"throughput:   {args.requests / total:.2f} req/s")
    print(f"latency p50:  {percentile(latencies, 50):.3f}s")
    print(f"latency p90:  {percentile(latencies, 90):.3f}s")
    print(f"latency p99:  {percentile(latencies, 99):.3f}s")
    print(f"latency mean: {statistics.mean(latencies):.3f}s")


if __name__ == "__main__":
    main()
