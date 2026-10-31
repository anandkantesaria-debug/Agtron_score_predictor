"""Benchmark warm server-side batch latency for the /recommend endpoint.

Usage:
    # In-process benchmark via FastAPI TestClient (default: 5 warmup + 100 measured requests)
    python benchmark_recommend.py

    # Benchmark against a live local or remote server (e.g., Render deployment)
    python benchmark_recommend.py --url https://your-app.onrender.com --requests 30
"""

from __future__ import annotations

import argparse
import statistics
import time
from typing import Any

import httpx
from fastapi.testclient import TestClient

from api.main import app

SAMPLE_PAYLOAD: dict[str, Any] = {
    "Consumed": 500.0,
    "Temp": 440.0,
    "Quench": 73.0,
    "Month": "1",
    "Blend": "284",
    "target_agtron": 45.0,
}


def percentile(sorted_values: list[float], pct: float) -> float:
    """Compute linear-interpolated percentile from a pre-sorted list of floats."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (pct / 100.0) * (len(sorted_values) - 1)
    low = int(rank)
    high = min(low + 1, len(sorted_values) - 1)
    weight = rank - low
    return sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight


def summarize(values: list[float]) -> dict[str, float]:
    """Return min, P50, P95, P99, mean, and max for a list of latency measurements."""
    ordered = sorted(values)
    return {
        "min": round(ordered[0], 3),
        "p50": round(percentile(ordered, 50), 3),
        "p95": round(percentile(ordered, 95), 3),
        "p99": round(percentile(ordered, 99), 3),
        "mean": round(statistics.fmean(ordered), 3),
        "max": round(ordered[-1], 3),
    }


def run_benchmark(url: str | None, warmup: int, num_requests: int) -> None:
    """Run warmup and measured requests against /recommend and print summary metrics."""
    inference_times: list[float] = []
    compute_times: list[float] = []
    total_handler_times: list[float] = []
    client_rtt_times: list[float] = []
    candidates_evaluated = 0
    last_server_timing = ""

    if url:
        base_url = url.rstrip("/")
        target_label = f"Remote Server ({base_url}/recommend)"
        with httpx.Client(base_url=base_url, timeout=30.0) as http_client:
            for _ in range(warmup):
                resp = http_client.post("/recommend", json=SAMPLE_PAYLOAD)
                resp.raise_for_status()

            for _ in range(num_requests):
                t0 = time.perf_counter()
                resp = http_client.post("/recommend", json=SAMPLE_PAYLOAD)
                rtt_ms = (time.perf_counter() - t0) * 1000.0
                resp.raise_for_status()
                body = resp.json()
                inference_times.append(float(body["inference_latency_ms"]))
                compute_times.append(float(body["compute_latency_ms"]))
                total_handler_times.append(float(body["total_latency_ms"]))
                client_rtt_times.append(rtt_ms)
                candidates_evaluated = int(body["candidates_evaluated"])
                last_server_timing = resp.headers.get("server-timing", "")
    else:
        target_label = "In-Process FastAPI TestClient (/recommend)"
        with TestClient(app) as test_client:
            for _ in range(warmup):
                resp = test_client.post("/recommend", json=SAMPLE_PAYLOAD)
                assert resp.status_code == 200

            for _ in range(num_requests):
                t0 = time.perf_counter()
                resp = test_client.post("/recommend", json=SAMPLE_PAYLOAD)
                rtt_ms = (time.perf_counter() - t0) * 1000.0
                assert resp.status_code == 200
                body = resp.json()
                inference_times.append(float(body["inference_latency_ms"]))
                compute_times.append(float(body["compute_latency_ms"]))
                total_handler_times.append(float(body["total_latency_ms"]))
                client_rtt_times.append(rtt_ms)
                candidates_evaluated = int(body["candidates_evaluated"])
                last_server_timing = resp.headers.get("server-timing", "")

    metrics = [
        ("CatBoost Batch Inference (pure C++)", summarize(inference_times)),
        ("Recommendation Compute (grid+DF+rank)", summarize(compute_times)),
        ("Total Server Handler (/recommend)", summarize(total_handler_times)),
        ("Client-Observed Round-Trip (RTT)", summarize(client_rtt_times)),
    ]

    print("=" * 84)
    print(f"Benchmark Target      : {target_label}")
    print(f"Warmup / Measured     : {warmup} warmup / {num_requests} warm requests")
    print(f"Candidates per Batch  : {candidates_evaluated} (Temp x Quench parameter grid)")
    print(f"Sample Server-Timing  : {last_server_timing}")
    print("=" * 84)
    print(
        f"{'Metric Tier':<40} {'P50 (ms)':>10} {'P95 (ms)':>10} {'P99 (ms)':>10} {'Mean (ms)':>10}"
    )
    print("-" * 84)
    for label, stats in metrics:
        print(
            f"{label:<40} {stats['p50']:>10.3f} {stats['p95']:>10.3f} "
            f"{stats['p99']:>10.3f} {stats['mean']:>10.3f}"
        )
    print("=" * 84)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark warm batch latency for the /recommend endpoint."
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="Optional base URL of a running server (e.g. https://your-app.onrender.com).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of warmup requests before recording latency (default: 5).",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=100,
        help="Number of warm requests to measure (default: 100).",
    )
    args = parser.parse_args()
    run_benchmark(url=args.url, warmup=args.warmup, num_requests=args.requests)


if __name__ == "__main__":
    main()

