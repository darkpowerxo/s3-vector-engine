#!/usr/bin/env python3
"""Load-testing benchmark for production capacity validation.

Targets from PRD §13:
  - p50 latency ≤ 7ms (hot tier)
  - ≥ 50K writes/sec sustained
  - ≥ 10K queries/sec at recall@10 ≥ 0.95
  - 50M vectors per shard

Usage:
    # Quick smoke test
    python scripts/benchmark_load.py --mode smoke

    # Sustained write throughput
    python scripts/benchmark_load.py --mode write --duration 60 --concurrency 64

    # Query latency profiling
    python scripts/benchmark_load.py --mode query --num-queries 50000 --concurrency 128

    # Mixed read/write workload
    python scripts/benchmark_load.py --mode mixed --duration 120 --concurrency 64 --write-ratio 0.3

    # Full production validation
    python scripts/benchmark_load.py --mode full --duration 300
"""

import argparse
import asyncio
import time
import sys
import os
import statistics
import json
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Result dataclass ────────────────────────────────────────────────────────

@dataclass
class BenchmarkResult:
    mode: str
    duration_seconds: float
    total_ops: int
    errors: int
    ops_per_second: float
    latency_p50_ms: float
    latency_p90_ms: float
    latency_p99_ms: float
    latency_min_ms: float
    latency_max_ms: float
    latency_mean_ms: float
    extra: dict = field(default_factory=dict)

    def meets_target(self, target_ops: float, target_p50_ms: float) -> bool:
        return self.ops_per_second >= target_ops and self.latency_p50_ms <= target_p50_ms

    def summary(self) -> str:
        lines = [
            f"  Mode:           {self.mode}",
            f"  Duration:       {self.duration_seconds:.1f}s",
            f"  Total ops:      {self.total_ops:,}",
            f"  Errors:         {self.errors}",
            f"  Throughput:     {self.ops_per_second:,.0f} ops/s",
            f"  Latency p50:    {self.latency_p50_ms:.2f} ms",
            f"  Latency p90:    {self.latency_p90_ms:.2f} ms",
            f"  Latency p99:    {self.latency_p99_ms:.2f} ms",
            f"  Latency min:    {self.latency_min_ms:.2f} ms",
            f"  Latency max:    {self.latency_max_ms:.2f} ms",
            f"  Latency mean:   {self.latency_mean_ms:.2f} ms",
        ]
        for k, v in self.extra.items():
            lines.append(f"  {k}: {v}")
        return "\n".join(lines)


def compute_result(mode: str, latencies: list[float], errors: int, wall_time: float) -> BenchmarkResult:
    if not latencies:
        return BenchmarkResult(
            mode=mode, duration_seconds=wall_time, total_ops=0, errors=errors,
            ops_per_second=0, latency_p50_ms=0, latency_p90_ms=0, latency_p99_ms=0,
            latency_min_ms=0, latency_max_ms=0, latency_mean_ms=0,
        )
    ms = [t * 1000 for t in latencies]
    return BenchmarkResult(
        mode=mode,
        duration_seconds=wall_time,
        total_ops=len(latencies),
        errors=errors,
        ops_per_second=len(latencies) / wall_time if wall_time > 0 else 0,
        latency_p50_ms=float(np.percentile(ms, 50)),
        latency_p90_ms=float(np.percentile(ms, 90)),
        latency_p99_ms=float(np.percentile(ms, 99)),
        latency_min_ms=min(ms),
        latency_max_ms=max(ms),
        latency_mean_ms=statistics.mean(ms),
    )


# ── Workload generators ────────────────────────────────────────────────────

async def write_worker(
    client: httpx.AsyncClient,
    dim: int,
    namespace: str,
    stop_event: asyncio.Event,
    latencies: list,
    error_count: list,
    batch_size: int = 50,
):
    """Continuously upsert batches until stop_event is set."""
    idx = 0
    while not stop_event.is_set():
        vectors = []
        for _ in range(batch_size):
            vec = np.random.randn(dim).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            vectors.append({
                "id": f"load-{os.getpid()}-{idx}",
                "vector": vec.tolist(),
                "payload": {"bench": True, "idx": idx},
            })
            idx += 1
        t0 = time.perf_counter()
        try:
            resp = await client.post(
                f"/namespaces/{namespace}/vectors",
                json={"vectors": vectors},
            )
            elapsed = time.perf_counter() - t0
            if resp.status_code == 200:
                latencies.append(elapsed)
            else:
                error_count.append(1)
        except Exception:
            error_count.append(1)


async def query_worker(
    client: httpx.AsyncClient,
    dim: int,
    namespace: str,
    stop_event: asyncio.Event,
    latencies: list,
    error_count: list,
    top_k: int = 10,
):
    """Continuously query until stop_event is set."""
    while not stop_event.is_set():
        vec = np.random.randn(dim).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        t0 = time.perf_counter()
        try:
            resp = await client.post(
                "/search",
                json={
                    "vector": vec.tolist(),
                    "top_k": top_k,
                    "tenant_id": namespace,
                },
            )
            elapsed = time.perf_counter() - t0
            if resp.status_code == 200:
                latencies.append(elapsed)
            else:
                error_count.append(1)
        except Exception:
            error_count.append(1)


async def fixed_query_burst(
    client: httpx.AsyncClient,
    dim: int,
    namespace: str,
    num_queries: int,
    semaphore: asyncio.Semaphore,
    latencies: list,
    error_count: list,
    top_k: int = 10,
):
    """Fire exactly num_queries queries with bounded concurrency."""
    async def _one():
        async with semaphore:
            vec = np.random.randn(dim).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            t0 = time.perf_counter()
            try:
                resp = await client.post(
                    "/search",
                    json={
                        "vector": vec.tolist(),
                        "top_k": top_k,
                        "tenant_id": namespace,
                    },
                )
                elapsed = time.perf_counter() - t0
                if resp.status_code == 200:
                    latencies.append(elapsed)
                else:
                    error_count.append(1)
            except Exception:
                error_count.append(1)

    await asyncio.gather(*[_one() for _ in range(num_queries)])


# ── Benchmark modes ─────────────────────────────────────────────────────────

async def bench_smoke(base_url: str, dim: int, namespace: str) -> BenchmarkResult:
    """Quick health check: 100 queries, 10 writes."""
    print("  [smoke] Running quick smoke test...")
    latencies, errors = [], []
    async with httpx.AsyncClient(base_url=base_url, timeout=30) as client:
        # health
        r = await client.get("/health")
        assert r.status_code == 200, f"Health check failed: {r.status_code}"

        # upsert a few
        for i in range(10):
            vec = np.random.randn(dim).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            await client.post(
                f"/namespaces/{namespace}/vectors",
                json={"vectors": [{"id": f"smoke-{i}", "vector": vec.tolist()}]},
            )

        # query
        sem = asyncio.Semaphore(10)
        t0 = time.perf_counter()
        await fixed_query_burst(client, dim, namespace, 100, sem, latencies, errors)
        wall = time.perf_counter() - t0

    return compute_result("smoke", latencies, len(errors), wall)


async def bench_write(
    base_url: str, dim: int, namespace: str, duration: int, concurrency: int, batch_size: int,
) -> BenchmarkResult:
    """Sustained write throughput test."""
    print(f"  [write] {concurrency} workers, batch={batch_size}, {duration}s...")
    stop = asyncio.Event()
    latencies, errors = [], []
    async with httpx.AsyncClient(base_url=base_url, timeout=30) as client:
        tasks = [
            asyncio.create_task(write_worker(client, dim, namespace, stop, latencies, errors, batch_size))
            for _ in range(concurrency)
        ]
        await asyncio.sleep(duration)
        stop.set()
        await asyncio.gather(*tasks, return_exceptions=True)

    total_vectors = len(latencies) * batch_size
    result = compute_result("write", latencies, len(errors), duration)
    result.extra["total_vectors_written"] = f"{total_vectors:,}"
    result.extra["vectors_per_second"] = f"{total_vectors / duration:,.0f}"
    return result


async def bench_query(
    base_url: str, dim: int, namespace: str, num_queries: int, concurrency: int, top_k: int,
) -> BenchmarkResult:
    """Query latency profiling with fixed query count."""
    print(f"  [query] {num_queries:,} queries, concurrency={concurrency}, top_k={top_k}...")
    latencies, errors = [], []
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(base_url=base_url, timeout=30) as client:
        t0 = time.perf_counter()
        await fixed_query_burst(client, dim, namespace, num_queries, sem, latencies, errors, top_k)
        wall = time.perf_counter() - t0

    return compute_result("query", latencies, len(errors), wall)


async def bench_mixed(
    base_url: str, dim: int, namespace: str, duration: int, concurrency: int, write_ratio: float,
) -> BenchmarkResult:
    """Mixed read/write workload."""
    num_writers = max(1, int(concurrency * write_ratio))
    num_readers = concurrency - num_writers
    print(f"  [mixed] {num_writers} writers + {num_readers} readers, {duration}s...")

    stop = asyncio.Event()
    write_lat, read_lat, errors = [], [], []

    async with httpx.AsyncClient(base_url=base_url, timeout=30) as client:
        tasks = []
        for _ in range(num_writers):
            tasks.append(asyncio.create_task(
                write_worker(client, dim, namespace, stop, write_lat, errors)
            ))
        for _ in range(num_readers):
            tasks.append(asyncio.create_task(
                query_worker(client, dim, namespace, stop, read_lat, errors)
            ))
        await asyncio.sleep(duration)
        stop.set()
        await asyncio.gather(*tasks, return_exceptions=True)

    all_lat = write_lat + read_lat
    result = compute_result("mixed", all_lat, len(errors), duration)
    result.extra["write_ops"] = f"{len(write_lat):,}"
    result.extra["read_ops"] = f"{len(read_lat):,}"
    if write_lat:
        result.extra["write_p50_ms"] = f"{float(np.percentile([t*1000 for t in write_lat], 50)):.2f}"
    if read_lat:
        result.extra["read_p50_ms"] = f"{float(np.percentile([t*1000 for t in read_lat], 50)):.2f}"
    return result


async def bench_full(base_url: str, dim: int, namespace: str, duration: int) -> list[BenchmarkResult]:
    """Full production validation suite."""
    results = []

    # 1. Smoke
    results.append(await bench_smoke(base_url, dim, namespace))

    # 2. Write throughput
    results.append(await bench_write(base_url, dim, namespace, duration // 3, 64, 50))

    # 3. Query latency
    results.append(await bench_query(base_url, dim, namespace, 50_000, 128, 10))

    # 4. Mixed
    results.append(await bench_mixed(base_url, dim, namespace, duration // 3, 64, 0.3))

    return results


# ── Main ────────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="S3-Vector-Engine Load Benchmark")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--mode", choices=["smoke", "write", "query", "mixed", "full"], default="smoke")
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--namespace", default="benchmark")
    parser.add_argument("--duration", type=int, default=60, help="Duration in seconds (write/mixed/full)")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--num-queries", type=int, default=10_000)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--write-ratio", type=float, default=0.3, help="Write ratio for mixed mode")
    parser.add_argument("--output", type=str, help="Write JSON results to file")
    args = parser.parse_args()

    print("=" * 60)
    print("S3-Vector-Engine — Load Benchmark")
    print("=" * 60)
    print(f"  URL:          {args.base_url}")
    print(f"  Mode:         {args.mode}")
    print(f"  Namespace:    {args.namespace}")
    print(f"  Dimensions:   {args.dim}")
    print()

    if args.mode == "smoke":
        result = await bench_smoke(args.base_url, args.dim, args.namespace)
        results = [result]
    elif args.mode == "write":
        result = await bench_write(
            args.base_url, args.dim, args.namespace, args.duration, args.concurrency, args.batch_size,
        )
        results = [result]
    elif args.mode == "query":
        result = await bench_query(
            args.base_url, args.dim, args.namespace, args.num_queries, args.concurrency, args.top_k,
        )
        results = [result]
    elif args.mode == "mixed":
        result = await bench_mixed(
            args.base_url, args.dim, args.namespace, args.duration, args.concurrency, args.write_ratio,
        )
        results = [result]
    elif args.mode == "full":
        results = await bench_full(args.base_url, args.dim, args.namespace, args.duration)

    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)
    for r in results:
        print(r.summary())
        print("-" * 40)

    # PRD targets
    print()
    print("PRD §13 Target Check:")
    for r in results:
        if r.mode == "query":
            ok_p50 = r.latency_p50_ms <= 7.0
            ok_qps = r.ops_per_second >= 10_000
            print(f"  Query p50 ≤ 7ms:     {'PASS' if ok_p50 else 'FAIL'} ({r.latency_p50_ms:.2f}ms)")
            print(f"  Query ≥ 10K qps:     {'PASS' if ok_qps else 'FAIL'} ({r.ops_per_second:,.0f} qps)")
        if r.mode == "write":
            vps = int(r.extra.get("vectors_per_second", "0").replace(",", ""))
            ok_wps = vps >= 50_000
            print(f"  Write ≥ 50K vec/s:   {'PASS' if ok_wps else 'FAIL'} ({vps:,} vec/s)")

    if args.output:
        with open(args.output, "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
