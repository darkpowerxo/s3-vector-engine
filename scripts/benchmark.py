#!/usr/bin/env python3
"""Benchmark the S3-native vector engine.

Runs N random queries against the engine and reports latency percentiles.
Proves sub-10ms is achievable on object storage.

Usage:
    python scripts/benchmark.py --num-queries 1000 --top-k 10
    python scripts/benchmark.py --num-queries 100 --top-k 10 --include-timings
"""

import argparse
import asyncio
import time
import sys
import os
import statistics

import numpy as np
import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def run_benchmark(
    base_url: str,
    num_queries: int,
    dimensions: int,
    top_k: int,
    tenant_id: str,
    concurrency: int,
    include_timings: bool,
):
    """Run the benchmark."""
    print("=" * 60)
    print("S3-Native Vector Engine — Benchmark")
    print("=" * 60)

    # First, check stats
    async with httpx.AsyncClient(base_url=base_url, timeout=30) as client:
        stats = (await client.get("/stats")).json()
        print(f"  Endpoint:     {base_url}")
        print(f"  Tenant:       {tenant_id}")
        print(f"  Tenants:      {stats.get('tenants', '?')}")
        print(f"  Shards:       {stats.get('total_shards', '?')}")
        print(f"  Vectors:      {stats.get('total_vectors', '?'):,}")
        print(f"  Queries:      {num_queries}")
        print(f"  Top-k:        {top_k}")
        print(f"  Concurrency:  {concurrency}")
        print()

    # Generate random query vectors
    print("Generating query vectors...")
    query_vectors = np.random.randn(num_queries, dimensions).astype(np.float32)
    norms = np.linalg.norm(query_vectors, axis=1, keepdims=True)
    query_vectors = query_vectors / norms

    latencies = []
    errors = 0
    shard_fetch_times = []
    shard_scan_times = []

    semaphore = asyncio.Semaphore(concurrency)

    async def _query(idx: int, client: httpx.AsyncClient):
        nonlocal errors
        async with semaphore:
            payload = {
                "vector": query_vectors[idx].tolist(),
                "top_k": top_k,
                "tenant_id": tenant_id,
                "include_timings": include_timings,
            }
            try:
                t0 = time.perf_counter()
                resp = await client.post("/search", json=payload)
                wall_ms = (time.perf_counter() - t0) * 1000

                if resp.status_code == 200:
                    data = resp.json()
                    # Use server-reported latency (more accurate)
                    server_ms = data.get("latency_ms", wall_ms)
                    latencies.append(server_ms)

                    if include_timings and data.get("shard_timings"):
                        for st in data["shard_timings"]:
                            shard_fetch_times.append(st["fetch_ms"])
                            shard_scan_times.append(st["scan_ms"])
                else:
                    errors += 1
            except Exception as e:
                errors += 1

    # Run queries
    print(f"Running {num_queries} queries...")
    t_start = time.perf_counter()

    async with httpx.AsyncClient(base_url=base_url, timeout=30) as client:
        tasks = [_query(i, client) for i in range(num_queries)]
        await asyncio.gather(*tasks)

    t_total = time.perf_counter() - t_start

    # Report
    if not latencies:
        print("No successful queries!")
        return

    latencies.sort()
    n = len(latencies)

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Successful queries: {n}/{num_queries}")
    print(f"  Errors:             {errors}")
    print(f"  Wall time:          {t_total:.2f}s")
    print(f"  Throughput:         {n / t_total:.1f} qps")
    print()
    print("  Latency (server-reported):")
    print(f"    min:  {latencies[0]:.2f} ms")
    print(f"    p25:  {latencies[int(n * 0.25)]:.2f} ms")
    print(f"    p50:  {latencies[int(n * 0.50)]:.2f} ms")
    print(f"    p75:  {latencies[int(n * 0.75)]:.2f} ms")
    print(f"    p90:  {latencies[int(n * 0.90)]:.2f} ms")
    print(f"    p95:  {latencies[int(n * 0.95)]:.2f} ms")
    print(f"    p99:  {latencies[int(n * 0.99)]:.2f} ms")
    print(f"    max:  {latencies[-1]:.2f} ms")
    print(f"    avg:  {statistics.mean(latencies):.2f} ms")
    print(f"    std:  {statistics.stdev(latencies):.2f} ms" if n > 1 else "")

    if include_timings and shard_fetch_times:
        shard_fetch_times.sort()
        shard_scan_times.sort()
        fn = len(shard_fetch_times)
        print()
        print("  Per-shard breakdown:")
        print(f"    S3 fetch p50: {shard_fetch_times[fn//2]:.2f} ms")
        print(f"    S3 fetch p99: {shard_fetch_times[int(fn*0.99)]:.2f} ms")
        print(f"    Scan p50:     {shard_scan_times[fn//2]:.2f} ms")
        print(f"    Scan p99:     {shard_scan_times[int(fn*0.99)]:.2f} ms")

    # Sub-10ms check
    p99 = latencies[int(n * 0.99)]
    print()
    if p99 < 10:
        print(f"  ✅ SUB-10MS ACHIEVED — p99 = {p99:.2f}ms")
    else:
        print(f"  ⚠️  p99 = {p99:.2f}ms (above 10ms target)")
        print(f"     p50 = {latencies[n//2]:.2f}ms")
        if shard_fetch_times:
            print(f"     Bottleneck: S3 fetch p99 = {shard_fetch_times[int(fn*0.99)]:.2f}ms")
            print(f"     Consider: more shards (smaller each), NVMe-backed MinIO, 10GbE")

    print()
    print("  --- Cost at this scale ---")
    vecs = stats.get("total_vectors", 0)
    if vecs > 0:
        s3_annual = (vecs * 1024 * 4) / (1024**3) * 0.023 * 12
        managed_annual = (vecs / 1_000_000) * 7 * 12
        print(f"  S3 storage (annual):      ${s3_annual:,.0f}")
        print(f"  Managed VDB (est annual): ${managed_annual:,.0f}")
        print(f"  Savings:                  ${managed_annual - s3_annual:,.0f}/yr")


def main():
    parser = argparse.ArgumentParser(description="Benchmark S3-native vector engine")
    parser.add_argument("--url", default="http://localhost:8000", help="Engine URL")
    parser.add_argument("--num-queries", type=int, default=100)
    parser.add_argument("--dimensions", type=int, default=1024)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--tenant-id", default="default")
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--include-timings", action="store_true")
    args = parser.parse_args()

    asyncio.run(run_benchmark(
        base_url=args.url,
        num_queries=args.num_queries,
        dimensions=args.dimensions,
        top_k=args.top_k,
        tenant_id=args.tenant_id,
        concurrency=args.concurrency,
        include_timings=args.include_timings,
    ))


if __name__ == "__main__":
    main()
