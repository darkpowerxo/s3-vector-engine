"""Coordinator — the fan-out merge engine.

Receives a query vector, fans out to all shard workers in parallel,
collects local top-k from each, merges into global top-k.

This is the Ray-coordinator equivalent from the Mixpeek architecture,
implemented with pure asyncio for simplicity in this POC.
"""

import time
import asyncio
import logging
from dataclasses import dataclass, field

import numpy as np

from .config import get_settings
from .registry import get_tenant_shards, record_query_latency
from .worker import scan_shard, ShardResult

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Global search result after merging all shard results."""
    results: list[dict]          # [{id, score, shard}, ...]
    latency_ms: float
    shards_scanned: int
    shards_failed: int
    total_vectors_scanned: int
    shard_timings: list[dict]    # per-shard timing breakdown
    errors: list[dict] = field(default_factory=list)


async def search(
    query_vector: np.ndarray,
    tenant_id: str,
    top_k: int | None = None,
) -> SearchResult:
    """Execute a distributed vector search across all tenant shards.

    Flow:
      1. Look up tenant's shard keys from Redis registry
      2. Fan out async tasks — one per shard, bounded by semaphore
      3. Each task: fetch shard from S3 → scan → return local top-k
      4. Merge all local results → global top-k
      5. Record latency metric in Redis

    Error tolerance: individual shard failures are logged but don't
    fail the whole query. Partial results are returned.
    """
    settings = get_settings()
    if top_k is None:
        top_k = settings.top_k_default

    t0 = time.perf_counter()

    # ── Step 1: Get shard list from Redis ──
    shard_keys = await get_tenant_shards(tenant_id)
    if not shard_keys:
        return SearchResult(
            results=[],
            latency_ms=0,
            shards_scanned=0,
            shards_failed=0,
            total_vectors_scanned=0,
            shard_timings=[],
        )

    # ── Step 2: Fan out to workers with concurrency limit ──
    semaphore = asyncio.Semaphore(settings.max_concurrent_shards)

    async def _bounded_scan(shard_key: str) -> ShardResult | Exception:
        async with semaphore:
            try:
                return await asyncio.wait_for(
                    scan_shard(shard_key, query_vector, top_k),
                    timeout=settings.shard_timeout_seconds,
                )
            except Exception as exc:
                logger.error("shard_scan_failed", extra={
                    "shard": shard_key, "error": str(exc),
                })
                return exc

    tasks = [_bounded_scan(sk) for sk in shard_keys]
    raw_results = await asyncio.gather(*tasks)

    # ── Step 3: Merge results (skip failed shards) ──
    all_candidates = []
    total_vectors = 0
    shard_timings = []
    errors = []
    shards_ok = 0

    for sk, result in zip(shard_keys, raw_results):
        if isinstance(result, Exception):
            errors.append({"shard": sk, "error": str(result)})
            continue
        sr: ShardResult = result
        shards_ok += 1
        total_vectors += sr.vectors_scanned
        shard_timings.append({
            "shard": sr.shard_key,
            "fetch_ms": round(sr.fetch_ms, 2),
            "scan_ms": round(sr.scan_ms, 2),
            "total_ms": round(sr.total_ms, 2),
            "vectors": sr.vectors_scanned,
        })
        for vid, score in zip(sr.ids, sr.scores):
            all_candidates.append({
                "id": vid,
                "score": round(score, 6),
                "shard": sr.shard_key,
            })

    # Sort by score descending, take global top-k
    all_candidates.sort(key=lambda x: x["score"], reverse=True)
    global_top_k = all_candidates[:top_k]

    t_end = time.perf_counter()
    latency_ms = (t_end - t0) * 1000

    # ── Step 4: Record metrics ──
    await record_query_latency(tenant_id, latency_ms)

    logger.info(
        f"Search complete: {shards_ok}/{len(shard_keys)} shards, "
        f"{total_vectors} vectors, {latency_ms:.1f}ms"
    )

    return SearchResult(
        results=global_top_k,
        latency_ms=round(latency_ms, 2),
        shards_scanned=shards_ok,
        shards_failed=len(errors),
        total_vectors_scanned=total_vectors,
        shard_timings=shard_timings,
        errors=errors,
    )
