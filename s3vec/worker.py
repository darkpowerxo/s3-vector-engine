"""Shard worker — the core scan engine.

Each worker:
  1. Fetches one shard from S3 (object storage, not RAM)
  2. Computes cosine similarity between query vector and all shard vectors
  3. Returns local top-k results
  4. Discards the shard from memory

The vectors are TRANSIENT in memory — loaded for the scan, then gone.
No persistent RAM allocation. This is the key insight.

The cosine similarity scan uses a Rust extension (_engine) when available
for SIMD-accelerated, GIL-free computation. Falls back to numpy otherwise.
"""

import time
import logging
from dataclasses import dataclass

import numpy as np

from .storage import fetch_shard

logger = logging.getLogger(__name__)

# ── Try to load Rust shard scanner ──────────────────────────────────────────
try:
    from s3vec._engine import cosine_topk as _rust_cosine_topk
    _USE_RUST = True
    logger.info("Rust shard scanner loaded (_engine.cosine_topk)")
except ImportError:
    _USE_RUST = False
    logger.info("Rust extension not available — using numpy fallback")


@dataclass
class ShardResult:
    """Results from scanning a single shard."""
    shard_key: str
    ids: list[str]
    scores: list[float]
    vectors_scanned: int
    fetch_ms: float
    scan_ms: float
    total_ms: float


async def scan_shard(
    shard_key: str,
    query_vector: np.ndarray,
    top_k: int = 10,
) -> ShardResult:
    """Fetch shard from S3, scan for top-k similar vectors, discard shard.

    This is the hot path. Performance breakdown:
      - S3 fetch: ~1-5ms (MinIO local network)
      - Cosine similarity (numpy vectorized): ~0.1-1ms for 10K vectors
      - Total per shard: ~2-6ms
    """
    t0 = time.perf_counter()

    # ── Step 1: Fetch shard from S3 ──
    vectors, ids = await fetch_shard(shard_key)
    t_fetch = time.perf_counter()

    # ── Step 2: Cosine similarity + top-k ──
    k = min(top_k, len(ids))

    if _USE_RUST:
        # Rust path: SIMD-accelerated, GIL-free
        indices, scores = _rust_cosine_topk(
            query_vector.astype(np.float32),
            vectors.astype(np.float32),
            k,
        )
        top_indices = indices.astype(int)
        result_scores = [float(s) for s in scores]
        result_ids = [ids[i] for i in top_indices]
    else:
        # Numpy fallback
        query_norm = query_vector / (np.linalg.norm(query_vector) + 1e-10)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        vectors_norm = vectors / norms
        similarities = vectors_norm @ query_norm

        if k < len(similarities):
            top_indices = np.argpartition(similarities, -k)[-k:]
            top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
        else:
            top_indices = np.argsort(similarities)[::-1]

        result_ids = [ids[i] for i in top_indices]
        result_scores = [float(similarities[i]) for i in top_indices]

    t_scan = time.perf_counter()

    # ── Step 4: Shard is now out of scope and will be GC'd ──
    # vectors, ids, vectors_norm — all transient, no persistent RAM

    fetch_ms = (t_fetch - t0) * 1000
    scan_ms = (t_scan - t_fetch) * 1000
    total_ms = (t_scan - t0) * 1000

    logger.debug(
        f"Shard {shard_key}: fetch={fetch_ms:.1f}ms scan={scan_ms:.1f}ms "
        f"total={total_ms:.1f}ms ({len(ids)} vectors)"
    )

    return ShardResult(
        shard_key=shard_key,
        ids=result_ids,
        scores=result_scores,
        vectors_scanned=len(ids),
        fetch_ms=fetch_ms,
        scan_ms=scan_ms,
        total_ms=total_ms,
    )
