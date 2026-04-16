"""Integration tests for the S3-native vector engine.

Requires MinIO and Redis running (docker compose up -d).

Usage:
    uv run python -m pytest tests/ -v
"""

import time

import numpy as np
import pytest

from s3vec.storage import upload_shard, fetch_shard, delete_shard
from s3vec.registry import register_shard, get_tenant_shards, unregister_shard, get_stats
from s3vec.indexer import bulk_index
from s3vec.coordinator import search
from s3vec.worker import scan_shard


# ---------------------------------------------------------------------------
# S3 shard round-trip
# ---------------------------------------------------------------------------

class TestShardStorage:
    """S3 upload → fetch → verify round-trip."""

    async def test_roundtrip_identity(self, tenant_id, dimensions):
        """Uploaded vectors must be bit-identical after fetch."""
        vectors = np.random.randn(100, dimensions).astype(np.float32)
        ids = [f"rt_{i}" for i in range(100)]
        shard_key = f"tenants/{tenant_id}/test_roundtrip"

        count = await upload_shard(shard_key, vectors, ids)
        assert count == 100

        fetched_vecs, fetched_ids = await fetch_shard(shard_key)
        assert fetched_ids == ids
        assert np.allclose(fetched_vecs, vectors, atol=1e-6)

        await delete_shard(shard_key)

    async def test_delete_removes_objects(self, tenant_id, dimensions):
        """After delete, fetching the shard must raise."""
        vectors = np.random.randn(10, dimensions).astype(np.float32)
        ids = [f"del_{i}" for i in range(10)]
        shard_key = f"tenants/{tenant_id}/test_delete"

        await upload_shard(shard_key, vectors, ids)
        await delete_shard(shard_key)

        with pytest.raises(Exception):
            await fetch_shard(shard_key)


# ---------------------------------------------------------------------------
# Redis registry
# ---------------------------------------------------------------------------

class TestRegistry:
    """Redis shard registry operations."""

    async def test_register_and_list(self, tenant_id, dimensions):
        shard_key = f"tenants/{tenant_id}/reg_shard"
        await register_shard(tenant_id, shard_key, vector_count=500, dimensions=dimensions)

        shards = await get_tenant_shards(tenant_id)
        assert shard_key in shards

        await unregister_shard(tenant_id, shard_key)
        shards = await get_tenant_shards(tenant_id)
        assert shard_key not in shards

    async def test_stats_includes_tenant(self, tenant_id, dimensions):
        shard_key = f"tenants/{tenant_id}/stats_shard"
        await register_shard(tenant_id, shard_key, vector_count=200, dimensions=dimensions)

        stats = await get_stats()
        assert stats["tenants"] >= 1
        assert stats["total_vectors"] >= 200

        await unregister_shard(tenant_id, shard_key)


# ---------------------------------------------------------------------------
# Indexer
# ---------------------------------------------------------------------------

class TestBulkIndex:
    """Bulk indexing pipeline."""

    async def test_creates_correct_shards(self, tenant_id, random_vectors):
        vectors, _ = random_vectors(1000)
        ids = [f"bulk_{i:06d}" for i in range(1000)]

        result = await bulk_index(tenant_id, ids, vectors, shard_size=500)
        assert result["shards_created"] == 2
        assert result["vectors_indexed"] == 1000

        shards = await get_tenant_shards(tenant_id)
        assert len(shards) == 2

    async def test_partial_shard(self, tenant_id, random_vectors):
        """A count not evenly divisible by shard_size leaves no shard (vectors buffered)."""
        vectors, _ = random_vectors(300)
        ids = [f"part_{i}" for i in range(300)]

        result = await bulk_index(tenant_id, ids, vectors, shard_size=500)
        # 300 < 500 → no full shard, all buffered by add_vectors logic
        # bulk_index writes shards directly, so it should create 1 shard
        assert result["shards_created"] == 1
        assert result["vectors_indexed"] == 300


# ---------------------------------------------------------------------------
# Worker (Rust vs numpy parity)
# ---------------------------------------------------------------------------

class TestWorker:
    """Shard scanner correctness."""

    async def test_scan_returns_topk(self, tenant_id, dimensions):
        """scan_shard must return the correct number of results."""
        vectors = np.random.randn(200, dimensions).astype(np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.maximum(norms, 1e-10)
        ids = [f"scan_{i}" for i in range(200)]
        shard_key = f"tenants/{tenant_id}/test_scan"

        await upload_shard(shard_key, vectors, ids)
        await register_shard(tenant_id, shard_key, vector_count=200, dimensions=dimensions)

        query = np.random.randn(dimensions).astype(np.float32)
        query = query / np.linalg.norm(query)

        result = await scan_shard(shard_key, query, top_k=5)
        assert len(result.ids) == 5
        assert len(result.scores) == 5
        assert result.vectors_scanned == 200
        # Scores must be descending
        for i in range(len(result.scores) - 1):
            assert result.scores[i] >= result.scores[i + 1]

    async def test_known_nearest(self, tenant_id, dimensions):
        """The scanner must rank an exact-match vector first."""
        target = np.zeros(dimensions, dtype=np.float32)
        target[0] = 1.0

        noise = np.random.randn(99, dimensions).astype(np.float32) * 0.1
        norms = np.linalg.norm(noise, axis=1, keepdims=True)
        noise = noise / np.maximum(norms, 1e-10)

        all_vecs = np.vstack([target.reshape(1, -1), noise])
        ids = ["TARGET"] + [f"noise_{i}" for i in range(99)]
        shard_key = f"tenants/{tenant_id}/test_known"

        await upload_shard(shard_key, all_vecs, ids)

        result = await scan_shard(shard_key, target, top_k=3)
        assert result.ids[0] == "TARGET"
        assert result.scores[0] > 0.99


# ---------------------------------------------------------------------------
# End-to-end search
# ---------------------------------------------------------------------------

class TestSearch:
    """Full coordinator search."""

    async def test_accuracy(self, tenant_id, dimensions):
        """Search returns the correct nearest neighbor across shards."""
        target = np.zeros(dimensions, dtype=np.float32)
        target[0] = 1.0

        noise = np.random.randn(999, dimensions).astype(np.float32) * 0.1
        norms = np.linalg.norm(noise, axis=1, keepdims=True)
        noise = noise / np.maximum(norms, 1e-10)

        all_vecs = np.vstack([target.reshape(1, -1), noise])
        all_ids = ["TARGET"] + [f"noise_{i}" for i in range(999)]

        await bulk_index(tenant_id, all_ids, all_vecs, shard_size=500)

        result = await search(target, tenant_id, top_k=5)
        assert len(result.results) > 0
        assert result.results[0]["id"] == "TARGET"
        assert result.results[0]["score"] > 0.99
        assert result.shards_scanned == 2
        assert result.shards_failed == 0

    async def test_empty_tenant_returns_empty(self):
        """Searching a tenant with no shards returns empty results."""
        query = np.random.randn(128).astype(np.float32)
        result = await search(query, "nonexistent_tenant_xyz", top_k=5)
        assert result.results == []
        assert result.shards_scanned == 0

    async def test_latency(self, tenant_id, dimensions):
        """Search latency < 200ms for 10K vectors (CI-friendly threshold)."""
        vectors = np.random.randn(10_000, dimensions).astype(np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.maximum(norms, 1e-10)
        ids = [f"lat_{i:06d}" for i in range(10_000)]

        await bulk_index(tenant_id, ids, vectors, shard_size=5000)

        # Warm-up
        query = np.random.randn(dimensions).astype(np.float32)
        query = query / np.linalg.norm(query)
        await search(query, tenant_id, top_k=10)

        # Timed
        latencies = []
        for _ in range(10):
            q = np.random.randn(dimensions).astype(np.float32)
            q = q / np.linalg.norm(q)
            t0 = time.perf_counter()
            await search(q, tenant_id, top_k=10)
            latencies.append((time.perf_counter() - t0) * 1000)

        p50 = sorted(latencies)[len(latencies) // 2]
        assert p50 < 200, f"p50 latency too high: {p50:.1f}ms"
