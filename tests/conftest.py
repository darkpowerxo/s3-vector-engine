"""Shared pytest fixtures for integration tests.

Requires MinIO and Redis running (docker compose up -d).
"""

import asyncio

import numpy as np
import pytest

from s3vec.config import get_settings
from s3vec.storage import ensure_bucket, delete_shard, close_s3, list_shard_keys
from s3vec.registry import (
    get_tenant_shards,
    unregister_shard,
    close as close_redis,
)

# Smaller dims for fast tests
TEST_DIM = 128
TEST_TENANTS: list[str] = []


@pytest.fixture(scope="session")
def settings():
    return get_settings()


@pytest.fixture(scope="session")
def dimensions():
    return TEST_DIM


@pytest.fixture(scope="session", autouse=True)
def _ensure_bucket():
    """Create the S3 bucket once per session."""
    ensure_bucket()


@pytest.fixture()
def random_vectors():
    """Factory fixture — returns normalised random vectors."""

    def _make(n: int, dim: int = TEST_DIM) -> tuple[np.ndarray, list[str]]:
        vecs = np.random.randn(n, dim).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / np.maximum(norms, 1e-10)
        ids = [f"vec_{i:06d}" for i in range(n)]
        return vecs, ids

    return _make


async def _cleanup_tenant(tenant_id: str):
    """Remove all shards for a tenant from S3 + Redis."""
    shard_keys = await get_tenant_shards(tenant_id)
    for sk in shard_keys:
        try:
            await delete_shard(sk)
        except Exception:
            pass
        try:
            await unregister_shard(tenant_id, sk)
        except Exception:
            pass


@pytest.fixture()
def tenant_id():
    """Provide a unique, auto-cleaned tenant for each test."""
    import uuid

    tid = f"test_{uuid.uuid4().hex[:8]}"
    TEST_TENANTS.append(tid)
    return tid


@pytest.fixture(scope="session", autouse=True)
async def _session_cleanup():
    """Clean up all test tenants and close connections after the session."""
    yield
    for tid in TEST_TENANTS:
        try:
            await _cleanup_tenant(tid)
        except Exception:
            pass
    await close_s3()
    await close_redis()
