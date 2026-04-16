"""Indexer — builds shards from vectors and uploads to S3.

When new vectors arrive (e.g., new translation memory segments),
the indexer:
  1. Buffers them until a full shard is ready
  2. Serializes the shard as numpy binary
  3. Uploads to S3 under the tenant's namespace
  4. Registers the shard in Redis
"""

import asyncio
import logging
import math
from typing import Optional

import numpy as np

from .config import get_settings
from .storage import upload_shard
from .registry import register_shard, get_tenant_shards

logger = logging.getLogger(__name__)


# In-memory buffer for vectors not yet committed to a shard.
# In production, this would be a Redis list or persistent queue.
_buffers: dict[str, dict] = {}


def _get_buffer(tenant_id: str) -> dict:
    if tenant_id not in _buffers:
        _buffers[tenant_id] = {"ids": [], "vectors": []}
    return _buffers[tenant_id]


async def add_vectors(
    tenant_id: str,
    ids: list[str],
    vectors: list[list[float]],
) -> dict:
    """Add vectors to the index.

    Vectors are buffered and flushed to S3 when a full shard is ready.
    Returns status with counts.
    """
    settings = get_settings()
    buf = _get_buffer(tenant_id)

    buf["ids"].extend(ids)
    buf["vectors"].extend(vectors)

    shards_created = 0
    vectors_indexed = 0

    # Flush full shards
    while len(buf["ids"]) >= settings.shard_size:
        shard_ids = buf["ids"][: settings.shard_size]
        shard_vecs = buf["vectors"][: settings.shard_size]
        buf["ids"] = buf["ids"][settings.shard_size :]
        buf["vectors"] = buf["vectors"][settings.shard_size :]

        # Determine shard number
        existing = await get_tenant_shards(tenant_id)
        shard_num = len(existing)
        shard_key = f"tenants/{tenant_id}/shard_{shard_num:06d}"

        # Upload to S3
        vec_array = np.array(shard_vecs, dtype=np.float32)
        count = await upload_shard(shard_key, vec_array, shard_ids)

        # Register in Redis
        await register_shard(
            tenant_id=tenant_id,
            shard_key=shard_key,
            vector_count=count,
            dimensions=settings.vector_dimensions,
        )

        shards_created += 1
        vectors_indexed += count
        logger.info(
            f"Created shard '{shard_key}' for tenant '{tenant_id}' "
            f"({count} vectors)"
        )

    return {
        "shards_created": shards_created,
        "vectors_indexed": vectors_indexed,
        "vectors_buffered": len(buf["ids"]),
    }


async def flush_buffer(tenant_id: str) -> dict:
    """Force-flush any buffered vectors as a partial shard."""
    settings = get_settings()
    buf = _get_buffer(tenant_id)

    if not buf["ids"]:
        return {"shards_created": 0, "vectors_indexed": 0}

    existing = await get_tenant_shards(tenant_id)
    shard_num = len(existing)
    shard_key = f"tenants/{tenant_id}/shard_{shard_num:06d}"

    vec_array = np.array(buf["vectors"], dtype=np.float32)
    count = await upload_shard(shard_key, vec_array, buf["ids"])

    await register_shard(
        tenant_id=tenant_id,
        shard_key=shard_key,
        vector_count=count,
        dimensions=settings.vector_dimensions,
    )

    buf["ids"] = []
    buf["vectors"] = []

    logger.info(
        f"Flushed partial shard '{shard_key}' for tenant '{tenant_id}' "
        f"({count} vectors)"
    )

    return {"shards_created": 1, "vectors_indexed": count}


async def bulk_index(
    tenant_id: str,
    ids: list[str],
    vectors: np.ndarray,
    shard_size: Optional[int] = None,
) -> dict:
    """Bulk index a large batch of vectors directly into shards.

    Used by the seed script. Skips the buffer and writes shards directly.
    """
    settings = get_settings()
    if shard_size is None:
        shard_size = settings.shard_size

    num_shards = math.ceil(len(ids) / shard_size)
    total_indexed = 0

    existing = await get_tenant_shards(tenant_id)
    start_num = len(existing)

    tasks = []
    for i in range(num_shards):
        start = i * shard_size
        end = min(start + shard_size, len(ids))
        shard_ids = ids[start:end]
        shard_vecs = vectors[start:end]
        shard_key = f"tenants/{tenant_id}/shard_{start_num + i:06d}"

        tasks.append(_upload_and_register(
            tenant_id, shard_key, shard_vecs, shard_ids
        ))

    results = await asyncio.gather(*tasks)
    total_indexed = sum(results)

    return {
        "shards_created": num_shards,
        "vectors_indexed": total_indexed,
    }


async def _upload_and_register(
    tenant_id: str,
    shard_key: str,
    vectors: np.ndarray,
    ids: list[str],
) -> int:
    """Helper to upload shard and register it."""
    count = await upload_shard(shard_key, vectors, ids)
    await register_shard(
        tenant_id=tenant_id,
        shard_key=shard_key,
        vector_count=count,
        dimensions=vectors.shape[1],
    )
    return count
