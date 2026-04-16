"""Redis shard registry.

Redis stores METADATA ONLY — shard keys, vector counts, tenant mappings.
Vectors live on S3. Redis is the index of shards, not the store of vectors.
"""

import json
import time
import logging
from typing import Optional

import redis.asyncio as aioredis

from .config import get_settings

logger = logging.getLogger(__name__)

_pool: Optional[aioredis.ConnectionPool] = None

# Lightweight in-process stats cache to avoid scanning all shards
# on every /stats call. Invalidated on shard register/unregister.
_stats_cache: Optional[dict] = None
_stats_cache_ts: float = 0
_STATS_TTL = 5.0  # seconds


async def get_redis() -> aioredis.Redis:
    """Get async Redis connection from pool."""
    global _pool
    if _pool is None:
        settings = get_settings()
        _pool = aioredis.ConnectionPool.from_url(
            settings.redis_url, max_connections=20, decode_responses=True
        )
    return aioredis.Redis(connection_pool=_pool)


# ── Shard Registry ──────────────────────────────────────────────────────────

async def register_shard(
    tenant_id: str,
    shard_key: str,
    vector_count: int,
    dimensions: int,
):
    """Register a shard in the registry."""
    global _stats_cache
    r = await get_redis()
    shard_meta = json.dumps({
        "shard_key": shard_key,
        "vector_count": vector_count,
        "dimensions": dimensions,
    })
    pipe = r.pipeline()
    pipe.sadd(f"tenant:{tenant_id}:shards", shard_key)
    pipe.set(f"shard:{shard_key}:meta", shard_meta)
    await pipe.execute()
    _stats_cache = None  # invalidate
    logger.info(
        f"Registered shard '{shard_key}' for tenant '{tenant_id}' "
        f"({vector_count} vectors, {dimensions}d)"
    )


async def unregister_shard(tenant_id: str, shard_key: str):
    """Remove a shard from the registry."""
    global _stats_cache
    r = await get_redis()
    pipe = r.pipeline()
    pipe.srem(f"tenant:{tenant_id}:shards", shard_key)
    pipe.delete(f"shard:{shard_key}:meta")
    await pipe.execute()
    _stats_cache = None  # invalidate


async def get_tenant_shards(tenant_id: str) -> list[str]:
    """Get all shard keys for a tenant."""
    r = await get_redis()
    shard_keys = await r.smembers(f"tenant:{tenant_id}:shards")
    return sorted(shard_keys) if shard_keys else []


async def get_all_tenants() -> list[str]:
    """List all tenants."""
    r = await get_redis()
    keys = []
    async for key in r.scan_iter(match="tenant:*:shards"):
        tenant_id = key.split(":")[1]
        keys.append(tenant_id)
    return sorted(set(keys))


async def get_shard_meta(shard_key: str) -> Optional[dict]:
    """Get metadata for a shard."""
    r = await get_redis()
    raw = await r.get(f"shard:{shard_key}:meta")
    return json.loads(raw) if raw else None


async def get_stats() -> dict:
    """Get global engine stats from Redis (cached for _STATS_TTL seconds)."""
    global _stats_cache, _stats_cache_ts
    now = time.monotonic()
    if _stats_cache is not None and (now - _stats_cache_ts) < _STATS_TTL:
        return _stats_cache

    r = await get_redis()
    tenants = await get_all_tenants()
    total_shards = 0
    total_vectors = 0

    # Batch fetch all shard metadata with pipeline
    for tenant_id in tenants:
        shard_keys = await get_tenant_shards(tenant_id)
        total_shards += len(shard_keys)
        if shard_keys:
            pipe = r.pipeline()
            for sk in shard_keys:
                pipe.get(f"shard:{sk}:meta")
            results = await pipe.execute()
            for raw in results:
                if raw:
                    meta = json.loads(raw)
                    total_vectors += meta.get("vector_count", 0)

    _stats_cache = {
        "tenants": len(tenants),
        "total_shards": total_shards,
        "total_vectors": total_vectors,
        "tenant_list": tenants,
    }
    _stats_cache_ts = now
    return _stats_cache


# ── Query Metrics ───────────────────────────────────────────────────────────

async def record_query_latency(tenant_id: str, latency_ms: float):
    """Record query latency for monitoring."""
    r = await get_redis()
    pipe = r.pipeline()
    pipe.incr("metrics:total_queries")
    pipe.incr(f"metrics:tenant:{tenant_id}:queries")
    pipe.lpush("metrics:latencies", f"{latency_ms:.2f}")
    pipe.ltrim("metrics:latencies", 0, 9999)  # keep last 10K
    await pipe.execute()


async def get_metrics() -> dict:
    """Get query metrics."""
    r = await get_redis()
    pipe = r.pipeline()
    pipe.get("metrics:total_queries")
    pipe.lrange("metrics:latencies", 0, 999)
    total, latencies_raw = await pipe.execute()

    latencies = [float(x) for x in latencies_raw] if latencies_raw else []

    return {
        "total_queries": int(total) if total else 0,
        "recent_latencies_count": len(latencies),
        "p50_ms": sorted(latencies)[len(latencies) // 2] if latencies else 0,
        "p99_ms": sorted(latencies)[int(len(latencies) * 0.99)] if latencies else 0,
        "avg_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0,
    }


async def close():
    """Close Redis connection pool."""
    global _pool
    if _pool:
        await _pool.disconnect()
        _pool = None
