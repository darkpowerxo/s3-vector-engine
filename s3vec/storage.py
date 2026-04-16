"""S3/MinIO storage client for shard operations.

Vectors live HERE — on object storage — not in RAM.
This module handles all interactions with MinIO/S3.

Connection reuse: We maintain one async client context per event loop
to avoid the overhead of creating a new TLS/TCP connection per call.
"""

import io
import asyncio
import logging
from contextlib import asynccontextmanager

import aioboto3
import boto3
import numpy as np
from botocore.config import Config as BotoConfig

from .config import get_settings

logger = logging.getLogger(__name__)

# Reusable session for async operations
_session = aioboto3.Session()

# Per-loop connection reuse
_client_ctx = None
_client = None
_client_lock = asyncio.Lock() if hasattr(asyncio, "Lock") else None


def _boto_config() -> BotoConfig:
    """Boto config tuned for low-latency local MinIO."""
    return BotoConfig(
        connect_timeout=2,
        read_timeout=5,
        max_pool_connections=50,
        retries={"max_attempts": 2, "mode": "standard"},
    )


async def _get_async_client():
    """Get or create a reusable async S3 client."""
    global _client_ctx, _client, _client_lock
    if _client_lock is None:
        _client_lock = asyncio.Lock()
    async with _client_lock:
        if _client is not None:
            return _client
        settings = get_settings()
        _client_ctx = _session.client(
            "s3",
            endpoint_url=settings.s3_endpoint,
            aws_access_key_id=settings.s3_access_key,
            aws_secret_access_key=settings.s3_secret_key,
            region_name=settings.s3_region,
            config=_boto_config(),
        )
        _client = await _client_ctx.__aenter__()
        return _client


async def close_s3():
    """Close the reusable async S3 client."""
    global _client_ctx, _client
    if _client_ctx is not None:
        await _client_ctx.__aexit__(None, None, None)
        _client_ctx = None
        _client = None


def get_sync_client():
    """Synchronous S3 client for setup / seeding operations."""
    settings = get_settings()
    return boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint,
        aws_access_key_id=settings.s3_access_key,
        aws_secret_access_key=settings.s3_secret_key,
        region_name=settings.s3_region,
        config=_boto_config(),
    )


def ensure_bucket():
    """Create the vector store bucket if it doesn't exist."""
    settings = get_settings()
    client = get_sync_client()
    try:
        client.head_bucket(Bucket=settings.s3_bucket)
        logger.info(f"Bucket '{settings.s3_bucket}' already exists")
    except client.exceptions.ClientError:
        client.create_bucket(Bucket=settings.s3_bucket)
        logger.info(f"Created bucket '{settings.s3_bucket}'")


async def upload_shard(
    shard_key: str,
    vectors: np.ndarray,
    ids: list[str],
) -> int:
    """Upload a shard (vectors + ids) to S3.

    Shard format on S3:
      {shard_key}.npy  — numpy array of shape (N, D), dtype float32
      {shard_key}.ids  — newline-delimited vector IDs

    Returns the number of vectors stored.
    """
    settings = get_settings()
    client = await _get_async_client()

    # Serialize vectors to bytes
    vec_buf = io.BytesIO()
    np.save(vec_buf, vectors.astype(np.float32))
    vec_bytes = vec_buf.getvalue()

    # Serialize IDs
    ids_bytes = "\n".join(ids).encode("utf-8")

    await asyncio.gather(
        client.put_object(
            Bucket=settings.s3_bucket,
            Key=f"{shard_key}.npy",
            Body=vec_bytes,
        ),
        client.put_object(
            Bucket=settings.s3_bucket,
            Key=f"{shard_key}.ids",
            Body=ids_bytes,
        ),
    )

    return len(ids)


async def fetch_shard(shard_key: str) -> tuple[np.ndarray, list[str]]:
    """Fetch a shard from S3. Returns (vectors, ids).

    This is the hot path. Vectors are fetched from object storage,
    scanned in transient memory, and then discarded.
    They do NOT persist in RAM.
    """
    settings = get_settings()
    client = await _get_async_client()

    # Parallel fetch of vectors and IDs
    vec_resp, ids_resp = await asyncio.gather(
        client.get_object(Bucket=settings.s3_bucket, Key=f"{shard_key}.npy"),
        client.get_object(Bucket=settings.s3_bucket, Key=f"{shard_key}.ids"),
    )

    vec_bytes = await vec_resp["Body"].read()
    ids_bytes = await ids_resp["Body"].read()

    # Deserialize
    vectors = np.load(io.BytesIO(vec_bytes))
    ids = ids_bytes.decode("utf-8").split("\n")

    return vectors, ids


async def delete_shard(shard_key: str):
    """Delete a shard from S3."""
    settings = get_settings()
    client = await _get_async_client()

    await asyncio.gather(
        client.delete_object(
            Bucket=settings.s3_bucket, Key=f"{shard_key}.npy"
        ),
        client.delete_object(
            Bucket=settings.s3_bucket, Key=f"{shard_key}.ids"
        ),
    )


async def list_shard_keys(prefix: str = "") -> list[str]:
    """List all shard keys under a prefix in S3."""
    settings = get_settings()
    client = await _get_async_client()
    shard_keys = set()

    paginator = client.get_paginator("list_objects_v2")
    async for page in paginator.paginate(
        Bucket=settings.s3_bucket, Prefix=prefix
    ):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".npy"):
                shard_keys.add(key[:-4])

    return sorted(shard_keys)
