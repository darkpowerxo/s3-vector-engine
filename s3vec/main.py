"""S3-Native Vector Engine — FastAPI Application.

Sub-10ms vector search on object storage.
No vectors in RAM. No Pinecone. No Qdrant Cloud.

Architecture:
  FastAPI gateway → Coordinator (async fan-out) → Shard workers → S3/MinIO
  Redis stores shard registry + tenant metadata (NOT vectors)
"""

import logging
import time
import uuid
from contextlib import asynccontextmanager

import numpy as np
import structlog
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel, Field

from .config import get_settings
from .storage import ensure_bucket, delete_shard, close_s3
from .registry import (
    get_stats, get_metrics, get_tenant_shards, get_all_tenants,
    unregister_shard, close as close_redis,
)
from .coordinator import search
from .indexer import add_vectors, flush_buffer


def _configure_logging(settings):
    """Set up structlog with JSON or console output."""
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    if settings.log_json:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(message)s",
    )


logger = structlog.get_logger("s3vec")


# ── Lifespan ────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    settings = get_settings()
    _configure_logging(settings)

    logger.info(
        "starting",
        s3_endpoint=settings.s3_endpoint,
        s3_bucket=settings.s3_bucket,
        redis_url=settings.redis_url,
        shard_size=settings.shard_size,
        dimensions=settings.vector_dimensions,
    )

    try:
        ensure_bucket()
    except Exception as e:
        logger.warning("bucket_check_failed", error=str(e))

    from . import RUST_AVAILABLE
    logger.info("engine_ready", rust_accel=RUST_AVAILABLE)

    yield

    await close_s3()
    await close_redis()
    logger.info("shutdown")


app = FastAPI(
    title="S3-Native Vector Engine",
    description=(
        "Sub-10ms vector search on object storage. "
        "No vectors in RAM. No managed vector DB tax."
    ),
    version="0.1.0",
    lifespan=lifespan,
    default_response_class=ORJSONResponse,
)


# ── Middleware ───────────────────────────────────────────────────────────────

@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """Attach a unique request ID for tracing."""
    request_id = request.headers.get("X-Request-ID", uuid.uuid4().hex[:16])
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(request_id=request_id)

    t0 = time.perf_counter()
    response: Response = await call_next(request)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.2f}"

    logger.info(
        "request",
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        ms=round(elapsed_ms, 2),
    )
    return response


def _setup_cors(app: FastAPI):
    settings = get_settings()
    origins = [o.strip() for o in settings.cors_origins.split(",")]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


_setup_cors(app)


# ── Request/Response Models ─────────────────────────────────────────────────

class SearchRequest(BaseModel):
    vector: list[float] = Field(..., description="Query vector")
    top_k: int = Field(default=10, ge=1, le=1000)
    tenant_id: str = Field(default="default", description="Tenant namespace")
    include_timings: bool = Field(
        default=False, description="Include per-shard timing breakdown"
    )


class SearchResponse(BaseModel):
    results: list[dict]
    latency_ms: float
    shards_scanned: int
    shards_failed: int = 0
    total_vectors_scanned: int
    shard_timings: list[dict] | None = None


class IndexRequest(BaseModel):
    vectors: list[dict] = Field(
        ...,
        description="List of {id, vector, metadata?} objects",
    )
    tenant_id: str = Field(default="default")


class IndexResponse(BaseModel):
    shards_created: int
    vectors_indexed: int
    vectors_buffered: int


class FlushRequest(BaseModel):
    tenant_id: str = Field(default="default")


class DeleteShardRequest(BaseModel):
    tenant_id: str
    shard_key: str


# ── Endpoints ───────────────────────────────────────────────────────────────

@app.post("/search", response_model=SearchResponse)
async def search_endpoint(req: SearchRequest):
    """Search for similar vectors across all tenant shards.

    The query vector is compared against every vector in the tenant's
    S3-resident shards via parallel async fan-out. Vectors are fetched
    from object storage, scanned, and discarded — never resident in RAM.
    """
    settings = get_settings()

    if len(req.vector) != settings.vector_dimensions:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Vector dimension mismatch: got {len(req.vector)}, "
                f"expected {settings.vector_dimensions}"
            ),
        )

    query_vec = np.array(req.vector, dtype=np.float32)
    result = await search(query_vec, req.tenant_id, req.top_k)

    response = SearchResponse(
        results=result.results,
        latency_ms=result.latency_ms,
        shards_scanned=result.shards_scanned,
        shards_failed=result.shards_failed,
        total_vectors_scanned=result.total_vectors_scanned,
    )

    if req.include_timings:
        response.shard_timings = result.shard_timings

    return response


@app.post("/index", response_model=IndexResponse)
async def index_endpoint(req: IndexRequest):
    """Add vectors to the index.

    Vectors are buffered and flushed to S3 as complete shards.
    Use /flush to force-write any remaining buffered vectors.
    """
    settings = get_settings()

    ids = []
    vectors = []
    for item in req.vectors:
        if "id" not in item or "vector" not in item:
            raise HTTPException(
                status_code=400,
                detail="Each vector must have 'id' and 'vector' fields",
            )
        if len(item["vector"]) != settings.vector_dimensions:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Vector dimension mismatch for '{item['id']}': "
                    f"got {len(item['vector'])}, expected {settings.vector_dimensions}"
                ),
            )
        ids.append(item["id"])
        vectors.append(item["vector"])

    result = await add_vectors(req.tenant_id, ids, vectors)
    return IndexResponse(**result)


@app.post("/flush")
async def flush_endpoint(req: FlushRequest):
    """Force-flush buffered vectors to S3 as a partial shard."""
    result = await flush_buffer(req.tenant_id)
    return result


@app.delete("/tenants/{tenant_id}/shards/{shard_key:path}")
async def delete_shard_endpoint(tenant_id: str, shard_key: str):
    """Delete a shard from S3 and unregister it."""
    await delete_shard(shard_key)
    await unregister_shard(tenant_id, shard_key)
    return {"deleted": shard_key}


@app.get("/health")
async def health():
    """Liveness check — returns 200 if the process is running."""
    return {"status": "ok", "engine": "s3-native-vector-engine"}


@app.get("/ready")
async def readiness():
    """Readiness check — verifies S3 and Redis are reachable."""
    from .registry import get_redis
    errors = []
    try:
        r = await get_redis()
        await r.ping()
    except Exception as e:
        errors.append(f"redis: {e}")

    try:
        from .storage import get_sync_client
        client = get_sync_client()
        settings = get_settings()
        client.head_bucket(Bucket=settings.s3_bucket)
    except Exception as e:
        errors.append(f"s3: {e}")

    if errors:
        raise HTTPException(status_code=503, detail={"errors": errors})
    return {"status": "ready"}


@app.get("/stats")
async def stats():
    """Engine statistics: tenants, shards, vectors."""
    return await get_stats()


@app.get("/metrics")
async def metrics():
    """Query performance metrics."""
    return await get_metrics()


@app.get("/tenants")
async def list_tenants():
    """List all tenants."""
    tenants = await get_all_tenants()
    return {"tenants": tenants, "count": len(tenants)}


@app.get("/tenants/{tenant_id}/shards")
async def tenant_shards(tenant_id: str):
    """List shards for a tenant."""
    shards = await get_tenant_shards(tenant_id)
    return {"tenant_id": tenant_id, "shards": shards, "count": len(shards)}


# ── Entrypoint ──────────────────────────────────────────────────────────────

def main():
    settings = get_settings()
    uvicorn.run(
        "s3vec.main:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        reload=False,
        log_level=settings.log_level,
    )


if __name__ == "__main__":
    main()
