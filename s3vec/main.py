"""S3-Native Vector Engine — FastAPI Gateway.

Routes requests to Rust shard gRPC servers via the GrpcCoordinator.
Supports dense, sparse, BM25, and hybrid search with payload filters.
Includes multi-stage retrieval pipeline engine with SSE streaming.
"""

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager

import orjson
import structlog
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from .config import get_settings
from .grpc_coordinator import GrpcCoordinator
from .pipeline import PipelineDefinition, PipelineEngine, StageDefinition
from .ray.progress import ProgressActor


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

# Global coordinator — initialized at startup
coordinator: GrpcCoordinator | None = None
pipeline_engine: PipelineEngine | None = None
progress_actor: ProgressActor | None = None


# ── Lifespan ────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    global coordinator, pipeline_engine, progress_actor
    settings = get_settings()
    _configure_logging(settings)

    # Parse shard addresses and init coordinator
    addrs = [a.strip() for a in settings.shard_addresses.split(",") if a.strip()]
    coordinator = GrpcCoordinator(
        shard_addresses=addrs,
        timeout_seconds=settings.shard_timeout_seconds,
    )
    pipeline_engine = PipelineEngine(coordinator)
    progress_actor = ProgressActor()

    logger.info(
        "starting",
        shard_addresses=addrs,
        dimensions=settings.vector_dimensions,
    )

    yield

    if coordinator:
        await coordinator.close()
    logger.info("shutdown")


app = FastAPI(
    title="S3-Native Vector Engine",
    description="Multimodal vector search — gateway to Rust shard cluster.",
    version="0.2.0",
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
    dense_vector: list[float] | None = Field(None, description="Dense query vector")
    sparse_indices: list[int] | None = Field(None, description="Sparse query token IDs")
    sparse_values: list[float] | None = Field(None, description="Sparse query weights")
    text_query: str | None = Field(None, description="BM25 text query")
    top_k: int = Field(default=10, ge=1, le=1000)
    namespace: str = Field(default="default", description="Namespace for routing")
    fusion: str = Field(default="rrf", description="Fusion: rrf, dbsf, linear")
    include_payloads: bool = Field(default=False)
    filter: dict | None = Field(None, description="Payload filter expression")


class SearchResponse(BaseModel):
    results: list[dict]
    latency_ms: float
    shards_queried: int
    shards_failed: int = 0


class UpsertRequest(BaseModel):
    id: str
    dense_vector: list[float] | None = None
    sparse_indices: list[int] | None = None
    sparse_values: list[float] | None = None
    text_fields: dict[str, str] | None = None
    payload: dict | None = None
    namespace: str = "default"


class UpsertResponse(BaseModel):
    wal_sequence: int
    shard: str


class DeleteRequest(BaseModel):
    id: str
    namespace: str = "default"


class DeleteResponse(BaseModel):
    wal_sequence: int
    found: bool
    shard: str


# ── Endpoints ───────────────────────────────────────────────────────────────

@app.post("/search", response_model=SearchResponse)
async def search_endpoint(req: SearchRequest):
    """Hybrid search across all shard servers."""
    if coordinator is None:
        raise HTTPException(status_code=503, detail="Coordinator not initialized")

    result = await coordinator.search(
        namespace=req.namespace,
        dense_vector=req.dense_vector,
        sparse_indices=req.sparse_indices,
        sparse_values=req.sparse_values,
        text_query=req.text_query,
        top_k=req.top_k,
        fusion=req.fusion,
        include_payloads=req.include_payloads,
        filter_expr=req.filter,
    )

    return SearchResponse(
        results=result.results,
        latency_ms=result.latency_ms,
        shards_queried=result.shards_queried,
        shards_failed=result.shards_failed,
    )


@app.post("/upsert", response_model=UpsertResponse)
async def upsert_endpoint(req: UpsertRequest):
    """Upsert a document into the shard cluster."""
    if coordinator is None:
        raise HTTPException(status_code=503, detail="Coordinator not initialized")

    result = await coordinator.upsert(
        namespace=req.namespace,
        id=req.id,
        dense_vector=req.dense_vector,
        sparse_indices=req.sparse_indices,
        sparse_values=req.sparse_values,
        text_fields=req.text_fields,
        payload=req.payload,
    )

    return UpsertResponse(**result)


@app.post("/delete", response_model=DeleteResponse)
async def delete_endpoint(req: DeleteRequest):
    """Delete a document from the shard cluster."""
    if coordinator is None:
        raise HTTPException(status_code=503, detail="Coordinator not initialized")

    result = await coordinator.delete(
        namespace=req.namespace,
        id=req.id,
    )

    return DeleteResponse(**result)


@app.get("/health")
async def health():
    """Liveness check."""
    return {"status": "ok", "engine": "s3-vector-engine", "version": "0.2.0"}


@app.get("/stats")
async def stats():
    """Get stats from all shard servers."""
    if coordinator is None:
        raise HTTPException(status_code=503, detail="Coordinator not initialized")

    return await coordinator.get_all_stats()


@app.get("/catalog")
async def catalog():
    """Namespace catalog — INFORMATION_SCHEMA equivalent.

    Returns shard topology, dimensions, vector counts, and WAL positions
    for every shard in the cluster.
    """
    if coordinator is None:
        raise HTTPException(status_code=503, detail="Coordinator not initialized")

    all_stats = await coordinator.get_all_stats()

    shards = []
    total_dense = 0
    total_sparse = 0
    total_payloads = 0

    for addr, shard_stats in all_stats.items():
        if "error" in shard_stats:
            shards.append({"address": addr, "status": "unreachable", "error": shard_stats["error"]})
            continue
        total_dense += shard_stats.get("dense_count", 0)
        total_sparse += shard_stats.get("sparse_count", 0)
        total_payloads += shard_stats.get("payload_count", 0)
        shards.append({
            "address": addr,
            "status": "online",
            "dense_count": shard_stats.get("dense_count", 0),
            "sparse_count": shard_stats.get("sparse_count", 0),
            "sparse_vocab_size": shard_stats.get("sparse_vocab_size", 0),
            "payload_count": shard_stats.get("payload_count", 0),
            "wal_sequence": shard_stats.get("wal_sequence", 0),
            "dim": shard_stats.get("dim", 0),
        })

    return {
        "cluster": {
            "total_shards": len(shards),
            "online_shards": sum(1 for s in shards if s["status"] == "online"),
            "total_dense_vectors": total_dense,
            "total_sparse_vectors": total_sparse,
            "total_payloads": total_payloads,
        },
        "shards": shards,
    }


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


# ── Pipeline Models ─────────────────────────────────────────────────────────

class PipelineStageRequest(BaseModel):
    stage_type: str = Field(..., description="Stage type: feature_search, attribute_filter, sort_relevance, sort_attribute, sample, group_by, aggregate, mmr, document_enrich, rerank, llm_filter")
    params: dict = Field(default_factory=dict, description="Stage-specific parameters")


class PipelineRequest(BaseModel):
    stages: list[PipelineStageRequest] = Field(..., min_length=1, description="Ordered list of pipeline stages")
    namespace: str = Field(default="default", description="Namespace for routing")
    stream: bool = Field(default=False, description="If true, stream results via SSE")


class PipelineResponse(BaseModel):
    results: list[dict]
    total_latency_ms: float
    stages_executed: int
    stage_timings: list[dict]
    metadata: dict = Field(default_factory=dict)


# ── Pipeline Endpoints ──────────────────────────────────────────────────────

@app.post("/pipeline")
async def pipeline_endpoint(req: PipelineRequest, request: Request):
    """Execute a multi-stage retrieval pipeline.

    Supports SSE streaming when stream=true. Each stage completes and
    emits its results before the next stage starts.
    """
    if pipeline_engine is None:
        raise HTTPException(status_code=503, detail="Pipeline engine not initialized")

    pipeline_def = PipelineDefinition(
        stages=[
            StageDefinition(stage_type=s.stage_type, params=s.params)
            for s in req.stages
        ],
        namespace=req.namespace,
    )

    if req.stream:
        cancel_event = asyncio.Event()

        async def event_generator():
            try:
                async for event in pipeline_engine.execute_streaming(
                    pipeline_def, cancel_event=cancel_event
                ):
                    if await request.is_disconnected():
                        cancel_event.set()
                        return
                    yield {
                        "event": event.get("event", "message"),
                        "data": orjson.dumps(event).decode(),
                    }
            except Exception as e:
                yield {
                    "event": "error",
                    "data": orjson.dumps({"error": str(e)}).decode(),
                }

        return EventSourceResponse(event_generator())

    result = await pipeline_engine.execute(pipeline_def)
    return ORJSONResponse(
        {
            "results": result.results,
            "total_latency_ms": result.total_latency_ms,
            "stages_executed": result.stages_executed,
            "stage_timings": result.stage_timings,
            "metadata": result.metadata,
        }
    )


# ── Ingest Models ───────────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    job_id: str
    status: str
    extractors: list[str]


class BatchIngestRequest(BaseModel):
    s3_prefix: str = Field(..., description="S3 prefix to scan for files")
    extractors: list[str] = Field(
        default_factory=lambda: ["e5_large"],
        description="List of extractor names to apply",
    )
    batch_size: int = Field(default=32, ge=1, le=1000)


class BatchIngestResponse(BaseModel):
    job_id: str
    status: str
    extractors: list[str]
    item_count: int | None = None


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    total: int
    completed: int
    failed: int
    pct: float
    elapsed_s: float
    stage: str
    error: str | None = None
    metadata: dict = Field(default_factory=dict)


# ── Ingest Endpoints ────────────────────────────────────────────────────────

@app.post("/namespaces/{ns}/ingest", response_model=IngestResponse)
async def ingest_file(
    ns: str,
    file: UploadFile = File(...),
    extractors: str = Form(default="e5_large"),
):
    """Ingest a single file — triggers extraction pipeline.

    Reads the uploaded file, runs specified extractors, and writes
    features to the shard cluster. Runs synchronously for single files.
    """
    if coordinator is None or progress_actor is None:
        raise HTTPException(status_code=503, detail="Not initialized")

    from .extraction import get_extractor, list_extractors as list_ext
    from .ray.pipeline import ExtractorConfig, PipelineConfig, run_extraction_pipeline_local

    settings = get_settings()
    addrs = [a.strip() for a in settings.shard_addresses.split(",") if a.strip()]

    ext_names = [e.strip() for e in extractors.split(",") if e.strip()]
    # Validate extractor names
    available = list_ext()
    for name in ext_names:
        if name not in available:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown extractor: {name!r}. Available: {sorted(available.keys())}",
            )

    content = await file.read()
    content_type = file.content_type or "application/octet-stream"
    doc_id = f"{ns}:{file.filename or uuid.uuid4().hex}"

    config = PipelineConfig(
        namespace=ns,
        extractors=[ExtractorConfig(name=n) for n in ext_names],
        shard_addresses=addrs,
    )

    # Run in thread pool to avoid blocking the event loop
    import functools
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        functools.partial(
            run_extraction_pipeline_local,
            items=[{"id": doc_id, "content": content, "content_type": content_type}],
            config=config,
            progress=progress_actor,
        ),
    )

    return IngestResponse(
        job_id=result["job_id"],
        status=result["status"],
        extractors=ext_names,
    )


@app.post("/namespaces/{ns}/ingest/batch", response_model=BatchIngestResponse)
async def batch_ingest(ns: str, req: BatchIngestRequest):
    """Batch ingest from S3 prefix — triggers extraction pipeline.

    Scans the S3 prefix for files, creates a batch extraction job,
    and returns a job_id for progress polling via GET /jobs/{id}.
    """
    if coordinator is None or progress_actor is None:
        raise HTTPException(status_code=503, detail="Not initialized")

    from .extraction import list_extractors as list_ext
    from .ray.pipeline import ExtractorConfig, PipelineConfig

    settings = get_settings()
    addrs = [a.strip() for a in settings.shard_addresses.split(",") if a.strip()]

    # Validate extractors
    available = list_ext()
    for name in req.extractors:
        if name not in available:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown extractor: {name!r}. Available: {sorted(available.keys())}",
            )

    job_id = f"job-{uuid.uuid4().hex[:12]}"

    config = PipelineConfig(
        namespace=ns,
        extractors=[ExtractorConfig(name=n) for n in req.extractors],
        shard_addresses=addrs,
        job_id=job_id,
    )

    # List objects from S3 prefix
    try:
        import aioboto3

        session = aioboto3.Session()
        async with session.client(
            "s3",
            endpoint_url=settings.s3_endpoint,
            aws_access_key_id=settings.s3_access_key,
            aws_secret_access_key=settings.s3_secret_key,
            region_name=settings.s3_region,
        ) as s3:
            resp = await s3.list_objects_v2(
                Bucket=settings.s3_bucket,
                Prefix=req.s3_prefix,
            )
            objects = resp.get("Contents", [])
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"S3 list failed: {e}")

    item_count = len(objects)
    progress_actor.start_job(
        job_id,
        total=item_count * len(req.extractors),
        stage="queued",
        metadata={"namespace": ns, "s3_prefix": req.s3_prefix},
    )

    # Dispatch async — download and process in background
    async def _run_batch():
        from .ray.pipeline import run_extraction_pipeline_local

        progress_actor.set_stage(job_id, "downloading")

        items = []
        try:
            session = aioboto3.Session()
            async with session.client(
                "s3",
                endpoint_url=settings.s3_endpoint,
                aws_access_key_id=settings.s3_access_key,
                aws_secret_access_key=settings.s3_secret_key,
                region_name=settings.s3_region,
            ) as s3:
                for obj in objects:
                    key = obj["Key"]
                    resp = await s3.get_object(Bucket=settings.s3_bucket, Key=key)
                    content = await resp["Body"].read()
                    # Infer content type from extension
                    ext = key.rsplit(".", 1)[-1].lower() if "." in key else ""
                    ct_map = {
                        "jpg": "image/jpeg", "jpeg": "image/jpeg",
                        "png": "image/png", "gif": "image/gif",
                        "mp4": "video/mp4", "mov": "video/quicktime",
                        "wav": "audio/wav", "mp3": "audio/mpeg",
                        "txt": "text/plain", "json": "application/json",
                    }
                    content_type = ct_map.get(ext, "application/octet-stream")
                    items.append({
                        "id": f"{ns}:{key}",
                        "content": content,
                        "content_type": content_type,
                    })
        except Exception as e:
            progress_actor.fail_job(job_id, f"S3 download failed: {e}")
            return

        import functools
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            functools.partial(
                run_extraction_pipeline_local,
                items=items,
                config=config,
                progress=progress_actor,
            ),
        )

    asyncio.create_task(_run_batch())

    return BatchIngestResponse(
        job_id=job_id,
        status="queued",
        extractors=req.extractors,
        item_count=item_count,
    )


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get batch job status and progress."""
    if progress_actor is None:
        raise HTTPException(status_code=503, detail="Not initialized")

    progress = progress_actor.get_progress(job_id)
    if progress is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    return JobStatusResponse(**progress)


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
