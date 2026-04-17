"""Multi-stage retrieval pipeline engine.

Executes ordered stage pipelines against the shard cluster. Each stage
receives the result context from the previous stage and produces output
for the next.  Stages run in sequence (DAG is linear for v1).

Stage types:
  - feature_search: hybrid dense+sparse+BM25 search
  - attribute_filter: post-filter on payload fields
  - sort_relevance: sort by search score
  - sort_attribute: sort by a payload field
  - sample: random/head/tail sampling
  - group_by: group results by a payload field
  - aggregate: compute aggregates (count, sum, avg, min, max)
  - mmr: Maximal Marginal Relevance diversity re-ranking
  - document_enrich: semantic JOIN across namespaces
  - rerank: placeholder for cross-encoder re-ranking (requires Ray)
  - llm_filter: placeholder for LLM-based filtering (requires Ray)
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import orjson

from .grpc_coordinator import GrpcCoordinator

logger = logging.getLogger(__name__)


# ── Pipeline Data Structures ────────────────────────────────────────────────


@dataclass
class StageDefinition:
    """A single pipeline stage."""

    stage_type: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineDefinition:
    """An ordered list of stages."""

    stages: list[StageDefinition]
    namespace: str = "default"


@dataclass
class PipelineContext:
    """Mutable context passed through the pipeline."""

    results: list[dict] = field(default_factory=list)
    namespace: str = "default"
    stage_timings: list[dict] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    """Final output of a pipeline execution."""

    results: list[dict]
    total_latency_ms: float
    stages_executed: int
    stage_timings: list[dict]
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Stage Registry ──────────────────────────────────────────────────────────

_STAGE_HANDLERS: dict[str, Any] = {}


def _register(name: str):
    """Decorator to register a stage handler."""

    def decorator(fn):
        _STAGE_HANDLERS[name] = fn
        return fn

    return decorator


# ── Pipeline Engine ─────────────────────────────────────────────────────────


class PipelineEngine:
    """Execute multi-stage retrieval pipelines."""

    def __init__(self, coordinator: GrpcCoordinator):
        self.coordinator = coordinator

    async def execute(
        self,
        pipeline: PipelineDefinition,
        *,
        cancel_event: asyncio.Event | None = None,
    ) -> PipelineResult:
        """Run the pipeline stages sequentially, passing context through."""
        t0 = time.perf_counter()
        ctx = PipelineContext(namespace=pipeline.namespace)

        for i, stage in enumerate(pipeline.stages):
            # Check cancellation
            if cancel_event and cancel_event.is_set():
                logger.info("pipeline cancelled at stage %d", i)
                break

            handler = _STAGE_HANDLERS.get(stage.stage_type)
            if handler is None:
                raise ValueError(f"unknown stage type: {stage.stage_type}")

            stage_t0 = time.perf_counter()
            await handler(ctx, stage.params, self.coordinator)
            stage_ms = (time.perf_counter() - stage_t0) * 1000

            ctx.stage_timings.append(
                {
                    "stage": i,
                    "type": stage.stage_type,
                    "latency_ms": round(stage_ms, 2),
                    "result_count": len(ctx.results),
                }
            )

        total_ms = (time.perf_counter() - t0) * 1000

        return PipelineResult(
            results=ctx.results,
            total_latency_ms=round(total_ms, 2),
            stages_executed=len(ctx.stage_timings),
            stage_timings=ctx.stage_timings,
            metadata=ctx.metadata,
        )

    async def execute_streaming(
        self,
        pipeline: PipelineDefinition,
        *,
        cancel_event: asyncio.Event | None = None,
    ):
        """Generator that yields stage completions for SSE streaming."""
        t0 = time.perf_counter()
        ctx = PipelineContext(namespace=pipeline.namespace)

        for i, stage in enumerate(pipeline.stages):
            if cancel_event and cancel_event.is_set():
                yield {
                    "event": "cancelled",
                    "stage": i,
                    "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
                }
                return

            handler = _STAGE_HANDLERS.get(stage.stage_type)
            if handler is None:
                yield {"event": "error", "message": f"unknown stage: {stage.stage_type}"}
                return

            stage_t0 = time.perf_counter()
            await handler(ctx, stage.params, self.coordinator)
            stage_ms = (time.perf_counter() - stage_t0) * 1000

            yield {
                "event": "stage_complete",
                "stage": i,
                "type": stage.stage_type,
                "latency_ms": round(stage_ms, 2),
                "result_count": len(ctx.results),
                "results": ctx.results,
            }

        yield {
            "event": "done",
            "total_latency_ms": round((time.perf_counter() - t0) * 1000, 2),
            "stages_executed": len(ctx.stage_timings),
            "result_count": len(ctx.results),
            "results": ctx.results,
        }


# ── Stage Implementations ──────────────────────────────────────────────────


@_register("feature_search")
async def _stage_feature_search(
    ctx: PipelineContext,
    params: dict[str, Any],
    coordinator: GrpcCoordinator,
) -> None:
    """Hybrid search: dense + sparse + BM25 with fusion."""
    result = await coordinator.search(
        namespace=ctx.namespace,
        dense_vector=params.get("dense_vector"),
        sparse_indices=params.get("sparse_indices"),
        sparse_values=params.get("sparse_values"),
        text_query=params.get("text_query"),
        top_k=params.get("top_k", 50),
        fusion=params.get("fusion", "rrf"),
        include_payloads=params.get("include_payloads", True),
        filter_expr=params.get("filter"),
    )
    ctx.results = result.results


@_register("attribute_filter")
async def _stage_attribute_filter(
    ctx: PipelineContext,
    params: dict[str, Any],
    _coordinator: GrpcCoordinator,
) -> None:
    """Filter results by payload attributes.

    Params:
      conditions: list of {"field": str, "op": str, "value": Any}
      logic: "and" | "or" (default: "and")
    """
    conditions = params.get("conditions", [])
    logic = params.get("logic", "and")

    if not conditions:
        return

    def _matches(doc: dict) -> bool:
        payload = doc.get("payload", {})
        if payload is None:
            payload = {}
        results = []
        for cond in conditions:
            field_name = cond["field"]
            op = cond.get("op", "eq")
            value = cond.get("value")
            doc_value = payload.get(field_name)

            match = _eval_condition(doc_value, op, value)
            results.append(match)

        if logic == "or":
            return any(results)
        return all(results)

    ctx.results = [doc for doc in ctx.results if _matches(doc)]


def _eval_condition(doc_value: Any, op: str, value: Any) -> bool:
    """Evaluate a single filter condition."""
    if doc_value is None:
        return False
    try:
        match op:
            case "eq":
                return doc_value == value
            case "ne":
                return doc_value != value
            case "gt":
                return float(doc_value) > float(value)
            case "gte":
                return float(doc_value) >= float(value)
            case "lt":
                return float(doc_value) < float(value)
            case "lte":
                return float(doc_value) <= float(value)
            case "in":
                return doc_value in value
            case "contains":
                return value in doc_value
            case _:
                return False
    except (TypeError, ValueError):
        return False


@_register("sort_relevance")
async def _stage_sort_relevance(
    ctx: PipelineContext,
    params: dict[str, Any],
    _coordinator: GrpcCoordinator,
) -> None:
    """Sort results by score.

    Params:
      order: "desc" | "asc" (default: "desc")
    """
    order = params.get("order", "desc")
    ctx.results.sort(
        key=lambda d: d.get("score", 0.0),
        reverse=(order == "desc"),
    )


@_register("sort_attribute")
async def _stage_sort_attribute(
    ctx: PipelineContext,
    params: dict[str, Any],
    _coordinator: GrpcCoordinator,
) -> None:
    """Sort results by a payload field.

    Params:
      field: str — payload field name
      order: "desc" | "asc" (default: "asc")
      missing: "last" | "first" (default: "last") — where to place missing values
    """
    field_name = params.get("field")
    if not field_name:
        return

    order = params.get("order", "asc")
    missing = params.get("missing", "last")
    reverse = order == "desc"

    sentinel = float("inf") if (missing == "last") != reverse else float("-inf")

    def sort_key(doc: dict):
        payload = doc.get("payload", {}) or {}
        val = payload.get(field_name)
        if val is None:
            return sentinel
        try:
            return float(val)
        except (TypeError, ValueError):
            return str(val)

    ctx.results.sort(key=sort_key, reverse=reverse)


@_register("sample")
async def _stage_sample(
    ctx: PipelineContext,
    params: dict[str, Any],
    _coordinator: GrpcCoordinator,
) -> None:
    """Sample results.

    Params:
      n: int — number of samples
      method: "random" | "head" | "tail" (default: "head")
      seed: int | None — random seed for reproducibility
    """
    n = params.get("n", 10)
    method = params.get("method", "head")

    if method == "head":
        ctx.results = ctx.results[:n]
    elif method == "tail":
        ctx.results = ctx.results[-n:]
    elif method == "random":
        seed = params.get("seed")
        rng = random.Random(seed)
        ctx.results = rng.sample(ctx.results, min(n, len(ctx.results)))


@_register("group_by")
async def _stage_group_by(
    ctx: PipelineContext,
    params: dict[str, Any],
    _coordinator: GrpcCoordinator,
) -> None:
    """Group results by a payload field.

    Params:
      field: str — payload field to group by
      top_k_per_group: int — max results per group (default: 3)

    Replaces results with grouped structure and stores groups in metadata.
    """
    field_name = params.get("field")
    if not field_name:
        return

    top_k_per_group = params.get("top_k_per_group", 3)

    groups: dict[str, list[dict]] = defaultdict(list)
    for doc in ctx.results:
        payload = doc.get("payload", {}) or {}
        key = str(payload.get(field_name, "__none__"))
        if len(groups[key]) < top_k_per_group:
            groups[key].append(doc)

    ctx.metadata["groups"] = {k: len(v) for k, v in groups.items()}
    ctx.metadata["group_field"] = field_name

    # Flatten back: interleave groups
    ctx.results = []
    for docs in groups.values():
        ctx.results.extend(docs)


@_register("aggregate")
async def _stage_aggregate(
    ctx: PipelineContext,
    params: dict[str, Any],
    _coordinator: GrpcCoordinator,
) -> None:
    """Compute aggregates over results.

    Params:
      operations: list of {"op": str, "field": str, "alias": str}
        ops: "count", "sum", "avg", "min", "max"

    Stores aggregation results in metadata["aggregations"].
    Does NOT modify ctx.results.
    """
    operations = params.get("operations", [])
    agg_results: dict[str, Any] = {}

    for op_def in operations:
        op = op_def.get("op", "count")
        field_name = op_def.get("field", "score")
        alias = op_def.get("alias", f"{op}_{field_name}")

        values = []
        for doc in ctx.results:
            if field_name == "score":
                val = doc.get("score")
            else:
                payload = doc.get("payload", {}) or {}
                val = payload.get(field_name)
            if val is not None:
                try:
                    values.append(float(val))
                except (TypeError, ValueError):
                    pass

        match op:
            case "count":
                agg_results[alias] = len(ctx.results)
            case "sum":
                agg_results[alias] = sum(values) if values else 0
            case "avg":
                agg_results[alias] = (sum(values) / len(values)) if values else 0
            case "min":
                agg_results[alias] = min(values) if values else None
            case "max":
                agg_results[alias] = max(values) if values else None

    ctx.metadata["aggregations"] = agg_results


@_register("mmr")
async def _stage_mmr(
    ctx: PipelineContext,
    params: dict[str, Any],
    coordinator: GrpcCoordinator,
) -> None:
    """Maximal Marginal Relevance diversity re-ranking.

    Requires a dense_vector query to compute relevance + diversity.

    Params:
      dense_vector: list[float] — the original query vector (required)
      lambda: float — relevance/diversity trade-off (0.0–1.0, default 0.5)
      k: int — number of results (default: 10)
    """
    dense_vector = params.get("dense_vector")
    if dense_vector is None:
        return  # Cannot do MMR without a query vector

    lambda_val = params.get("lambda", 0.5)
    k = params.get("k", 10)

    # We need document vectors for MMR. If results have them, use those.
    # Otherwise fall back to score-only approximate MMR.
    docs_with_vectors = []
    docs_without_vectors = []

    for doc in ctx.results:
        vec = doc.get("dense_vector")
        if vec:
            docs_with_vectors.append(doc)
        else:
            docs_without_vectors.append(doc)

    if not docs_with_vectors:
        # No vectors available — fall back to simple score-based dedup
        # Use score as the sole signal (no diversity possible)
        ctx.results = ctx.results[:k]
        return

    # Compute MMR in Python (coordinator-side, cross-shard)
    import math

    def _cosine_sim(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0
        return dot / (norm_a * norm_b)

    # Normalize query similarities
    query_sims = [_cosine_sim(d.get("dense_vector", []), dense_vector) for d in docs_with_vectors]
    min_s = min(query_sims) if query_sims else 0
    max_s = max(query_sims) if query_sims else 1
    range_s = max(max_s - min_s, 1e-10)
    norm_sims = [(s - min_s) / range_s for s in query_sims]

    remaining = list(range(len(docs_with_vectors)))
    selected: list[int] = []
    result: list[dict] = []

    for _ in range(min(k, len(docs_with_vectors))):
        if not remaining:
            break

        best_idx = 0
        best_mmr = float("-inf")

        for ri, doc_idx in enumerate(remaining):
            relevance = norm_sims[doc_idx]
            max_sim = 0.0
            if selected:
                max_sim = max(
                    _cosine_sim(
                        docs_with_vectors[doc_idx].get("dense_vector", []),
                        docs_with_vectors[sel_idx].get("dense_vector", []),
                    )
                    for sel_idx in selected
                )
                max_sim = (max_sim + 1.0) / 2.0

            mmr_score = lambda_val * relevance - (1.0 - lambda_val) * max_sim
            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_idx = ri

        chosen = remaining.pop(best_idx)
        selected.append(chosen)
        doc = docs_with_vectors[chosen].copy()
        doc["mmr_score"] = best_mmr
        result.append(doc)

    ctx.results = result


@_register("document_enrich")
async def _stage_document_enrich(
    ctx: PipelineContext,
    params: dict[str, Any],
    coordinator: GrpcCoordinator,
) -> None:
    """Semantic JOIN: enrich results with data from another namespace.

    Params:
      target_namespace: str — namespace to search for enrichment
      vector_field: str — field in result payload containing the join vector
        (or "dense_vector" to use the result's dense vector)
      attach_fields: list[str] — fields from the matched doc to attach
      top_k: int — matches per document (default: 1)
      prefix: str — prefix for attached fields (default: "enriched_")
    """
    target_ns = params.get("target_namespace")
    if not target_ns:
        return

    vector_field = params.get("vector_field", "dense_vector")
    attach_fields = params.get("attach_fields", [])
    top_k_enrich = params.get("top_k", 1)
    prefix = params.get("prefix", "enriched_")

    for doc in ctx.results:
        # Get the vector to use for the join
        if vector_field == "dense_vector":
            join_vector = doc.get("dense_vector")
        else:
            payload = doc.get("payload", {}) or {}
            join_vector = payload.get(vector_field)

        if not join_vector:
            continue

        # Search the target namespace
        enrich_result = await coordinator.search(
            namespace=target_ns,
            dense_vector=join_vector,
            top_k=top_k_enrich,
            include_payloads=True,
        )

        if enrich_result.results:
            matched = enrich_result.results[0]
            matched_payload = matched.get("payload", {}) or {}
            if not attach_fields:
                # Attach all fields
                for k, v in matched_payload.items():
                    doc.setdefault("payload", {})[f"{prefix}{k}"] = v
            else:
                for f in attach_fields:
                    if f in matched_payload:
                        doc.setdefault("payload", {})[f"{prefix}{f}"] = matched_payload[f]
            doc.setdefault("payload", {})[f"{prefix}score"] = matched.get("score", 0)


@_register("rerank")
async def _stage_rerank(
    ctx: PipelineContext,
    params: dict[str, Any],
    _coordinator: GrpcCoordinator,
) -> None:
    """Placeholder for cross-encoder re-ranking (requires Ray Serve).

    Params:
      model: str — model name (e.g. "cross-encoder/ms-marco-MiniLM-L-12-v2")
      query: str — the original text query
      top_k: int — number of results after re-ranking
      endpoint: str — Ray Serve endpoint URL (optional)

    Currently a no-op placeholder. When Ray is available, this will call
    a cross-encoder model to re-score results.
    """
    top_k = params.get("top_k")
    if top_k:
        ctx.results = ctx.results[:top_k]
    logger.info(
        "rerank stage: no-op placeholder (Ray Serve not configured), "
        "results truncated to %d",
        len(ctx.results),
    )


@_register("llm_filter")
async def _stage_llm_filter(
    ctx: PipelineContext,
    params: dict[str, Any],
    _coordinator: GrpcCoordinator,
) -> None:
    """Placeholder for LLM-based semantic filtering (requires Ray Serve).

    Params:
      prompt: str — the filter prompt
      model: str — LLM model name
      endpoint: str — Ray Serve endpoint URL

    Currently a no-op placeholder.
    """
    logger.info("llm_filter stage: no-op placeholder (Ray Serve not configured)")
