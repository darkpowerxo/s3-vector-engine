"""Integration test: Pipeline engine against a running shard-server.

Expects shard-server running at localhost:9051 with dim=4 and text-fields=text.
Expects data already seeded (run test_grpc_integration.py first).
"""

import asyncio
import sys


async def main():
    from s3vec.grpc_coordinator import GrpcCoordinator
    from s3vec.pipeline import PipelineDefinition, PipelineEngine, StageDefinition

    addr = "localhost:9051"
    coord = GrpcCoordinator(shard_addresses=[addr])
    engine = PipelineEngine(coord)

    # Seed data first
    print("Seeding data...", end=" ")
    vectors = {
        "v1": ([1.0, 0.0, 0.0, 0.0], {"label": "alpha", "category": "A", "priority": 1}),
        "v2": ([0.0, 1.0, 0.0, 0.0], {"label": "beta",  "category": "B", "priority": 2}),
        "v3": ([0.0, 0.0, 1.0, 0.0], {"label": "gamma", "category": "A", "priority": 3}),
        "v4": ([0.7, 0.7, 0.0, 0.0], {"label": "delta", "category": "B", "priority": 4}),
        "v5": ([0.5, 0.5, 0.5, 0.5], {"label": "epsilon", "category": "A", "priority": 5}),
    }
    for vid, (vec, payload) in vectors.items():
        await coord.upsert(id=vid, dense_vector=vec, payload=payload)

    await coord.upsert(
        id="doc1",
        dense_vector=[0.1, 0.2, 0.3, 0.4],
        text_fields={"text": "the quick brown fox jumps over the lazy dog"},
        payload={"source": "test", "animal": "fox", "category": "C", "priority": 10},
    )
    await coord.upsert(
        id="doc2",
        dense_vector=[0.4, 0.3, 0.2, 0.1],
        text_fields={"text": "a fast red cat sits on the warm mat"},
        payload={"source": "test", "animal": "cat", "category": "C", "priority": 20},
    )
    print("OK")

    # ── 1. Simple feature_search pipeline ───────────────────────────
    print("1. feature_search pipeline...", end=" ")
    pipeline = PipelineDefinition(
        stages=[
            StageDefinition(
                stage_type="feature_search",
                params={
                    "dense_vector": [1.0, 0.0, 0.0, 0.0],
                    "top_k": 5,
                    "include_payloads": True,
                },
            ),
        ],
    )
    result = await engine.execute(pipeline)
    assert len(result.results) >= 3, f"expected >=3 results, got {len(result.results)}"
    assert result.stages_executed == 1
    assert result.stage_timings[0]["type"] == "feature_search"
    print(f"OK  results={len(result.results)}  latency={result.total_latency_ms:.1f}ms")

    # ── 2. feature_search → attribute_filter ────────────────────────
    print("2. feature_search + attribute_filter...", end=" ")
    pipeline = PipelineDefinition(
        stages=[
            StageDefinition(
                stage_type="feature_search",
                params={
                    "dense_vector": [0.5, 0.5, 0.5, 0.5],
                    "top_k": 10,
                    "include_payloads": True,
                },
            ),
            StageDefinition(
                stage_type="attribute_filter",
                params={
                    "conditions": [
                        {"field": "category", "op": "eq", "value": "A"},
                    ],
                },
            ),
        ],
    )
    result = await engine.execute(pipeline)
    # Should only return category=A docs
    for doc in result.results:
        payload = doc.get("payload", {})
        assert payload.get("category") == "A", f"expected category A, got {payload}"
    assert result.stages_executed == 2
    print(f"OK  filtered_results={len(result.results)}")

    # ── 3. feature_search → sort_attribute ──────────────────────────
    print("3. feature_search + sort_attribute...", end=" ")
    pipeline = PipelineDefinition(
        stages=[
            StageDefinition(
                stage_type="feature_search",
                params={
                    "dense_vector": [0.5, 0.5, 0.5, 0.5],
                    "top_k": 10,
                    "include_payloads": True,
                },
            ),
            StageDefinition(
                stage_type="sort_attribute",
                params={"field": "priority", "order": "asc"},
            ),
        ],
    )
    result = await engine.execute(pipeline)
    priorities = [
        doc.get("payload", {}).get("priority", 999)
        for doc in result.results
        if doc.get("payload", {}).get("priority") is not None
    ]
    for i in range(len(priorities) - 1):
        assert priorities[i] <= priorities[i + 1], f"not sorted asc: {priorities}"
    print(f"OK  priorities={priorities}")

    # ── 4. feature_search → sort_relevance (desc) ───────────────────
    print("4. feature_search + sort_relevance...", end=" ")
    pipeline = PipelineDefinition(
        stages=[
            StageDefinition(
                stage_type="feature_search",
                params={
                    "dense_vector": [1.0, 0.0, 0.0, 0.0],
                    "top_k": 5,
                    "include_payloads": True,
                },
            ),
            StageDefinition(
                stage_type="sort_relevance",
                params={"order": "desc"},
            ),
        ],
    )
    result = await engine.execute(pipeline)
    scores = [doc["score"] for doc in result.results]
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1], f"not sorted desc: {scores}"
    print(f"OK  scores={[round(s, 3) for s in scores]}")

    # ── 5. feature_search → sample (head) ───────────────────────────
    print("5. feature_search + sample (head 2)...", end=" ")
    pipeline = PipelineDefinition(
        stages=[
            StageDefinition(
                stage_type="feature_search",
                params={"dense_vector": [0.5, 0.5, 0.5, 0.5], "top_k": 10, "include_payloads": True},
            ),
            StageDefinition(
                stage_type="sample",
                params={"n": 2, "method": "head"},
            ),
        ],
    )
    result = await engine.execute(pipeline)
    assert len(result.results) == 2, f"expected 2 samples, got {len(result.results)}"
    print(f"OK  results={len(result.results)}")

    # ── 6. feature_search → group_by ────────────────────────────────
    print("6. feature_search + group_by (category)...", end=" ")
    pipeline = PipelineDefinition(
        stages=[
            StageDefinition(
                stage_type="feature_search",
                params={"dense_vector": [0.5, 0.5, 0.5, 0.5], "top_k": 10, "include_payloads": True},
            ),
            StageDefinition(
                stage_type="group_by",
                params={"field": "category", "top_k_per_group": 2},
            ),
        ],
    )
    result = await engine.execute(pipeline)
    assert "groups" in result.metadata
    print(f"OK  groups={result.metadata['groups']}")

    # ── 7. feature_search → aggregate ───────────────────────────────
    print("7. feature_search + aggregate...", end=" ")
    pipeline = PipelineDefinition(
        stages=[
            StageDefinition(
                stage_type="feature_search",
                params={"dense_vector": [0.5, 0.5, 0.5, 0.5], "top_k": 10, "include_payloads": True},
            ),
            StageDefinition(
                stage_type="aggregate",
                params={
                    "operations": [
                        {"op": "count", "field": "score", "alias": "total"},
                        {"op": "avg", "field": "score", "alias": "avg_score"},
                        {"op": "max", "field": "priority", "alias": "max_priority"},
                    ],
                },
            ),
        ],
    )
    result = await engine.execute(pipeline)
    aggs = result.metadata.get("aggregations", {})
    assert "total" in aggs
    assert "avg_score" in aggs
    assert "max_priority" in aggs
    assert aggs["total"] > 0
    print(f"OK  aggregations={aggs}")

    # ── 8. Multi-stage: search → filter → sort → sample ────────────
    print("8. 4-stage pipeline: search → filter → sort → sample...", end=" ")
    pipeline = PipelineDefinition(
        stages=[
            StageDefinition(
                stage_type="feature_search",
                params={"dense_vector": [0.5, 0.5, 0.5, 0.5], "top_k": 10, "include_payloads": True},
            ),
            StageDefinition(
                stage_type="attribute_filter",
                params={"conditions": [{"field": "priority", "op": "gte", "value": 2}]},
            ),
            StageDefinition(
                stage_type="sort_attribute",
                params={"field": "priority", "order": "desc"},
            ),
            StageDefinition(
                stage_type="sample",
                params={"n": 3, "method": "head"},
            ),
        ],
    )
    result = await engine.execute(pipeline)
    assert result.stages_executed == 4
    assert len(result.results) <= 3
    assert len(result.stage_timings) == 4
    for doc in result.results:
        assert doc.get("payload", {}).get("priority", 0) >= 2
    print(f"OK  stages={result.stages_executed}  results={len(result.results)}")

    # ── 9. SSE streaming pipeline ───────────────────────────────────
    print("9. SSE streaming pipeline...", end=" ")
    pipeline = PipelineDefinition(
        stages=[
            StageDefinition(
                stage_type="feature_search",
                params={"dense_vector": [1.0, 0.0, 0.0, 0.0], "top_k": 5, "include_payloads": True},
            ),
            StageDefinition(
                stage_type="sort_relevance",
                params={"order": "desc"},
            ),
        ],
    )
    events = []
    async for event in engine.execute_streaming(pipeline):
        events.append(event)
    assert len(events) == 3  # 2 stage_complete + 1 done
    assert events[0]["event"] == "stage_complete"
    assert events[1]["event"] == "stage_complete"
    assert events[2]["event"] == "done"
    assert events[2]["result_count"] > 0
    print(f"OK  events={len(events)}  final_results={events[2]['result_count']}")

    # ── 10. Pipeline with cancellation ──────────────────────────────
    print("10. Pipeline cancellation...", end=" ")
    cancel = asyncio.Event()
    cancel.set()  # Cancel immediately
    pipeline = PipelineDefinition(
        stages=[
            StageDefinition(
                stage_type="feature_search",
                params={"dense_vector": [1.0, 0.0, 0.0, 0.0], "top_k": 5},
            ),
            StageDefinition(
                stage_type="sort_relevance",
                params={},
            ),
        ],
    )
    result = await engine.execute(pipeline, cancel_event=cancel)
    # Since cancel is set before first stage, should execute 0 stages
    assert result.stages_executed == 0
    print(f"OK  stages_executed={result.stages_executed}")

    # ── 11. Invalid stage type ─────────────────────────────────────
    print("11. Invalid stage type...", end=" ")
    pipeline = PipelineDefinition(
        stages=[
            StageDefinition(stage_type="nonexistent", params={}),
        ],
    )
    try:
        await engine.execute(pipeline)
        assert False, "should have raised ValueError"
    except ValueError as e:
        assert "unknown stage" in str(e)
        print(f"OK  error='{e}'")

    # ── 12. Rerank placeholder ─────────────────────────────────────
    print("12. Rerank placeholder...", end=" ")
    pipeline = PipelineDefinition(
        stages=[
            StageDefinition(
                stage_type="feature_search",
                params={"dense_vector": [1.0, 0.0, 0.0, 0.0], "top_k": 5, "include_payloads": True},
            ),
            StageDefinition(
                stage_type="rerank",
                params={"top_k": 2},
            ),
        ],
    )
    result = await engine.execute(pipeline)
    assert len(result.results) <= 2
    print(f"OK  results={len(result.results)}")

    await coord.close()
    print("\n✅ All pipeline tests passed!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"\n❌ FAILED: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
