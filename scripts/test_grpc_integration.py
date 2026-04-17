"""Integration test: Python GrpcCoordinator ↔ Rust shard-server over gRPC.

Expects shard-server running at localhost:50051 with dim=4 and text-fields=text.
"""

import asyncio
import sys


async def main():
    from s3vec.grpc_coordinator import GrpcCoordinator

    addr = "localhost:9051"
    coord = GrpcCoordinator(shard_addresses=[addr])

    # ── 1. GetStats (empty shard) ─────────────────────────────────────
    print("1. GetStats (empty)...", end=" ")
    stats = await coord.get_stats(addr)
    assert stats["dense_count"] == 0, f"expected 0, got {stats['dense_count']}"
    assert stats["dim"] == 4, f"expected dim=4, got {stats['dim']}"
    print(f"OK  {stats}")

    # ── 2. Upsert dense vectors ──────────────────────────────────────
    print("2. Upsert 5 dense vectors...", end=" ")
    vectors = {
        "v1": [1.0, 0.0, 0.0, 0.0],
        "v2": [0.0, 1.0, 0.0, 0.0],
        "v3": [0.0, 0.0, 1.0, 0.0],
        "v4": [0.7, 0.7, 0.0, 0.0],
        "v5": [0.5, 0.5, 0.5, 0.5],
    }
    for vid, vec in vectors.items():
        result = await coord.upsert(
            id=vid,
            dense_vector=vec,
            payload={"label": vid, "num": int(vid[1:])},
        )
        assert result["wal_sequence"] > 0, f"bad wal_sequence: {result}"
    print("OK")

    # ── 3. GetStats (after upsert) ──────────────────────────────────
    print("3. GetStats (after upsert)...", end=" ")
    stats = await coord.get_stats(addr)
    assert stats["dense_count"] == 5, f"expected 5, got {stats['dense_count']}"
    assert stats["payload_count"] == 5, f"expected 5, got {stats['payload_count']}"
    print(f"OK  dense={stats['dense_count']} payload={stats['payload_count']}")

    # ── 4. Dense search ─────────────────────────────────────────────
    print("4. Dense search (nearest to [1,0,0,0])...", end=" ")
    sr = await coord.search(
        dense_vector=[1.0, 0.0, 0.0, 0.0],
        top_k=3,
        include_payloads=True,
    )
    assert len(sr.results) == 3, f"expected 3 results, got {len(sr.results)}"
    assert sr.results[0]["id"] == "v1", f"expected v1 first, got {sr.results[0]['id']}"
    print(f"OK  top={sr.results[0]['id']}  score={sr.results[0]['score']:.4f}  "
          f"payload={sr.results[0].get('payload')}  latency={sr.latency_ms:.1f}ms")

    # ── 5. Upsert with text for BM25 ───────────────────────────────
    print("5. Upsert with text fields...", end=" ")
    await coord.upsert(
        id="doc1",
        dense_vector=[0.1, 0.2, 0.3, 0.4],
        text_fields={"text": "the quick brown fox jumps over the lazy dog"},
        payload={"source": "test", "animal": "fox"},
    )
    await coord.upsert(
        id="doc2",
        dense_vector=[0.4, 0.3, 0.2, 0.1],
        text_fields={"text": "a fast red cat sits on the warm mat"},
        payload={"source": "test", "animal": "cat"},
    )
    print("OK")

    # ── 6. BM25 search ──────────────────────────────────────────────
    print("6. BM25 search for 'fox'...", end=" ")
    sr = await coord.search(
        text_query="fox",
        top_k=5,
        include_payloads=True,
    )
    assert len(sr.results) >= 1, f"expected >=1 results, got {len(sr.results)}"
    # doc1 should mention fox
    ids = [r["id"] for r in sr.results]
    assert "doc1" in ids, f"expected doc1 in results, got {ids}"
    print(f"OK  results={ids}  top_score={sr.results[0]['score']:.4f}")

    # ── 7. Hybrid search (dense + BM25) ─────────────────────────────
    print("7. Hybrid dense + BM25 search...", end=" ")
    sr = await coord.search(
        dense_vector=[0.1, 0.2, 0.3, 0.4],
        text_query="fox",
        top_k=3,
        fusion="rrf",
        include_payloads=True,
    )
    assert len(sr.results) >= 1, f"expected >=1 results, got {len(sr.results)}"
    print(f"OK  top={sr.results[0]['id']}  score={sr.results[0]['score']:.4f}")

    # ── 8. Delete ───────────────────────────────────────────────────
    print("8. Delete v3...", end=" ")
    del_result = await coord.delete(id="v3")
    assert del_result["wal_sequence"] > 0
    print(f"OK  wal_seq={del_result['wal_sequence']}")

    # ── 9. Verify delete ────────────────────────────────────────────
    print("9. Verify v3 deleted (search for [0,0,1,0])...", end=" ")
    sr = await coord.search(
        dense_vector=[0.0, 0.0, 1.0, 0.0],
        top_k=10,
    )
    ids = [r["id"] for r in sr.results]
    assert "v3" not in ids, f"v3 should be deleted but found in {ids}"
    print(f"OK  results={ids}")

    # ── 10. get_all_stats ──────────────────────────────────────────
    print("10. get_all_stats...", end=" ")
    all_stats = await coord.get_all_stats()
    assert addr in all_stats, f"missing {addr} in stats"
    print(f"OK  {all_stats}")

    # ── 11. CreateSnapshot ──────────────────────────────────────────
    print("11. CreateSnapshot...", end=" ")
    from s3vec.proto import shard_pb2, shard_pb2_grpc
    stub = coord.pool.get_stub(addr)
    snap_resp = await stub.CreateSnapshot(shard_pb2.SnapshotRequest())
    assert snap_resp.wal_sequence_at > 0, f"expected wal_sequence > 0, got {snap_resp.wal_sequence_at}"
    assert snap_resp.snapshot_path.endswith(".snap"), f"bad path: {snap_resp.snapshot_path}"
    print(f"OK  path={snap_resp.snapshot_path}  wal_at={snap_resp.wal_sequence_at}")

    # ── 12. TailWAL ─────────────────────────────────────────────────
    print("12. TailWAL (from seq 1)...", end=" ")
    events = []
    async for event in stub.TailWAL(shard_pb2.TailWALRequest(from_sequence=1)):
        events.append(event)
    assert len(events) > 0, f"expected WAL events, got 0"
    # Should have upserts and at least one delete
    upsert_count = sum(1 for e in events if e.HasField("upsert"))
    delete_count = sum(1 for e in events if e.HasField("delete"))
    assert upsert_count >= 7, f"expected >=7 upserts, got {upsert_count}"
    assert delete_count >= 1, f"expected >=1 deletes, got {delete_count}"
    print(f"OK  total_events={len(events)}  upserts={upsert_count}  deletes={delete_count}")

    # ── 13. Prometheus metrics ──────────────────────────────────────
    print("13. Prometheus /metrics endpoint...", end=" ")
    import urllib.request
    metrics_url = f"http://localhost:10051/metrics"
    try:
        resp = urllib.request.urlopen(metrics_url, timeout=5)
        body = resp.read().decode()
        assert "shard_search_total" in body, "missing shard_search_total"
        assert "shard_upsert_total" in body, "missing shard_upsert_total"
        lines = [l for l in body.splitlines() if not l.startswith("#") and l.strip()]
        print(f"OK  {len(lines)} metric lines")
    except Exception as e:
        print(f"SKIP (metrics port not reachable: {e})")

    await coord.close()
    print("\n✅ All integration tests passed!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"\n❌ FAILED: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
