"""ShardDatasink — Ray Data Datasink for distributed gRPC writes.

Workers write extracted features directly to Rust shard servers via gRPC,
eliminating single-node bottlenecks. Features are routed through the
consistent hash ring to the correct shard.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable

logger = logging.getLogger(__name__)


class ShardDatasink:
    """Ray Data Datasink that writes features to shard cluster via gRPC.

    Usage with Ray Data::

        ds.write_datasink(ShardDatasink(
            namespace="my_namespace",
            shard_addresses=["localhost:9051", "localhost:9052"],
        ))
    """

    def __init__(
        self,
        namespace: str,
        shard_addresses: list[str],
        batch_size: int = 100,
        timeout_seconds: float = 30.0,
    ):
        self._namespace = namespace
        self._shard_addresses = shard_addresses
        self._batch_size = batch_size
        self._timeout = timeout_seconds

    def on_write_start(self) -> None:
        """Called once at the start of the write operation."""
        logger.info(
            "ShardDatasink started",
            extra={
                "namespace": self._namespace,
                "shard_count": len(self._shard_addresses),
            },
        )

    def write(self, blocks: Iterable[dict[str, Any]]) -> dict[str, int]:
        """Write a batch of feature records to shards via gRPC.

        Each record must contain:
        - ``id``: Document ID
        - ``vector``: Dense vector (list[float]) or None
        - ``sparse_indices``/``sparse_values``: Sparse vector or None
        - ``payload``: JSON-serializable metadata dict or None
        - ``text_fields``: dict[str, str] for BM25 indexing or None

        Returns write stats: {written, failed, shards_hit}.
        """
        import grpc
        import orjson

        from s3vec.proto import shard_pb2, shard_pb2_grpc

        # Build per-shard batches
        from s3vec.grpc_coordinator import ConsistentHashRing

        ring = ConsistentHashRing()
        for addr in self._shard_addresses:
            ring.add_node(addr)

        shard_batches: dict[str, list[dict]] = {a: [] for a in self._shard_addresses}
        total = 0

        for record in blocks:
            doc_id = record["id"]
            target = ring.get_node(doc_id)
            if target:
                shard_batches[target].append(record)
                total += 1

        # Write to each shard
        written = 0
        failed = 0
        shards_hit = 0

        for addr, batch in shard_batches.items():
            if not batch:
                continue
            shards_hit += 1

            try:
                channel = grpc.insecure_channel(addr)
                stub = shard_pb2_grpc.ShardServiceStub(channel)

                for record in batch:
                    dense = None
                    if record.get("vector"):
                        dense = shard_pb2.DenseVector(values=record["vector"])

                    sparse = None
                    if record.get("sparse_indices"):
                        sparse = shard_pb2.SparseVectorMsg(
                            indices=record["sparse_indices"],
                            values=record.get("sparse_values", []),
                        )

                    payload_bytes = None
                    if record.get("payload"):
                        payload_bytes = orjson.dumps(record["payload"])

                    text_fields = record.get("text_fields") or {}

                    req = shard_pb2.UpsertRequest(
                        id=record["id"],
                        dense_vector=dense,
                        sparse_vector=sparse,
                        text_fields=text_fields,
                        payload=payload_bytes,
                    )
                    stub.Upsert(req, timeout=self._timeout)
                    written += 1

                channel.close()
            except Exception:
                logger.exception("ShardDatasink write failed", extra={"shard": addr})
                failed += len(batch)

        return {"written": written, "failed": failed, "shards_hit": shards_hit}

    def on_write_complete(self) -> None:
        """Called once after all writes complete."""
        logger.info("ShardDatasink complete", extra={"namespace": self._namespace})

    @property
    def num_rows_per_write(self) -> int:
        return self._batch_size
