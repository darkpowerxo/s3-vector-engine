"""gRPC Coordinator — fan-out/gather across Rust shard servers.

Routes queries to Rust shard gRPC servers using consistent hashing,
fans out in parallel, and fuses cross-shard results.

This replaces the old HTTP/S3-based coordinator with gRPC.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field

import grpc
import orjson

from .proto import shard_pb2, shard_pb2_grpc

logger = logging.getLogger(__name__)


# ── Consistent Hash Ring ────────────────────────────────────────────────────


class ConsistentHashRing:
    """Consistent hash ring for namespace → shard routing.

    Each shard gets `vnodes` virtual nodes on the ring for even distribution.
    """

    def __init__(self, vnodes: int = 128):
        self._vnodes = vnodes
        self._ring: dict[int, str] = {}
        self._sorted_keys: list[int] = []
        self._nodes: set[str] = set()

    def _hash(self, key: str) -> int:
        return int(hashlib.md5(key.encode()).hexdigest(), 16)

    def add_node(self, node: str) -> None:
        """Add a shard node (address like 'host:port') to the ring."""
        self._nodes.add(node)
        for i in range(self._vnodes):
            vnode_key = self._hash(f"{node}:{i}")
            self._ring[vnode_key] = node
        self._sorted_keys = sorted(self._ring.keys())

    def remove_node(self, node: str) -> None:
        """Remove a shard node from the ring."""
        self._nodes.discard(node)
        for i in range(self._vnodes):
            vnode_key = self._hash(f"{node}:{i}")
            self._ring.pop(vnode_key, None)
        self._sorted_keys = sorted(self._ring.keys())

    def get_node(self, key: str) -> str | None:
        """Get the shard responsible for a given key (namespace)."""
        if not self._ring:
            return None
        h = self._hash(key)
        for k in self._sorted_keys:
            if k >= h:
                return self._ring[k]
        return self._ring[self._sorted_keys[0]]

    def get_nodes(self, key: str, n: int) -> list[str]:
        """Get n distinct shard nodes for a key (for replication/fan-out)."""
        if not self._ring:
            return []
        result: list[str] = []
        h = self._hash(key)
        idx = 0
        for i, k in enumerate(self._sorted_keys):
            if k >= h:
                idx = i
                break

        visited: set[str] = set()
        total = len(self._sorted_keys)
        for i in range(total):
            node = self._ring[self._sorted_keys[(idx + i) % total]]
            if node not in visited:
                visited.add(node)
                result.append(node)
                if len(result) >= n:
                    break
        return result

    @property
    def nodes(self) -> set[str]:
        return self._nodes.copy()


# ── gRPC Channel Pool ──────────────────────────────────────────────────────


class ShardChannelPool:
    """Maintains a pool of gRPC channels to shard servers."""

    def __init__(self, tls_cert: str = "", tls_key: str = "", tls_ca: str = ""):
        self._channels: dict[str, grpc.aio.Channel] = {}
        self._stubs: dict[str, shard_pb2_grpc.ShardServiceStub] = {}
        self._credentials = None
        if tls_cert and tls_key and tls_ca:
            with open(tls_ca, "rb") as f:
                ca_cert = f.read()
            with open(tls_cert, "rb") as f:
                client_cert = f.read()
            with open(tls_key, "rb") as f:
                client_key = f.read()
            self._credentials = grpc.ssl_channel_credentials(
                root_certificates=ca_cert,
                private_key=client_key,
                certificate_chain=client_cert,
            )

    def get_stub(self, address: str) -> shard_pb2_grpc.ShardServiceStub:
        """Get or create a gRPC stub for a shard address."""
        if address not in self._stubs:
            if self._credentials:
                channel = grpc.aio.secure_channel(address, self._credentials)
            else:
                channel = grpc.aio.insecure_channel(address)
            self._channels[address] = channel
            self._stubs[address] = shard_pb2_grpc.ShardServiceStub(channel)
        return self._stubs[address]

    async def close(self) -> None:
        """Close all channels."""
        for ch in self._channels.values():
            await ch.close()
        self._channels.clear()
        self._stubs.clear()


# ── Search Result ───────────────────────────────────────────────────────────


@dataclass
class SearchResult:
    """Global search result after merging all shard results."""

    results: list[dict] = field(default_factory=list)
    latency_ms: float = 0.0
    shards_queried: int = 0
    shards_failed: int = 0
    total_duration_us: int = 0
    errors: list[dict] = field(default_factory=list)


# ── Coordinator ─────────────────────────────────────────────────────────────


class GrpcCoordinator:
    """Coordinates queries across multiple Rust shard gRPC servers.

    Supports:
    - Consistent-hash routing (namespace → shard assignment)
    - Parallel fan-out to multiple shards
    - Cross-shard result fusion (RRF, DBSF, Linear)
    - Health checking via GetStats
    """

    def __init__(
        self,
        shard_addresses: list[str] | None = None,
        timeout_seconds: float = 10.0,
        tls_cert: str = "",
        tls_key: str = "",
        tls_ca: str = "",
    ):
        self.ring = ConsistentHashRing()
        self.pool = ShardChannelPool(tls_cert=tls_cert, tls_key=tls_key, tls_ca=tls_ca)
        self.timeout = timeout_seconds

        if shard_addresses:
            for addr in shard_addresses:
                self.ring.add_node(addr)

    def add_shard(self, address: str) -> None:
        """Add a shard server to the routing ring."""
        self.ring.add_node(address)

    def remove_shard(self, address: str) -> None:
        """Remove a shard server from the routing ring."""
        self.ring.remove_node(address)

    async def search(
        self,
        *,
        namespace: str = "default",
        dense_vector: list[float] | None = None,
        sparse_indices: list[int] | None = None,
        sparse_values: list[float] | None = None,
        text_query: str | None = None,
        top_k: int = 10,
        fusion: str = "rrf",
        include_payloads: bool = False,
        filter_expr: dict | None = None,
    ) -> SearchResult:
        """Execute a distributed search across relevant shards.

        Fan-out to all shards (for simplicity; could be namespace-scoped
        via consistent hash). Merges results with RRF.
        """
        t0 = time.perf_counter()

        # Build the proto request
        request = shard_pb2.SearchRequest(
            top_k=top_k,
            include_payloads=include_payloads,
        )

        # Set fusion strategy
        fusion_map = {"rrf": 0, "dbsf": 1, "linear": 2}
        request.fusion = fusion_map.get(fusion, 0)

        # Dense query
        if dense_vector is not None:
            request.dense.CopyFrom(
                shard_pb2.DenseQuery(vector=dense_vector)
            )

        # Sparse query
        if sparse_indices is not None and sparse_values is not None:
            request.sparse.CopyFrom(
                shard_pb2.SparseQuery(
                    indices=sparse_indices,
                    values=sparse_values,
                )
            )

        # BM25 query
        if text_query is not None:
            request.bm25_query = text_query

        # Filter
        if filter_expr is not None:
            pb_filter = _dict_to_filter_expression(filter_expr)
            if pb_filter is not None:
                request.filter.CopyFrom(pb_filter)

        # Fan out to all shard nodes
        targets = list(self.ring.nodes)
        if not targets:
            return SearchResult(latency_ms=(time.perf_counter() - t0) * 1000)

        # Parallel gRPC calls
        tasks = []
        for addr in targets:
            tasks.append(self._search_shard(addr, request))

        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Merge results
        all_docs: list[dict] = []
        errors: list[dict] = []
        shards_ok = 0
        total_duration_us = 0

        for addr, result in zip(targets, raw_results):
            if isinstance(result, Exception):
                errors.append({"shard": addr, "error": str(result)})
                continue
            shards_ok += 1
            total_duration_us += result.duration_us
            for doc in result.results:
                entry = {"id": doc.id, "score": doc.score, "shard": addr}
                if doc.payload:
                    entry["payload"] = orjson.loads(doc.payload)
                all_docs.append(entry)

        # Sort by score descending, take top_k
        all_docs.sort(key=lambda d: d["score"], reverse=True)
        all_docs = all_docs[:top_k]

        latency_ms = (time.perf_counter() - t0) * 1000

        return SearchResult(
            results=all_docs,
            latency_ms=latency_ms,
            shards_queried=len(targets),
            shards_failed=len(errors),
            total_duration_us=total_duration_us,
            errors=errors,
        )

    async def upsert(
        self,
        *,
        namespace: str = "default",
        id: str,
        dense_vector: list[float] | None = None,
        sparse_indices: list[int] | None = None,
        sparse_values: list[float] | None = None,
        text_fields: dict[str, str] | None = None,
        payload: dict | None = None,
    ) -> dict:
        """Upsert a document to the appropriate shard."""
        target = self.ring.get_node(namespace)
        if target is None:
            raise RuntimeError("No shards available")

        request = shard_pb2.UpsertRequest(id=id)

        if dense_vector is not None:
            request.dense_vector.CopyFrom(
                shard_pb2.DenseVector(values=dense_vector)
            )

        if sparse_indices is not None and sparse_values is not None:
            request.sparse_vector.CopyFrom(
                shard_pb2.SparseVectorMsg(
                    indices=sparse_indices,
                    values=sparse_values,
                )
            )

        if text_fields:
            for k, v in text_fields.items():
                request.text_fields[k] = v

        if payload is not None:
            request.payload = orjson.dumps(payload)

        stub = self.pool.get_stub(target)
        response = await stub.Upsert(request, timeout=self.timeout)

        return {
            "wal_sequence": response.wal_sequence,
            "shard": target,
        }

    async def delete(
        self,
        *,
        namespace: str = "default",
        id: str,
    ) -> dict:
        """Delete a document from the appropriate shard."""
        target = self.ring.get_node(namespace)
        if target is None:
            raise RuntimeError("No shards available")

        request = shard_pb2.DeleteRequest(id=id)
        stub = self.pool.get_stub(target)
        response = await stub.Delete(request, timeout=self.timeout)

        return {
            "wal_sequence": response.wal_sequence,
            "found": response.found,
            "shard": target,
        }

    async def get_stats(self, address: str) -> dict:
        """Get stats from a specific shard."""
        stub = self.pool.get_stub(address)
        response = await stub.GetStats(
            shard_pb2.StatsRequest(), timeout=self.timeout
        )
        return {
            "dense_count": response.dense_count,
            "sparse_count": response.sparse_count,
            "sparse_vocab_size": response.sparse_vocab_size,
            "payload_count": response.payload_count,
            "wal_sequence": response.wal_sequence,
            "dim": response.dim,
        }

    async def get_all_stats(self) -> dict[str, dict]:
        """Get stats from all shards."""
        tasks = {
            addr: self.get_stats(addr) for addr in self.ring.nodes
        }
        results = {}
        for addr, task in tasks.items():
            try:
                results[addr] = await task
            except Exception as e:
                results[addr] = {"error": str(e)}
        return results

    async def close(self) -> None:
        """Close all gRPC channels."""
        await self.pool.close()

    # ── Internal ────────────────────────────────────────────────────────────

    async def _search_shard(
        self, address: str, request: shard_pb2.SearchRequest
    ) -> shard_pb2.SearchResponse:
        """Send a search to a single shard with timeout."""
        stub = self.pool.get_stub(address)
        return await stub.Search(request, timeout=self.timeout)


# ── Filter Expression Builder ──────────────────────────────────────────────


def _dict_to_filter_expression(
    d: dict,
) -> shard_pb2.FilterExpression | None:
    """Convert a dict filter description to a protobuf FilterExpression.

    Supported formats:
        {"field": "status", "op": "eq", "value": "active"}
        {"and": [{"field": "x", "op": "gt", "value": 5}, ...]}
        {"or": [...]}
        {"not": {...}}
    """
    if "field" in d:
        op_map = {
            "eq": shard_pb2.EQ,
            "ne": shard_pb2.NE,
            "gt": shard_pb2.GT,
            "gte": shard_pb2.GTE,
            "lt": shard_pb2.LT,
            "lte": shard_pb2.LTE,
            "in": shard_pb2.IN,
            "contains": shard_pb2.CONTAINS,
        }
        op_str = d.get("op", "eq").lower()
        op = op_map.get(op_str, shard_pb2.EQ)
        value_json = orjson.dumps(d.get("value", "")).decode()

        return shard_pb2.FilterExpression(
            field=shard_pb2.FieldFilter(
                field_name=d["field"],
                op=op,
                value_json=value_json,
            )
        )

    for composite_op, pb_op in [
        ("and", shard_pb2.AND),
        ("or", shard_pb2.OR),
    ]:
        if composite_op in d:
            children = [
                _dict_to_filter_expression(c) for c in d[composite_op]
            ]
            children = [c for c in children if c is not None]
            return shard_pb2.FilterExpression(
                composite=shard_pb2.CompositeFilter(
                    op=pb_op, children=children
                )
            )

    if "not" in d:
        child = _dict_to_filter_expression(d["not"])
        if child is not None:
            return shard_pb2.FilterExpression(
                composite=shard_pb2.CompositeFilter(
                    op=shard_pb2.NOT, children=[child]
                )
            )

    return None
