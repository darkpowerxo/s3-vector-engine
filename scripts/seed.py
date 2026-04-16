#!/usr/bin/env python3
"""Seed the vector engine with test data.

Generates random vectors, shards them, and uploads to MinIO.
Simulates translation memory embeddings at various scales.

Usage:
    python scripts/seed.py --num-vectors 100000 --dimensions 1024 --shard-size 10000
    python scripts/seed.py --num-vectors 1000000 --tenant-id bank_client_a
"""

import argparse
import asyncio
import time
import sys
import os

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from s3vec.config import get_settings
from s3vec.storage import ensure_bucket
from s3vec.indexer import bulk_index


def generate_vectors(num_vectors: int, dimensions: int) -> tuple[list[str], np.ndarray]:
    """Generate random normalized vectors simulating embeddings.

    In production, these would come from BGE-M3, OpenAI, etc.
    We normalize them to unit length (cosine similarity ready).
    """
    print(f"Generating {num_vectors:,} random vectors ({dimensions}d)...")
    vectors = np.random.randn(num_vectors, dimensions).astype(np.float32)

    # Normalize to unit vectors (standard for cosine similarity)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / norms

    # Generate IDs (simulating TM segment IDs)
    ids = [f"vec_{i:08d}" for i in range(num_vectors)]

    size_mb = (vectors.nbytes) / (1024 * 1024)
    print(f"  Total size: {size_mb:.1f} MB ({vectors.dtype})")
    print(f"  Per vector: {vectors.shape[1] * 4} bytes")

    return ids, vectors


async def main():
    parser = argparse.ArgumentParser(description="Seed the S3 vector engine")
    parser.add_argument(
        "--num-vectors", type=int, default=100_000,
        help="Number of vectors to generate (default: 100000)"
    )
    parser.add_argument(
        "--dimensions", type=int, default=1024,
        help="Vector dimensions (default: 1024, matching BGE-M3)"
    )
    parser.add_argument(
        "--shard-size", type=int, default=10_000,
        help="Vectors per shard (default: 10000)"
    )
    parser.add_argument(
        "--tenant-id", type=str, default="default",
        help="Tenant ID namespace (default: 'default')"
    )
    args = parser.parse_args()

    # Override settings if provided
    settings = get_settings()

    print("=" * 60)
    print("S3-Native Vector Engine — Seed Script")
    print("=" * 60)
    print(f"  S3 endpoint:  {settings.s3_endpoint}")
    print(f"  S3 bucket:    {settings.s3_bucket}")
    print(f"  Tenant:       {args.tenant_id}")
    print(f"  Vectors:      {args.num_vectors:,}")
    print(f"  Dimensions:   {args.dimensions}")
    print(f"  Shard size:   {args.shard_size}")
    print(f"  Shards:       {args.num_vectors // args.shard_size}")
    print()

    # Ensure bucket exists
    ensure_bucket()

    # Generate vectors
    ids, vectors = generate_vectors(args.num_vectors, args.dimensions)

    # Upload shards
    print(f"\nUploading {args.num_vectors // args.shard_size} shards to S3...")
    t0 = time.perf_counter()

    result = await bulk_index(
        tenant_id=args.tenant_id,
        ids=ids,
        vectors=vectors,
        shard_size=args.shard_size,
    )

    elapsed = time.perf_counter() - t0

    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Shards created:   {result['shards_created']}")
    print(f"  Vectors indexed:  {result['vectors_indexed']:,}")
    print(f"  Throughput:       {result['vectors_indexed'] / elapsed:,.0f} vectors/sec")

    # Print cost comparison
    vecs = args.num_vectors
    s3_cost_mo = (vecs * args.dimensions * 4) / (1024**3) * 0.023  # S3 Standard $/GB/mo
    pinecone_cost_mo = (vecs / 1_000_000) * 7  # rough estimate
    print(f"\n--- Cost Comparison (monthly) ---")
    print(f"  S3 storage:       ${s3_cost_mo:.2f}/mo")
    print(f"  Pinecone (est):   ${pinecone_cost_mo:.2f}/mo")
    print(f"  Savings:          {(1 - s3_cost_mo/max(pinecone_cost_mo, 0.01))*100:.0f}%")


if __name__ == "__main__":
    asyncio.run(main())
