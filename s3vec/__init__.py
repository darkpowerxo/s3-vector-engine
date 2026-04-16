"""S3-Native Vector Engine — sub-10ms vector search on object storage.

Python + Rust hybrid: Python handles orchestration (FastAPI, async S3 I/O, Redis),
Rust handles the hot path (SIMD cosine similarity + top-k selection).
"""

try:
    from s3vec._engine import cosine_topk, cosine_topk_batch  # noqa: F401
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
