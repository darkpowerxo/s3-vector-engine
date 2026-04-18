"""Configuration for the S3-native vector engine."""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # S3 / MinIO
    s3_endpoint: str = "http://localhost:9000"
    s3_access_key: str = "minioadmin"
    s3_secret_key: str = "minioadmin"
    s3_bucket: str = "vector-store"
    s3_region: str = "us-east-1"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Engine tuning
    shard_size: int = 10_000          # vectors per shard
    vector_dimensions: int = 1024     # embedding dimensions
    top_k_default: int = 10           # default results per query
    max_concurrent_shards: int = 32   # max parallel S3 fetches
    shard_timeout_seconds: float = 10.0  # per-shard fetch+scan timeout
    query_timeout_seconds: float = 30.0  # total query timeout

    # gRPC shard servers (comma-separated)
    shard_addresses: str = "localhost:9051,localhost:9052,localhost:9053"

    # mTLS (set all three to enable)
    tls_cert: str = ""        # client cert PEM path
    tls_key: str = ""         # client key PEM path
    tls_ca: str = ""          # CA cert PEM path

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1                  # uvicorn workers (set >1 behind LB)
    log_level: str = "info"
    log_json: bool = False            # JSON structured logging for prod
    cors_origins: str = "*"           # comma-separated origins, or *

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache()
def get_settings() -> Settings:
    return Settings()
