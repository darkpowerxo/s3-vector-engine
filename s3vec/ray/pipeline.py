"""Ray Data batch extraction pipeline.

Implements the three-phase pipeline from PRD §7.3:
1. Preprocessing — content loading and normalization (runs once)
2. Fan-out — parallel extraction across model-specific actor pools
3. Write — distributed shard writes via ShardDatasink

Designed to work with or without a live Ray cluster.
Without Ray, falls back to sequential local execution.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from s3vec.extraction import (
    ExtractionResult,
    get_extractor,
    list_extractors,
)
from s3vec.extraction.feature_uri import FeatureURI
from s3vec.ray.progress import JobStatus, ProgressActor
from s3vec.ray.serve_config import DeploymentConfig, get_deployment

logger = logging.getLogger(__name__)


@dataclass
class ExtractorConfig:
    """Configuration for a single extractor in the pipeline."""

    name: str  # extractor registry key
    deployment: str | None = None  # Ray Serve deployment name (optional)
    batch_size: int = 32
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    """Configuration for a full extraction pipeline run."""

    namespace: str
    extractors: list[ExtractorConfig]
    shard_addresses: list[str]
    job_id: str = field(default_factory=lambda: f"job-{uuid.uuid4().hex[:12]}")
    write_batch_size: int = 100


def run_extraction_pipeline_local(
    items: list[dict[str, Any]],
    config: PipelineConfig,
    progress: ProgressActor | None = None,
) -> dict[str, Any]:
    """Run extraction pipeline locally (no Ray cluster required).

    Each item dict must have:
    - ``id``: Unique document ID
    - ``content``: Raw bytes
    - ``content_type``: MIME type (e.g. "image/jpeg", "audio/wav")

    Returns pipeline run summary with per-extractor results.
    """
    job_id = config.job_id
    total_extractions = len(items) * len(config.extractors)

    if progress:
        progress.start_job(job_id, total=total_extractions, stage="initializing")

    start = time.time()
    all_results: list[ExtractionResult] = []
    features_for_write: list[dict[str, Any]] = []

    # Phase 1: Preprocessing — normalize items
    if progress:
        progress.set_stage(job_id, "preprocessing")

    preprocessed = []
    for item in items:
        preprocessed.append({
            "id": item["id"],
            "content": item["content"],
            "content_type": item.get("content_type", "application/octet-stream"),
            "metadata": item.get("metadata", {}),
        })

    # Phase 2: Fan-out to extractors
    if progress:
        progress.set_stage(job_id, "extracting")

    for ext_cfg in config.extractors:
        extractor_cls = get_extractor(ext_cfg.name)
        extractor = extractor_cls(**ext_cfg.kwargs)

        for item in preprocessed:
            result = ExtractionResult(source_id=item["id"])
            try:
                features = extractor.extract(
                    item["content"], item["content_type"]
                )
                result.features = features

                # Prepare features for shard write
                for feat in features:
                    if feat.vector is not None:
                        features_for_write.append({
                            "id": f"{item['id']}:{feat.uri}",
                            "vector": feat.vector,
                            "payload": {
                                "source_id": item["id"],
                                "feature_uri": str(feat.uri),
                                "extractor": ext_cfg.name,
                                **(feat.metadata or {}),
                                **({"text": feat.text} if feat.text else {}),
                                **(
                                    {"timestamp_ms": feat.timestamp_ms}
                                    if feat.timestamp_ms is not None
                                    else {}
                                ),
                            },
                            "text_fields": (
                                {"content": feat.text} if feat.text else None
                            ),
                        })
            except Exception as e:
                result.errors.append(f"{ext_cfg.name}: {e}")
                if progress:
                    progress.increment_failed(job_id)

            all_results.append(result)
            if progress:
                progress.increment(job_id)

    # Phase 3: Write to shards
    if progress:
        progress.set_stage(job_id, "writing")

    write_stats = {"written": 0, "failed": 0}
    if features_for_write and config.shard_addresses:
        from s3vec.ray.datasink import ShardDatasink

        sink = ShardDatasink(
            namespace=config.namespace,
            shard_addresses=config.shard_addresses,
            batch_size=config.write_batch_size,
        )
        sink.on_write_start()
        write_stats = sink.write(features_for_write)
        sink.on_write_complete()

    elapsed = time.time() - start

    if progress:
        progress.set_stage(job_id, "completed")

    summary = {
        "job_id": job_id,
        "status": "completed",
        "items_processed": len(items),
        "extractors_used": [e.name for e in config.extractors],
        "features_extracted": sum(len(r.features) for r in all_results),
        "features_written": write_stats.get("written", 0),
        "write_failures": write_stats.get("failed", 0),
        "errors": [
            {"source_id": r.source_id, "errors": r.errors}
            for r in all_results
            if not r.success
        ],
        "elapsed_s": round(elapsed, 3),
    }

    logger.info("Pipeline complete", extra=summary)
    return summary


def run_extraction_pipeline_ray(
    items: list[dict[str, Any]],
    config: PipelineConfig,
) -> str:
    """Run extraction pipeline on Ray cluster.

    Submits the pipeline as an async Ray job. Returns the job_id for polling.

    Requires a running Ray cluster. Items are converted to a Ray Dataset
    and processed via ``map_batches`` with ``ActorPoolStrategy``.
    """
    try:
        import ray
        from ray.data import ActorPoolStrategy
    except ImportError:
        raise RuntimeError(
            "Ray is required for distributed pipelines. "
            "Install with: pip install 'ray[data,serve]'"
        )

    job_id = config.job_id

    # Create or get ProgressActor
    try:
        progress = ray.get_actor("progress_actor")
    except ValueError:
        progress = ray.remote(ProgressActor).options(
            name="progress_actor",
            num_cpus=0,
            lifetime="detached",
        ).remote()

    # Build Ray Dataset
    ds = ray.data.from_items(items)
    total = ds.count()
    ray.get(progress.start_job.remote(job_id, total=total * len(config.extractors)))

    # Fan-out to extractors
    for ext_cfg in config.extractors:
        extractor_cls = get_extractor(ext_cfg.name)

        # Resolve deployment config for resource hints
        try:
            dep_config = get_deployment(ext_cfg.deployment or ext_cfg.name)
            actor_kwargs = dep_config.to_actor_pool_kwargs()
        except KeyError:
            actor_kwargs = {"batch_size": ext_cfg.batch_size}

        def make_extract_fn(ext_cls, ext_kwargs, progress_ref, jid, ext_name):
            class _ExtractFn:
                def __init__(self):
                    self._extractor = ext_cls(**ext_kwargs)

                def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
                    results = self._extractor.extract_batch(
                        [
                            {
                                "content": c,
                                "content_type": ct,
                                "text": t,
                            }
                            for c, ct, t in zip(
                                batch.get("content", []),
                                batch.get("content_type", []),
                                batch.get("text", batch.get("content", [])),
                            )
                        ]
                    )
                    # Fire-and-forget progress update
                    progress_ref.increment.remote(jid, len(results))
                    return batch

            return _ExtractFn

        extract_fn = make_extract_fn(
            extractor_cls, ext_cfg.kwargs, progress, job_id, ext_cfg.name
        )

        min_actors = 1
        max_actors = 8
        ds = ds.map_batches(
            extract_fn,
            compute=ActorPoolStrategy(size=min_actors),
            **{k: v for k, v in actor_kwargs.items() if k != "ray_actor_options"},
        )

    # Write to shards via datasink
    ray.get(progress.set_stage.remote(job_id, "writing"))

    from s3vec.ray.datasink import ShardDatasink

    sink = ShardDatasink(
        namespace=config.namespace,
        shard_addresses=config.shard_addresses,
        batch_size=config.write_batch_size,
    )

    # Materialize (triggers execution)
    for batch in ds.iter_batches(batch_size=config.write_batch_size):
        records = []
        for i in range(len(batch.get("id", []))):
            records.append({
                "id": batch["id"][i],
                "vector": batch.get("vector", [None])[i],
                "payload": batch.get("payload", [None])[i],
            })
        sink.write(records)

    ray.get(progress.set_stage.remote(job_id, "completed"))
    return job_id
