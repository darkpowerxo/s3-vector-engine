"""Ray Serve deployment configurations for 10+ model endpoints.

Each deployment defines autoscaling, resource requirements, and batch settings
per the KubeRay architecture in PRD §7. Deployments are isolated via custom
resources to prevent batch jobs from stealing GPU workers from Serve replicas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AutoscalingConfig:
    """Ray Serve autoscaling settings."""

    min_replicas: int = 1
    max_replicas: int = 5
    target_ongoing_requests: int = 5
    upscale_delay_s: float = 10.0
    downscale_delay_s: float = 60.0


@dataclass(frozen=True)
class DeploymentConfig:
    """Complete deployment specification for a model endpoint."""

    name: str
    extractor: str  # Key into extraction registry
    autoscaling: AutoscalingConfig = field(default_factory=AutoscalingConfig)
    num_cpus: float = 1.0
    num_gpus: float = 0.0
    memory: int = 2 * 1024**3  # 2 GB default
    batch_size: int = 32
    max_ongoing_requests: int = 10
    custom_resources: dict[str, float] = field(default_factory=dict)

    def to_ray_config(self) -> dict[str, Any]:
        """Convert to Ray Serve deployment kwargs."""
        config: dict[str, Any] = {
            "name": self.name,
            "autoscaling_config": {
                "min_replicas": self.autoscaling.min_replicas,
                "max_replicas": self.autoscaling.max_replicas,
                "target_ongoing_requests": self.autoscaling.target_ongoing_requests,
                "upscale_delay_s": self.autoscaling.upscale_delay_s,
                "downscale_delay_s": self.autoscaling.downscale_delay_s,
            },
            "ray_actor_options": {
                "num_cpus": self.num_cpus,
                "memory": self.memory,
            },
            "max_ongoing_requests": self.max_ongoing_requests,
        }
        if self.num_gpus > 0:
            config["ray_actor_options"]["num_gpus"] = self.num_gpus
        if self.custom_resources:
            config["ray_actor_options"]["resources"] = self.custom_resources
        return config

    def to_actor_pool_kwargs(self) -> dict[str, Any]:
        """Convert to Ray Data ActorPoolStrategy kwargs for batch processing."""
        opts: dict[str, Any] = {
            "num_cpus": self.num_cpus,
            "memory": self.memory,
        }
        if self.num_gpus > 0:
            opts["num_gpus"] = self.num_gpus
        if self.custom_resources:
            opts.update(self.custom_resources)
        return {
            "batch_size": self.batch_size,
            "ray_actor_options": opts,
        }


# ── Deployment Catalog ──────────────────────────────────────────────────────

DEPLOYMENTS: dict[str, DeploymentConfig] = {
    # -- Text (CPU) --
    "text_embedder": DeploymentConfig(
        name="text_embedder",
        extractor="e5_large",
        autoscaling=AutoscalingConfig(min_replicas=2, max_replicas=10, target_ongoing_requests=5),
        num_cpus=1,
        memory=2 * 1024**3,
        batch_size=64,
    ),
    "text_embedder_multilingual": DeploymentConfig(
        name="text_embedder_multilingual",
        extractor="bge_m3",
        autoscaling=AutoscalingConfig(min_replicas=1, max_replicas=5, target_ongoing_requests=3),
        num_cpus=1,
        memory=4 * 1024**3,
        batch_size=32,
    ),

    # -- Image (GPU 0.25) --
    "image_embedder": DeploymentConfig(
        name="image_embedder",
        extractor="siglip",
        autoscaling=AutoscalingConfig(min_replicas=1, max_replicas=5, target_ongoing_requests=3),
        num_cpus=1,
        num_gpus=0.25,
        memory=4 * 1024**3,
        batch_size=16,
    ),
    "image_embedder_clip": DeploymentConfig(
        name="image_embedder_clip",
        extractor="clip",
        autoscaling=AutoscalingConfig(min_replicas=0, max_replicas=3, target_ongoing_requests=3),
        num_cpus=1,
        num_gpus=0.25,
        memory=4 * 1024**3,
        batch_size=16,
    ),
    "image_embedder_dino": DeploymentConfig(
        name="image_embedder_dino",
        extractor="dinov2",
        autoscaling=AutoscalingConfig(min_replicas=0, max_replicas=3, target_ongoing_requests=3),
        num_cpus=1,
        num_gpus=0.25,
        memory=4 * 1024**3,
        batch_size=16,
    ),

    # -- Face (GPU 0.25) --
    "face_detector": DeploymentConfig(
        name="face_detector",
        extractor="arcface",
        autoscaling=AutoscalingConfig(min_replicas=1, max_replicas=5, target_ongoing_requests=3),
        num_cpus=1,
        num_gpus=0.25,
        memory=4 * 1024**3,
        batch_size=8,
    ),

    # -- Audio (GPU 0.5) --
    "asr_model": DeploymentConfig(
        name="asr_model",
        extractor="whisper",
        autoscaling=AutoscalingConfig(min_replicas=0, max_replicas=3, target_ongoing_requests=2),
        num_cpus=1,
        num_gpus=0.5,
        memory=8 * 1024**3,
        batch_size=4,
    ),
    "audio_embedder": DeploymentConfig(
        name="audio_embedder",
        extractor="clap",
        autoscaling=AutoscalingConfig(min_replicas=0, max_replicas=3, target_ongoing_requests=3),
        num_cpus=1,
        num_gpus=0.25,
        memory=4 * 1024**3,
        batch_size=8,
    ),

    # -- Video (GPU 0.5) --
    "video_embedder": DeploymentConfig(
        name="video_embedder",
        extractor="vertex_multimodal",
        autoscaling=AutoscalingConfig(min_replicas=0, max_replicas=3, target_ongoing_requests=1),
        num_cpus=2,
        num_gpus=0.5,
        memory=8 * 1024**3,
        batch_size=2,
        custom_resources={"batch": 1},  # batch/serve isolation
    ),

    # -- Reranker (GPU 0.25) --
    "reranker": DeploymentConfig(
        name="reranker",
        extractor="e5_large",  # placeholder — use cross-encoder in production
        autoscaling=AutoscalingConfig(min_replicas=0, max_replicas=3, target_ongoing_requests=5),
        num_cpus=1,
        num_gpus=0.25,
        memory=4 * 1024**3,
        batch_size=16,
    ),

    # -- LLM Judge (GPU 1.0) --
    "llm_judge": DeploymentConfig(
        name="llm_judge",
        extractor="e5_large",  # placeholder — use Gemma/Llama in production
        autoscaling=AutoscalingConfig(min_replicas=0, max_replicas=2, target_ongoing_requests=2),
        num_cpus=2,
        num_gpus=1.0,
        memory=16 * 1024**3,
        batch_size=4,
    ),
}


def get_deployment(name: str) -> DeploymentConfig:
    """Get deployment config by name."""
    if name not in DEPLOYMENTS:
        available = ", ".join(sorted(DEPLOYMENTS.keys()))
        raise KeyError(f"Unknown deployment: {name!r}. Available: [{available}]")
    return DEPLOYMENTS[name]


def list_deployments() -> dict[str, DeploymentConfig]:
    """List all deployment configurations."""
    return dict(DEPLOYMENTS)


def build_serve_config(deployment_names: list[str] | None = None) -> dict[str, Any]:
    """Build a Ray Serve application config for the specified deployments.

    If ``deployment_names`` is None, includes all deployments.
    Returns a dict suitable for ``serve.run()`` or YAML serialization.
    """
    names = deployment_names or list(DEPLOYMENTS.keys())
    applications: list[dict] = []

    for name in names:
        dep = get_deployment(name)
        cfg = dep.to_ray_config()
        cfg["import_path"] = f"s3vec.ray.serve_app:{name}"
        applications.append(cfg)

    return {
        "proxy_location": "EveryNode",
        "http_options": {"host": "0.0.0.0", "port": 8000},
        "applications": applications,
    }
