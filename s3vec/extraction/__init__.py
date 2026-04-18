"""Multimodal feature extraction framework.

Base classes, model registry, and feature URI system for extracting
searchable features from multimodal content (text, image, video, audio, face).
"""

from __future__ import annotations

import abc
import enum
from dataclasses import dataclass, field
from typing import Any

from .feature_uri import FeatureURI


class Modality(str, enum.Enum):
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    FACE = "face"
    MULTIMODAL = "multimodal"


@dataclass(frozen=True)
class ModelSpec:
    """Specification for an embedding model."""

    name: str
    modality: Modality
    dimensions: int | None  # None for non-embedding models (e.g. Whisper ASR)
    version: str = "v1"
    description: str = ""

    @property
    def uri_prefix(self) -> str:
        return f"s3vec://{self.name}@{self.version}"


@dataclass
class ExtractedFeature:
    """A single extracted feature from content."""

    uri: FeatureURI
    vector: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    text: str | None = None
    timestamp_ms: float | None = None


@dataclass
class ExtractionResult:
    """Result of extracting features from a single content item."""

    source_id: str
    features: list[ExtractedFeature] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return len(self.errors) == 0


class BaseExtractor(abc.ABC):
    """Base class for all feature extractors.

    Subclasses are deployed as Ray Serve replicas. The ``__init__`` method
    loads model weights (called once per replica), and ``extract_batch``
    is invoked by ``ray.data.Dataset.map_batches``.
    """

    model_spec: ModelSpec

    @abc.abstractmethod
    def extract(
        self, content: bytes, content_type: str, **kwargs: Any
    ) -> list[ExtractedFeature]:
        """Extract features from a single content item."""
        ...

    @abc.abstractmethod
    def extract_batch(
        self, batch: list[dict[str, Any]]
    ) -> list[list[ExtractedFeature]]:
        """Extract features from a batch. Used by Ray Data map_batches."""
        ...

    def get_feature_uri(self, feature_type: str) -> FeatureURI:
        """Build a FeatureURI for this extractor."""
        return FeatureURI(
            extractor=self.model_spec.name,
            version=self.model_spec.version,
            feature_type=feature_type,
        )


# ── Model Registry ──────────────────────────────────────────────────────────

_REGISTRY: dict[str, type[BaseExtractor]] = {}


def register_extractor(name: str):
    """Decorator to register an extractor class."""

    def wrapper(cls: type[BaseExtractor]) -> type[BaseExtractor]:
        _REGISTRY[name] = cls
        return cls

    return wrapper


def get_extractor(name: str) -> type[BaseExtractor]:
    """Get an extractor class by name."""
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise KeyError(f"Unknown extractor: {name!r}. Available: [{available}]")
    return _REGISTRY[name]


def list_extractors() -> dict[str, type[BaseExtractor]]:
    """List all registered extractors."""
    return dict(_REGISTRY)


# ── Supported Models ────────────────────────────────────────────────────────

MODELS: dict[str, ModelSpec] = {
    "siglip": ModelSpec(
        "siglip", Modality.IMAGE, 1152,
        description="SigLIP SO400M — visual search, image-text retrieval",
    ),
    "clip": ModelSpec(
        "clip", Modality.IMAGE, 768,
        description="CLIP ViT-L/14 — cross-modal image-text retrieval",
    ),
    "e5_large": ModelSpec(
        "e5_large", Modality.TEXT, 1024,
        description="E5-Large-v2 — English text semantic search",
    ),
    "bge_m3": ModelSpec(
        "bge_m3", Modality.TEXT, 1024,
        description="BGE-M3 — multilingual text, dense + sparse",
    ),
    "dinov2": ModelSpec(
        "dinov2", Modality.IMAGE, 768,
        description="DINOv2 — visual features without text alignment",
    ),
    "arcface": ModelSpec(
        "arcface", Modality.FACE, 512,
        description="ArcFace — face identity recognition",
    ),
    "clap": ModelSpec(
        "clap", Modality.AUDIO, 512,
        description="CLAP — audio-text cross-modal",
    ),
    "whisper": ModelSpec(
        "whisper", Modality.AUDIO, None,
        description="Whisper Large v3 — ASR transcription",
    ),
    "vertex_multimodal": ModelSpec(
        "vertex_multimodal", Modality.MULTIMODAL, 1408,
        description="Vertex AI — unified multimodal embedding",
    ),
    "colqwen2": ModelSpec(
        "colqwen2", Modality.MULTIMODAL, 128,
        description="ColQwen2 — late interaction, 128d per patch",
    ),
}


# ── Auto-register extractors on import ──────────────────────────────────────
# Import submodules so @register_extractor decorators fire.
from . import text as _text  # noqa: F401, E402
from . import image as _image  # noqa: F401, E402
from . import face as _face  # noqa: F401, E402
from . import audio as _audio  # noqa: F401, E402
from . import video as _video  # noqa: F401, E402
