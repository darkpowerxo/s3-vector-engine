"""Image feature extractors — SigLIP, CLIP, DINOv2."""

from __future__ import annotations

import io
from typing import Any

from . import (
    BaseExtractor,
    ExtractedFeature,
    MODELS,
    register_extractor,
)


@register_extractor("siglip")
class SigLIPExtractor(BaseExtractor):
    """SigLIP SO400M — visual search, image-text retrieval (1152d).

    Deployed as GPU Ray Serve replica (0.25 GPU fraction).
    Handles both image and text inputs for cross-modal retrieval.
    """

    model_spec = MODELS["siglip"]

    def __init__(
        self,
        model_name: str = "google/siglip-so400m-patch14-384",
        device: str = "cuda",
    ):
        self._model_name = model_name
        self._device = device
        self._model = None
        self._processor = None

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ImportError:
            raise RuntimeError(
                "Install transformers + torch: pip install transformers torch"
            )
        self._processor = AutoProcessor.from_pretrained(self._model_name)
        self._model = AutoModel.from_pretrained(self._model_name).to(self._device)
        self._model.eval()

    def _embed_images(self, images: list) -> list[list[float]]:
        import torch

        self._load_model()
        inputs = self._processor(images=images, return_tensors="pt").to(self._device)
        with torch.no_grad():
            feats = self._model.get_image_features(**inputs)
        feats = torch.nn.functional.normalize(feats, p=2, dim=1)
        return feats.cpu().tolist()

    def extract(
        self, content: bytes, content_type: str, **kwargs: Any
    ) -> list[ExtractedFeature]:
        from PIL import Image

        img = Image.open(io.BytesIO(content)).convert("RGB")
        vectors = self._embed_images([img])
        return [
            ExtractedFeature(
                uri=self.get_feature_uri("embedding"),
                vector=vectors[0],
                metadata={"content_type": content_type},
            )
        ]

    def extract_batch(
        self, batch: list[dict[str, Any]]
    ) -> list[list[ExtractedFeature]]:
        from PIL import Image

        images = [
            Image.open(io.BytesIO(item["content"])).convert("RGB")
            for item in batch
        ]
        vectors = self._embed_images(images)
        results = []
        for item, vec in zip(batch, vectors):
            results.append([
                ExtractedFeature(
                    uri=self.get_feature_uri("embedding"),
                    vector=vec,
                    metadata={"content_type": item.get("content_type", "image/jpeg")},
                )
            ])
        return results


@register_extractor("clip")
class CLIPExtractor(BaseExtractor):
    """CLIP ViT-L/14 — cross-modal image-text retrieval (768d)."""

    model_spec = MODELS["clip"]

    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        device: str = "cuda",
    ):
        self._model_name = model_name
        self._device = device
        self._model = None
        self._processor = None

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError:
            raise RuntimeError(
                "Install transformers + torch: pip install transformers torch"
            )
        self._processor = CLIPProcessor.from_pretrained(self._model_name)
        self._model = CLIPModel.from_pretrained(self._model_name).to(self._device)
        self._model.eval()

    def _embed_images(self, images: list) -> list[list[float]]:
        import torch

        self._load_model()
        inputs = self._processor(images=images, return_tensors="pt").to(self._device)
        with torch.no_grad():
            feats = self._model.get_image_features(**inputs)
        feats = torch.nn.functional.normalize(feats, p=2, dim=1)
        return feats.cpu().tolist()

    def extract(
        self, content: bytes, content_type: str, **kwargs: Any
    ) -> list[ExtractedFeature]:
        from PIL import Image

        img = Image.open(io.BytesIO(content)).convert("RGB")
        vectors = self._embed_images([img])
        return [
            ExtractedFeature(
                uri=self.get_feature_uri("embedding"),
                vector=vectors[0],
                metadata={"content_type": content_type},
            )
        ]

    def extract_batch(
        self, batch: list[dict[str, Any]]
    ) -> list[list[ExtractedFeature]]:
        from PIL import Image

        images = [
            Image.open(io.BytesIO(item["content"])).convert("RGB")
            for item in batch
        ]
        vectors = self._embed_images(images)
        results = []
        for item, vec in zip(batch, vectors):
            results.append([
                ExtractedFeature(
                    uri=self.get_feature_uri("embedding"),
                    vector=vec,
                    metadata={"content_type": item.get("content_type", "image/jpeg")},
                )
            ])
        return results


@register_extractor("dinov2")
class DINOv2Extractor(BaseExtractor):
    """DINOv2 — visual features without text alignment (768d)."""

    model_spec = MODELS["dinov2"]

    def __init__(
        self,
        model_name: str = "facebook/dinov2-large",
        device: str = "cuda",
    ):
        self._model_name = model_name
        self._device = device
        self._model = None
        self._processor = None

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel
        except ImportError:
            raise RuntimeError(
                "Install transformers + torch: pip install transformers torch"
            )
        self._processor = AutoImageProcessor.from_pretrained(self._model_name)
        self._model = AutoModel.from_pretrained(self._model_name).to(self._device)
        self._model.eval()

    def _embed_images(self, images: list) -> list[list[float]]:
        import torch

        self._load_model()
        inputs = self._processor(images=images, return_tensors="pt").to(self._device)
        with torch.no_grad():
            out = self._model(**inputs)
        # Use CLS token
        feats = out.last_hidden_state[:, 0]
        feats = torch.nn.functional.normalize(feats, p=2, dim=1)
        return feats.cpu().tolist()

    def extract(
        self, content: bytes, content_type: str, **kwargs: Any
    ) -> list[ExtractedFeature]:
        from PIL import Image

        img = Image.open(io.BytesIO(content)).convert("RGB")
        vectors = self._embed_images([img])
        return [
            ExtractedFeature(
                uri=self.get_feature_uri("embedding"),
                vector=vectors[0],
                metadata={"content_type": content_type},
            )
        ]

    def extract_batch(
        self, batch: list[dict[str, Any]]
    ) -> list[list[ExtractedFeature]]:
        from PIL import Image

        images = [
            Image.open(io.BytesIO(item["content"])).convert("RGB")
            for item in batch
        ]
        vectors = self._embed_images(images)
        results = []
        for item, vec in zip(batch, vectors):
            results.append([
                ExtractedFeature(
                    uri=self.get_feature_uri("embedding"),
                    vector=vec,
                    metadata={"content_type": item.get("content_type", "image/jpeg")},
                )
            ])
        return results
