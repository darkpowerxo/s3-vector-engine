"""Video feature extractor — scene detection + Vertex AI multimodal embedding."""

from __future__ import annotations

import io
from typing import Any

from . import (
    BaseExtractor,
    ExtractedFeature,
    MODELS,
    register_extractor,
)


@register_extractor("vertex_multimodal")
class VertexMultimodalExtractor(BaseExtractor):
    """Vertex AI Multimodal Embedding — video/image/text (1408d).

    Pipeline:
    1. Scene detection splits video into segments at scene boundaries
    2. Keyframe sampling extracts representative frames per scene
    3. Vertex AI multimodalembedding@001 produces 1408d vectors per segment

    Deployed as GPU Ray Serve replica (0.5 GPU fraction).
    """

    model_spec = MODELS["vertex_multimodal"]

    def __init__(
        self,
        project_id: str | None = None,
        location: str = "us-central1",
        model_id: str = "multimodalembedding@001",
    ):
        self._project_id = project_id
        self._location = location
        self._model_id = model_id
        self._model = None

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            from google.cloud import aiplatform
            from vertexai.vision_models import MultiModalEmbeddingModel
        except ImportError:
            raise RuntimeError(
                "Install google-cloud-aiplatform: "
                "pip install google-cloud-aiplatform"
            )
        if self._project_id:
            aiplatform.init(project=self._project_id, location=self._location)
        self._model = MultiModalEmbeddingModel.from_pretrained(self._model_id)

    def _detect_scenes(self, video_bytes: bytes) -> list[dict]:
        """Detect scene boundaries and extract keyframes.

        Returns list of scene dicts with 'start_ms', 'end_ms', 'keyframe' (PIL Image).
        """
        try:
            import cv2
            import numpy as np
        except ImportError:
            raise RuntimeError("Install opencv: pip install opencv-python-headless")

        from PIL import Image

        # Decode video from bytes
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(video_bytes)
            tmp_path = f.name

        try:
            cap = cv2.VideoCapture(tmp_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            scenes = []
            prev_hist = None
            scene_start_frame = 0
            frame_idx = 0
            threshold = 0.4  # histogram difference threshold

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # Compute color histogram
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
                cv2.normalize(hist, hist)

                if prev_hist is not None:
                    diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
                    if diff > threshold:
                        # Scene boundary detected
                        mid_frame = (scene_start_frame + frame_idx) // 2
                        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
                        _, keyframe = cap.read()
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx + 1)

                        keyframe_rgb = cv2.cvtColor(keyframe, cv2.COLOR_BGR2RGB)
                        scenes.append({
                            "start_ms": scene_start_frame / fps * 1000,
                            "end_ms": frame_idx / fps * 1000,
                            "keyframe": Image.fromarray(keyframe_rgb),
                        })
                        scene_start_frame = frame_idx

                prev_hist = hist
                frame_idx += 1

            # Last scene
            if frame_idx > scene_start_frame:
                mid_frame = (scene_start_frame + frame_idx) // 2
                cap.set(cv2.CAP_PROP_POS_FRAMES, min(mid_frame, frame_idx - 1))
                ret, keyframe = cap.read()
                if ret:
                    keyframe_rgb = cv2.cvtColor(keyframe, cv2.COLOR_BGR2RGB)
                    scenes.append({
                        "start_ms": scene_start_frame / fps * 1000,
                        "end_ms": frame_idx / fps * 1000,
                        "keyframe": Image.fromarray(keyframe_rgb),
                    })

            cap.release()
        finally:
            os.unlink(tmp_path)

        return scenes

    def _embed_image(self, image) -> list[float]:
        """Embed a single image via Vertex AI."""
        from vertexai.vision_models import Image as VertexImage

        # Convert PIL to bytes
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        vertex_img = VertexImage(image_bytes=buf.getvalue())

        embedding = self._model.get_embeddings(image=vertex_img)
        return embedding.image_embedding

    def extract(
        self, content: bytes, content_type: str, **kwargs: Any
    ) -> list[ExtractedFeature]:
        self._load_model()

        if content_type.startswith("video/"):
            scenes = self._detect_scenes(content)
            features = []
            for i, scene in enumerate(scenes):
                vector = self._embed_image(scene["keyframe"])
                features.append(
                    ExtractedFeature(
                        uri=self.get_feature_uri("scene_embedding"),
                        vector=vector,
                        timestamp_ms=scene["start_ms"],
                        metadata={
                            "scene_index": i,
                            "start_ms": scene["start_ms"],
                            "end_ms": scene["end_ms"],
                        },
                    )
                )
            # Also emit scene boundary metadata
            features.append(
                ExtractedFeature(
                    uri=self.get_feature_uri("scene_boundaries"),
                    metadata={
                        "scene_count": len(scenes),
                        "scenes": [
                            {"start_ms": s["start_ms"], "end_ms": s["end_ms"]}
                            for s in scenes
                        ],
                    },
                )
            )
            return features
        else:
            # Single image
            from PIL import Image

            img = Image.open(io.BytesIO(content)).convert("RGB")
            vector = self._embed_image(img)
            return [
                ExtractedFeature(
                    uri=self.get_feature_uri("embedding"),
                    vector=vector,
                    metadata={"content_type": content_type},
                )
            ]

    def extract_batch(
        self, batch: list[dict[str, Any]]
    ) -> list[list[ExtractedFeature]]:
        self._load_model()
        results = []
        for item in batch:
            ct = item.get("content_type", "image/jpeg")
            feats = self.extract(item["content"], ct)
            results.append(feats)
        return results
