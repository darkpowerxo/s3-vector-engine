"""Face feature extractor — SCRFD detection + ArcFace embedding."""

from __future__ import annotations

import io
from typing import Any

from . import (
    BaseExtractor,
    ExtractedFeature,
    MODELS,
    register_extractor,
)


@register_extractor("arcface")
class ArcFaceExtractor(BaseExtractor):
    """SCRFD face detection → ArcFace identity embedding (512d).

    Two-stage pipeline:
    1. SCRFD detects face bounding boxes in the image
    2. ArcFace produces a 512d identity embedding per detected face

    Deployed as GPU Ray Serve replica (0.25 GPU fraction).
    A single image may yield multiple features (one per detected face).
    """

    model_spec = MODELS["arcface"]

    def __init__(
        self,
        detection_model: str = "buffalo_l",
        device: str = "cuda",
        det_thresh: float = 0.5,
    ):
        self._detection_model = detection_model
        self._device = device
        self._det_thresh = det_thresh
        self._app = None

    def _load_model(self) -> None:
        if self._app is not None:
            return
        try:
            import insightface
        except ImportError:
            raise RuntimeError(
                "Install insightface: pip install insightface onnxruntime-gpu"
            )
        self._app = insightface.app.FaceAnalysis(
            name=self._detection_model,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self._app.prepare(ctx_id=0, det_thresh=self._det_thresh)

    def _extract_faces(self, image) -> list[dict]:
        """Detect faces and return embedding + bbox for each."""
        import numpy as np

        self._load_model()
        faces = self._app.get(np.array(image))
        results = []
        for face in faces:
            embedding = face.embedding.tolist()
            bbox = face.bbox.tolist()
            results.append({
                "embedding": embedding,
                "bbox": bbox,
                "det_score": float(face.det_score),
            })
        return results

    def extract(
        self, content: bytes, content_type: str, **kwargs: Any
    ) -> list[ExtractedFeature]:
        from PIL import Image

        img = Image.open(io.BytesIO(content)).convert("RGB")
        faces = self._extract_faces(img)
        features = []
        for i, face in enumerate(faces):
            features.append(
                ExtractedFeature(
                    uri=self.get_feature_uri("embedding"),
                    vector=face["embedding"],
                    metadata={
                        "face_index": i,
                        "bbox": face["bbox"],
                        "det_score": face["det_score"],
                        "content_type": content_type,
                    },
                )
            )
        return features

    def extract_batch(
        self, batch: list[dict[str, Any]]
    ) -> list[list[ExtractedFeature]]:
        from PIL import Image

        results = []
        for item in batch:
            img = Image.open(io.BytesIO(item["content"])).convert("RGB")
            faces = self._extract_faces(img)
            features = []
            for i, face in enumerate(faces):
                features.append(
                    ExtractedFeature(
                        uri=self.get_feature_uri("embedding"),
                        vector=face["embedding"],
                        metadata={
                            "face_index": i,
                            "bbox": face["bbox"],
                            "det_score": face["det_score"],
                        },
                    )
                )
            results.append(features)
        return results
