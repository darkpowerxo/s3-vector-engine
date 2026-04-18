"""Text feature extractors — E5-Large-v2, BGE-M3."""

from __future__ import annotations

from typing import Any

from . import (
    BaseExtractor,
    ExtractedFeature,
    MODELS,
    register_extractor,
)


@register_extractor("e5_large")
class E5LargeExtractor(BaseExtractor):
    """E5-Large-v2 — English text semantic search (1024d).

    Deployed as a CPU-only Ray Serve replica.
    Uses ``query: `` / ``passage: `` prefixes per the E5 convention.
    """

    model_spec = MODELS["e5_large"]

    def __init__(
        self,
        model_name: str = "intfloat/e5-large-v2",
        device: str = "cpu",
    ):
        self._model_name = model_name
        self._device = device
        self._model = None
        self._tokenizer = None

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            raise RuntimeError(
                "Install transformers + torch: pip install transformers torch"
            )
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._model = AutoModel.from_pretrained(self._model_name).to(self._device)
        self._model.eval()

    def _embed(self, texts: list[str]) -> list[list[float]]:
        import torch

        self._load_model()
        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self._device)
        with torch.no_grad():
            out = self._model(**encoded)
        # Mean pooling over token embeddings
        mask = encoded["attention_mask"].unsqueeze(-1).float()
        embeddings = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
        # L2 normalize
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().tolist()

    def extract(
        self, content: bytes, content_type: str, **kwargs: Any
    ) -> list[ExtractedFeature]:
        text = content.decode("utf-8")
        prefix = kwargs.get("prefix", "passage: ")
        vectors = self._embed([f"{prefix}{text}"])
        return [
            ExtractedFeature(
                uri=self.get_feature_uri("embedding"),
                vector=vectors[0],
                text=text,
            )
        ]

    def extract_batch(
        self, batch: list[dict[str, Any]]
    ) -> list[list[ExtractedFeature]]:
        prefix = "passage: "
        texts = [f"{prefix}{item['text']}" for item in batch]
        vectors = self._embed(texts)
        results = []
        for item, vec in zip(batch, vectors):
            results.append([
                ExtractedFeature(
                    uri=self.get_feature_uri("embedding"),
                    vector=vec,
                    text=item.get("text"),
                )
            ])
        return results


@register_extractor("bge_m3")
class BGEM3Extractor(BaseExtractor):
    """BGE-M3 — multilingual text, dense + sparse (1024d).

    Produces both dense and sparse (lexical weight) representations.
    """

    model_spec = MODELS["bge_m3"]

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = "cpu",
    ):
        self._model_name = model_name
        self._device = device
        self._model = None

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError:
            raise RuntimeError(
                "Install FlagEmbedding: pip install FlagEmbedding"
            )
        self._model = BGEM3FlagModel(self._model_name, use_fp16=False)

    def _embed(self, texts: list[str]) -> tuple[list[list[float]], list[dict]]:
        self._load_model()
        output = self._model.encode(
            texts,
            return_dense=True,
            return_sparse=True,
        )
        dense = output["dense_vecs"].tolist()
        sparse = output.get("lexical_weights", [{}] * len(texts))
        return dense, sparse

    def extract(
        self, content: bytes, content_type: str, **kwargs: Any
    ) -> list[ExtractedFeature]:
        text = content.decode("utf-8")
        dense_vecs, sparse_weights = self._embed([text])
        features = [
            ExtractedFeature(
                uri=self.get_feature_uri("embedding"),
                vector=dense_vecs[0],
                text=text,
            ),
        ]
        if sparse_weights and sparse_weights[0]:
            features.append(
                ExtractedFeature(
                    uri=self.get_feature_uri("sparse"),
                    metadata={"lexical_weights": sparse_weights[0]},
                    text=text,
                )
            )
        return features

    def extract_batch(
        self, batch: list[dict[str, Any]]
    ) -> list[list[ExtractedFeature]]:
        texts = [item["text"] for item in batch]
        dense_vecs, sparse_weights = self._embed(texts)
        results = []
        for i, item in enumerate(batch):
            feats = [
                ExtractedFeature(
                    uri=self.get_feature_uri("embedding"),
                    vector=dense_vecs[i],
                    text=item.get("text"),
                )
            ]
            if sparse_weights and i < len(sparse_weights) and sparse_weights[i]:
                feats.append(
                    ExtractedFeature(
                        uri=self.get_feature_uri("sparse"),
                        metadata={"lexical_weights": sparse_weights[i]},
                        text=item.get("text"),
                    )
                )
            results.append(feats)
        return results
