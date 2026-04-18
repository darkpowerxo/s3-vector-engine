"""Audio feature extractors — Whisper ASR, CLAP embedding."""

from __future__ import annotations

import io
from typing import Any

from . import (
    BaseExtractor,
    ExtractedFeature,
    MODELS,
    register_extractor,
)


@register_extractor("whisper")
class WhisperExtractor(BaseExtractor):
    """Whisper Large v3 — ASR transcription with word-level timestamps.

    Produces text transcript features (no embedding vector).
    Downstream text embedders (E5, BGE-M3) convert transcripts to vectors.
    Deployed as GPU Ray Serve replica (0.5 GPU fraction).
    """

    model_spec = MODELS["whisper"]

    def __init__(
        self,
        model_name: str = "openai/whisper-large-v3",
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
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        except ImportError:
            raise RuntimeError(
                "Install transformers + torch: pip install transformers torch"
            )
        self._processor = AutoProcessor.from_pretrained(self._model_name)
        self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self._model_name,
            torch_dtype=torch.float16 if "cuda" in self._device else torch.float32,
        ).to(self._device)
        self._model.eval()

    def _transcribe(self, audio_bytes: bytes) -> dict:
        import torch
        import numpy as np

        self._load_model()
        # Decode audio bytes to numpy array
        try:
            import soundfile as sf
            audio_array, sample_rate = sf.read(io.BytesIO(audio_bytes))
        except ImportError:
            raise RuntimeError("Install soundfile: pip install soundfile")

        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1)  # mono

        inputs = self._processor(
            audio_array,
            sampling_rate=sample_rate,
            return_tensors="pt",
            return_attention_mask=True,
        ).to(self._device)

        with torch.no_grad():
            generated = self._model.generate(
                **inputs,
                return_timestamps=True,
                language="en",
            )

        transcript = self._processor.batch_decode(
            generated, skip_special_tokens=True
        )[0]

        return {
            "text": transcript.strip(),
            "sample_rate": sample_rate,
            "duration_s": len(audio_array) / sample_rate,
        }

    def extract(
        self, content: bytes, content_type: str, **kwargs: Any
    ) -> list[ExtractedFeature]:
        result = self._transcribe(content)
        return [
            ExtractedFeature(
                uri=self.get_feature_uri("transcript"),
                text=result["text"],
                metadata={
                    "sample_rate": result["sample_rate"],
                    "duration_s": result["duration_s"],
                    "content_type": content_type,
                },
            )
        ]

    def extract_batch(
        self, batch: list[dict[str, Any]]
    ) -> list[list[ExtractedFeature]]:
        results = []
        for item in batch:
            result = self._transcribe(item["content"])
            results.append([
                ExtractedFeature(
                    uri=self.get_feature_uri("transcript"),
                    text=result["text"],
                    metadata={
                        "sample_rate": result["sample_rate"],
                        "duration_s": result["duration_s"],
                    },
                )
            ])
        return results


@register_extractor("clap")
class CLAPExtractor(BaseExtractor):
    """CLAP — audio-text cross-modal embedding (512d).

    Produces audio fingerprint vectors for similarity search.
    Deployed as GPU Ray Serve replica (0.25 GPU fraction).
    """

    model_spec = MODELS["clap"]

    def __init__(
        self,
        model_name: str = "laion/larger_clap_music_and_speech",
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
            from transformers import ClapModel, ClapProcessor
        except ImportError:
            raise RuntimeError(
                "Install transformers + torch: pip install transformers torch"
            )
        self._processor = ClapProcessor.from_pretrained(self._model_name)
        self._model = ClapModel.from_pretrained(self._model_name).to(self._device)
        self._model.eval()

    def _embed_audio(self, audio_arrays: list, sample_rate: int = 48000) -> list[list[float]]:
        import torch

        self._load_model()
        inputs = self._processor(
            audios=audio_arrays,
            sampling_rate=sample_rate,
            return_tensors="pt",
        ).to(self._device)
        with torch.no_grad():
            feats = self._model.get_audio_features(**inputs)
        feats = torch.nn.functional.normalize(feats, p=2, dim=1)
        return feats.cpu().tolist()

    def extract(
        self, content: bytes, content_type: str, **kwargs: Any
    ) -> list[ExtractedFeature]:
        try:
            import soundfile as sf
        except ImportError:
            raise RuntimeError("Install soundfile: pip install soundfile")

        audio_array, sample_rate = sf.read(io.BytesIO(content))
        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1)

        vectors = self._embed_audio([audio_array], sample_rate=sample_rate)
        return [
            ExtractedFeature(
                uri=self.get_feature_uri("fingerprint"),
                vector=vectors[0],
                metadata={
                    "sample_rate": sample_rate,
                    "content_type": content_type,
                },
            )
        ]

    def extract_batch(
        self, batch: list[dict[str, Any]]
    ) -> list[list[ExtractedFeature]]:
        try:
            import soundfile as sf
        except ImportError:
            raise RuntimeError("Install soundfile: pip install soundfile")

        results = []
        for item in batch:
            audio_array, sr = sf.read(io.BytesIO(item["content"]))
            if len(audio_array.shape) > 1:
                audio_array = audio_array.mean(axis=1)
            vectors = self._embed_audio([audio_array], sample_rate=sr)
            results.append([
                ExtractedFeature(
                    uri=self.get_feature_uri("fingerprint"),
                    vector=vectors[0],
                    metadata={"sample_rate": sr},
                )
            ])
        return results
