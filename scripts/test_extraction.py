"""Phase 5 tests — Feature extraction framework, Ray infrastructure, ingest API.

Tests cover the extraction registry, Feature URI system, model specs,
Ray Serve deployment configs, ProgressActor, and extraction pipeline
(local mode, no ML models or Ray cluster required).
"""

import asyncio
import time
import pytest

# ── Feature URI ─────────────────────────────────────────────────────────────


class TestFeatureURI:
    def test_parse_valid(self):
        from s3vec.extraction.feature_uri import FeatureURI

        uri = FeatureURI.parse("s3vec://arcface@v1/embedding")
        assert uri.extractor == "arcface"
        assert uri.version == "v1"
        assert uri.feature_type == "embedding"

    def test_roundtrip(self):
        from s3vec.extraction.feature_uri import FeatureURI

        original = "s3vec://e5_large@v2/embedding"
        uri = FeatureURI.parse(original)
        assert str(uri) == original

    def test_parse_invalid(self):
        from s3vec.extraction.feature_uri import FeatureURI

        with pytest.raises(ValueError, match="Invalid feature URI"):
            FeatureURI.parse("invalid://bad")

    def test_parse_missing_version(self):
        from s3vec.extraction.feature_uri import FeatureURI

        with pytest.raises(ValueError):
            FeatureURI.parse("s3vec://model/embedding")

    def test_frozen(self):
        from s3vec.extraction.feature_uri import FeatureURI

        uri = FeatureURI(extractor="test", version="v1", feature_type="emb")
        with pytest.raises(AttributeError):
            uri.extractor = "changed"

    def test_equality(self):
        from s3vec.extraction.feature_uri import FeatureURI

        a = FeatureURI(extractor="x", version="v1", feature_type="emb")
        b = FeatureURI(extractor="x", version="v1", feature_type="emb")
        assert a == b

    def test_different_versions(self):
        from s3vec.extraction.feature_uri import FeatureURI

        a = FeatureURI.parse("s3vec://model@v1/embedding")
        b = FeatureURI.parse("s3vec://model@v2/embedding")
        assert a != b


# ── Model Specs ─────────────────────────────────────────────────────────────


class TestModelSpecs:
    def test_all_models_defined(self):
        from s3vec.extraction import MODELS

        expected = {
            "siglip", "clip", "e5_large", "bge_m3", "dinov2",
            "arcface", "clap", "whisper", "vertex_multimodal", "colqwen2",
        }
        assert set(MODELS.keys()) == expected

    def test_model_dimensions(self):
        from s3vec.extraction import MODELS

        assert MODELS["siglip"].dimensions == 1152
        assert MODELS["clip"].dimensions == 768
        assert MODELS["e5_large"].dimensions == 1024
        assert MODELS["bge_m3"].dimensions == 1024
        assert MODELS["arcface"].dimensions == 512
        assert MODELS["clap"].dimensions == 512
        assert MODELS["whisper"].dimensions is None  # ASR, no embedding
        assert MODELS["vertex_multimodal"].dimensions == 1408
        assert MODELS["colqwen2"].dimensions == 128

    def test_uri_prefix(self):
        from s3vec.extraction import MODELS

        assert MODELS["arcface"].uri_prefix == "s3vec://arcface@v1"
        assert MODELS["e5_large"].uri_prefix == "s3vec://e5_large@v1"


# ── Extractor Registry ─────────────────────────────────────────────────────


class TestExtractorRegistry:
    def test_all_extractors_registered(self):
        from s3vec.extraction import list_extractors

        registry = list_extractors()
        expected = {
            "e5_large", "bge_m3",
            "siglip", "clip", "dinov2",
            "arcface",
            "whisper", "clap",
            "vertex_multimodal",
        }
        assert set(registry.keys()) == expected

    def test_get_extractor(self):
        from s3vec.extraction import get_extractor
        from s3vec.extraction.text import E5LargeExtractor

        cls = get_extractor("e5_large")
        assert cls is E5LargeExtractor

    def test_get_unknown_raises(self):
        from s3vec.extraction import get_extractor

        with pytest.raises(KeyError, match="Unknown extractor"):
            get_extractor("nonexistent_model")

    def test_extractor_has_model_spec(self):
        from s3vec.extraction import get_extractor

        for name in ["e5_large", "siglip", "arcface", "whisper", "clap"]:
            cls = get_extractor(name)
            assert cls.model_spec is not None
            assert cls.model_spec.name == name

    def test_extractor_get_feature_uri(self):
        from s3vec.extraction import get_extractor

        cls = get_extractor("arcface")
        extractor = cls.__new__(cls)
        uri = extractor.get_feature_uri("embedding")
        assert str(uri) == "s3vec://arcface@v1/embedding"


# ── Ray Serve Deployment Configs ────────────────────────────────────────────


class TestServeConfig:
    def test_all_deployments_defined(self):
        from s3vec.ray.serve_config import DEPLOYMENTS

        assert len(DEPLOYMENTS) >= 10
        assert "text_embedder" in DEPLOYMENTS
        assert "image_embedder" in DEPLOYMENTS
        assert "face_detector" in DEPLOYMENTS
        assert "asr_model" in DEPLOYMENTS
        assert "audio_embedder" in DEPLOYMENTS
        assert "video_embedder" in DEPLOYMENTS

    def test_deployment_to_ray_config(self):
        from s3vec.ray.serve_config import get_deployment

        dep = get_deployment("text_embedder")
        cfg = dep.to_ray_config()
        assert cfg["name"] == "text_embedder"
        assert "autoscaling_config" in cfg
        assert cfg["autoscaling_config"]["min_replicas"] == 2
        assert cfg["autoscaling_config"]["max_replicas"] == 10
        assert cfg["ray_actor_options"]["num_cpus"] == 1
        assert "num_gpus" not in cfg["ray_actor_options"]  # CPU-only

    def test_gpu_deployment(self):
        from s3vec.ray.serve_config import get_deployment

        dep = get_deployment("image_embedder")
        cfg = dep.to_ray_config()
        assert cfg["ray_actor_options"]["num_gpus"] == 0.25

    def test_batch_isolation(self):
        from s3vec.ray.serve_config import get_deployment

        dep = get_deployment("video_embedder")
        cfg = dep.to_ray_config()
        assert cfg["ray_actor_options"]["resources"] == {"batch": 1}

    def test_build_serve_config(self):
        from s3vec.ray.serve_config import build_serve_config

        cfg = build_serve_config(["text_embedder", "image_embedder"])
        assert len(cfg["applications"]) == 2
        assert cfg["proxy_location"] == "EveryNode"

    def test_get_unknown_deployment(self):
        from s3vec.ray.serve_config import get_deployment

        with pytest.raises(KeyError, match="Unknown deployment"):
            get_deployment("does_not_exist")

    def test_actor_pool_kwargs(self):
        from s3vec.ray.serve_config import get_deployment

        dep = get_deployment("face_detector")
        kwargs = dep.to_actor_pool_kwargs()
        assert kwargs["batch_size"] == 8
        assert kwargs["ray_actor_options"]["num_gpus"] == 0.25


# ── ProgressActor ───────────────────────────────────────────────────────────


class TestProgressActor:
    def test_start_and_increment(self):
        from s3vec.ray.progress import ProgressActor, JobStatus

        pa = ProgressActor()
        pa.start_job("j1", total=100)

        p = pa.get_progress("j1")
        assert p["status"] == "running"
        assert p["total"] == 100
        assert p["completed"] == 0

        pa.increment("j1", 25)
        p = pa.get_progress("j1")
        assert p["completed"] == 25
        assert p["pct"] == 25.0

    def test_auto_complete(self):
        from s3vec.ray.progress import ProgressActor

        pa = ProgressActor()
        pa.start_job("j1", total=10)
        pa.increment("j1", 10)

        p = pa.get_progress("j1")
        assert p["status"] == "completed"
        assert p["pct"] == 100.0

    def test_fail_job(self):
        from s3vec.ray.progress import ProgressActor

        pa = ProgressActor()
        pa.start_job("j1", total=10)
        pa.fail_job("j1", "OOM")

        p = pa.get_progress("j1")
        assert p["status"] == "failed"
        assert p["error"] == "OOM"

    def test_cancel_job(self):
        from s3vec.ray.progress import ProgressActor

        pa = ProgressActor()
        pa.start_job("j1", total=10)
        pa.cancel_job("j1")
        assert pa.get_progress("j1")["status"] == "cancelled"

    def test_set_stage(self):
        from s3vec.ray.progress import ProgressActor

        pa = ProgressActor()
        pa.start_job("j1", total=10, stage="init")
        pa.set_stage("j1", "extracting")
        assert pa.get_progress("j1")["stage"] == "extracting"

    def test_set_total(self):
        from s3vec.ray.progress import ProgressActor

        pa = ProgressActor()
        pa.start_job("j1", total=0)
        pa.set_total("j1", 50)
        assert pa.get_progress("j1")["total"] == 50

    def test_unknown_job(self):
        from s3vec.ray.progress import ProgressActor

        pa = ProgressActor()
        assert pa.get_progress("nonexistent") is None

    def test_increment_unknown_job_noop(self):
        from s3vec.ray.progress import ProgressActor

        pa = ProgressActor()
        pa.increment("nonexistent", 5)  # should not raise

    def test_list_jobs(self):
        from s3vec.ray.progress import ProgressActor

        pa = ProgressActor()
        pa.start_job("j1", total=10)
        pa.start_job("j2", total=20)
        jobs = pa.list_jobs()
        assert len(jobs) == 2
        ids = {j["job_id"] for j in jobs}
        assert ids == {"j1", "j2"}

    def test_elapsed_time(self):
        from s3vec.ray.progress import ProgressActor

        pa = ProgressActor()
        pa.start_job("j1", total=10)
        p = pa.get_progress("j1")
        assert p["elapsed_s"] >= 0

    def test_increment_failed(self):
        from s3vec.ray.progress import ProgressActor

        pa = ProgressActor()
        pa.start_job("j1", total=10)
        pa.increment_failed("j1", 3)
        p = pa.get_progress("j1")
        assert p["failed"] == 3


# ── Extraction Data Types ──────────────────────────────────────────────────


class TestExtractionTypes:
    def test_extracted_feature(self):
        from s3vec.extraction import ExtractedFeature
        from s3vec.extraction.feature_uri import FeatureURI

        uri = FeatureURI(extractor="test", version="v1", feature_type="emb")
        feat = ExtractedFeature(uri=uri, vector=[1.0, 2.0, 3.0])
        assert feat.vector == [1.0, 2.0, 3.0]
        assert feat.text is None
        assert feat.timestamp_ms is None

    def test_extraction_result_success(self):
        from s3vec.extraction import ExtractionResult

        r = ExtractionResult(source_id="doc-1")
        assert r.success is True

    def test_extraction_result_failure(self):
        from s3vec.extraction import ExtractionResult

        r = ExtractionResult(source_id="doc-1", errors=["boom"])
        assert r.success is False

    def test_modality_enum(self):
        from s3vec.extraction import Modality

        assert Modality.TEXT == "text"
        assert Modality.IMAGE == "image"
        assert Modality.VIDEO == "video"
        assert Modality.AUDIO == "audio"
        assert Modality.FACE == "face"
        assert Modality.MULTIMODAL == "multimodal"


# ── ShardDatasink ───────────────────────────────────────────────────────────


class TestShardDatasink:
    def test_init(self):
        from s3vec.ray.datasink import ShardDatasink

        sink = ShardDatasink(
            namespace="test",
            shard_addresses=["localhost:9051", "localhost:9052"],
        )
        assert sink._namespace == "test"
        assert sink._batch_size == 100

    def test_num_rows_per_write(self):
        from s3vec.ray.datasink import ShardDatasink

        sink = ShardDatasink(
            namespace="t",
            shard_addresses=["localhost:9051"],
            batch_size=50,
        )
        assert sink.num_rows_per_write == 50


# ── Pipeline Config ─────────────────────────────────────────────────────────


class TestPipelineConfig:
    def test_default_job_id(self):
        from s3vec.ray.pipeline import PipelineConfig, ExtractorConfig

        cfg = PipelineConfig(
            namespace="test",
            extractors=[ExtractorConfig(name="e5_large")],
            shard_addresses=["localhost:9051"],
        )
        assert cfg.job_id.startswith("job-")

    def test_extractor_config(self):
        from s3vec.ray.pipeline import ExtractorConfig

        ec = ExtractorConfig(name="siglip", batch_size=16)
        assert ec.name == "siglip"
        assert ec.batch_size == 16
        assert ec.deployment is None


# ── Integration: Extraction + URI ───────────────────────────────────────────


class TestExtractionIntegration:
    def test_all_extractors_produce_valid_uris(self):
        """Every registered extractor should produce parseable Feature URIs."""
        from s3vec.extraction import list_extractors
        from s3vec.extraction.feature_uri import FeatureURI

        for name, cls in list_extractors().items():
            extractor = cls.__new__(cls)
            uri = extractor.get_feature_uri("embedding")
            assert isinstance(uri, FeatureURI)
            parsed = FeatureURI.parse(str(uri))
            assert parsed.extractor == name

    def test_model_spec_consistency(self):
        """Model specs referenced by extractors should exist in MODELS."""
        from s3vec.extraction import MODELS, list_extractors

        for name, cls in list_extractors().items():
            spec = cls.model_spec
            assert spec.name in MODELS
            assert MODELS[spec.name] is spec
