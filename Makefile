.PHONY: up down seed run bench bench-detail clean install build-rust test docker-build docker-up docker-down

# ── Setup ────────────────────────────────────────────────────────────────────

# Install Python dependencies with uv
install:
	uv sync

# Build Rust shard scanner extension
build-rust:
	maturin develop --release

# ── Infrastructure ───────────────────────────────────────────────────────────

# Start MinIO + Redis
up:
	docker compose up -d

# Stop infrastructure
down:
	docker compose down

# Clean up Docker volumes
clean:
	docker compose down -v

# ── Run ──────────────────────────────────────────────────────────────────────

# Run the engine locally
run:
	uv run python -m s3vec.main

# Run tests (requires MinIO + Redis)
test:
	uv run python -m pytest tests/ -v

# ── Seed ─────────────────────────────────────────────────────────────────────

# Seed with 100K test vectors
seed:
	uv run python scripts/seed.py --num-vectors 100000 --dimensions 1024 --shard-size 10000

# Seed with 1M vectors (takes ~30s)
seed-large:
	uv run python scripts/seed.py --num-vectors 1000000 --dimensions 1024 --shard-size 10000

# Seed multiple tenants
seed-multi:
	uv run python scripts/seed.py --num-vectors 50000 --tenant-id bank_a
	uv run python scripts/seed.py --num-vectors 50000 --tenant-id bank_b
	uv run python scripts/seed.py --num-vectors 50000 --tenant-id bank_c

# ── Benchmark ────────────────────────────────────────────────────────────────

# Quick benchmark (100 queries)
bench:
	uv run python scripts/benchmark.py --num-queries 100 --top-k 10

# Detailed benchmark with per-shard timings
bench-detail:
	uv run python scripts/benchmark.py --num-queries 500 --top-k 10 --include-timings

# ── Docker (full stack) ─────────────────────────────────────────────────────

# Build the engine Docker image
docker-build:
	docker compose build engine

# Build all images (gateway, shard, ray)
docker-build-all:
	docker build -t s3vec-engine:latest -f Dockerfile .
	docker build -t s3vec-shard:latest -f Dockerfile.shard .
	docker build -t s3vec-ray:latest -f Dockerfile.ray .

# Start full stack (MinIO + Redis + Engine)
docker-up:
	docker compose --profile full up -d --build

# Stop full stack
docker-down:
	docker compose --profile full down

# ── Kubernetes ──────────────────────────────────────────────────────────────

# Apply raw K8s manifests
k8s-apply:
	kubectl apply -f k8s/namespace.yaml
	kubectl apply -f k8s/config.yaml
	kubectl apply -f k8s/infra.yaml
	kubectl apply -f k8s/shard.yaml
	kubectl apply -f k8s/gateway.yaml

# Apply KubeRay (requires KubeRay operator installed)
k8s-ray:
	kubectl apply -f k8s/ray.yaml

# Helm install (dry-run)
helm-dry-run:
	helm install s3vec deploy/helm/s3vec/ --namespace s3vec --create-namespace --dry-run

# Helm install
helm-install:
	helm install s3vec deploy/helm/s3vec/ --namespace s3vec --create-namespace

# Helm upgrade
helm-upgrade:
	helm upgrade s3vec deploy/helm/s3vec/ --namespace s3vec

# ── Load Testing ────────────────────────────────────────────────────────────

# Quick smoke test
bench-smoke:
	uv run python scripts/benchmark_load.py --mode smoke

# Write throughput benchmark
bench-write:
	uv run python scripts/benchmark_load.py --mode write --duration 60 --concurrency 64

# Query latency profiling
bench-query:
	uv run python scripts/benchmark_load.py --mode query --num-queries 50000 --concurrency 128

# Full production validation
bench-full:
	uv run python scripts/benchmark_load.py --mode full --duration 300 --output results.json
