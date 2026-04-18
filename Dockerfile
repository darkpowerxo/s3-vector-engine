# Dockerfile — Gateway / Coordinator API server
# ── Stage 1: Build Rust extension + Python deps ─────────────────────────────
FROM python:3.11-slim AS builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl build-essential pkg-config protobuf-compiler && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /build

RUN pip install --no-cache-dir uv maturin

# Python deps (layer-cached)
COPY pyproject.toml .
RUN uv pip install --system -r pyproject.toml

# Build Rust PyO3 extension
COPY Cargo.toml build.rs ./
COPY .cargo/ .cargo/
COPY proto/ proto/
COPY src/ src/
COPY s3vec/ s3vec/
RUN maturin develop --release

# ── Stage 2: Slim runtime ───────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /build/s3vec/ /app/s3vec/
COPY scripts/ /app/scripts/

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["python", "-m", "s3vec.main"]
