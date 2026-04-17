//! Prometheus metrics for the shard server.
//!
//! Exposes counters and histograms for gRPC operations.

use prometheus::{
    Encoder, HistogramOpts, HistogramVec, IntCounter, IntCounterVec, IntGauge,
    Opts, Registry, TextEncoder,
};
use std::sync::LazyLock;

/// Global metrics registry.
pub static REGISTRY: LazyLock<Registry> = LazyLock::new(Registry::new);

/// Total number of search requests.
pub static SEARCH_TOTAL: LazyLock<IntCounter> = LazyLock::new(|| {
    let c = IntCounter::with_opts(
        Opts::new("shard_search_total", "Total search requests served"),
    )
    .unwrap();
    REGISTRY.register(Box::new(c.clone())).unwrap();
    c
});

/// Search latency in seconds.
pub static SEARCH_DURATION: LazyLock<HistogramVec> = LazyLock::new(|| {
    let h = HistogramVec::new(
        HistogramOpts::new("shard_search_duration_seconds", "Search latency histogram")
            .buckets(vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]),
        &["status"],
    )
    .unwrap();
    REGISTRY.register(Box::new(h.clone())).unwrap();
    h
});

/// Total number of upsert requests.
pub static UPSERT_TOTAL: LazyLock<IntCounter> = LazyLock::new(|| {
    let c = IntCounter::with_opts(
        Opts::new("shard_upsert_total", "Total upsert requests"),
    )
    .unwrap();
    REGISTRY.register(Box::new(c.clone())).unwrap();
    c
});

/// Total number of delete requests.
pub static DELETE_TOTAL: LazyLock<IntCounter> = LazyLock::new(|| {
    let c = IntCounter::with_opts(
        Opts::new("shard_delete_total", "Total delete requests"),
    )
    .unwrap();
    REGISTRY.register(Box::new(c.clone())).unwrap();
    c
});

/// Documents upserted (may be batched).
pub static DOCUMENTS_UPSERTED: LazyLock<IntCounter> = LazyLock::new(|| {
    let c = IntCounter::with_opts(
        Opts::new("shard_documents_upserted_total", "Total documents upserted"),
    )
    .unwrap();
    REGISTRY.register(Box::new(c.clone())).unwrap();
    c
});

/// Current dense vector count.
pub static DENSE_COUNT: LazyLock<IntGauge> = LazyLock::new(|| {
    let g = IntGauge::with_opts(
        Opts::new("shard_dense_vectors", "Current dense vector count"),
    )
    .unwrap();
    REGISTRY.register(Box::new(g.clone())).unwrap();
    g
});

/// Current WAL sequence number.
pub static WAL_SEQUENCE: LazyLock<IntGauge> = LazyLock::new(|| {
    let g = IntGauge::with_opts(
        Opts::new("shard_wal_sequence", "Current WAL sequence number"),
    )
    .unwrap();
    REGISTRY.register(Box::new(g.clone())).unwrap();
    g
});

/// Snapshot operations counter by status.
pub static SNAPSHOT_TOTAL: LazyLock<IntCounterVec> = LazyLock::new(|| {
    let c = IntCounterVec::new(
        Opts::new("shard_snapshot_total", "Total snapshot operations"),
        &["status"],
    )
    .unwrap();
    REGISTRY.register(Box::new(c.clone())).unwrap();
    c
});

/// Render all metrics as Prometheus text format.
pub fn gather_metrics() -> String {
    let encoder = TextEncoder::new();
    let metric_families = REGISTRY.gather();
    let mut buffer = Vec::new();
    encoder.encode(&metric_families, &mut buffer).unwrap();
    String::from_utf8(buffer).unwrap()
}
