//! Shard gRPC server binary.
//!
//! Starts a single shard process that serves the ShardService over gRPC.
//!
//! Usage:
//!   shard-server --data-dir ./data/shard-0 --port 50051 --dim 768

use _engine::index::bm25::BM25Config;
use _engine::index::hnsw::HnswConfig;
use _engine::metrics;
use _engine::proto::s3vec::shard::shard_service_server::ShardServiceServer;
use _engine::shard::engine::ShardConfig;
use _engine::shard::server::ShardServiceImpl;
use _engine::types::DistanceMetric;

use clap::Parser;
use std::path::PathBuf;
use tonic::transport::Server;
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
#[command(name = "shard-server", about = "S3 Vector Engine — Shard gRPC Server")]
struct Args {
    /// Data directory for this shard.
    #[arg(long, default_value = "./data/shard-0")]
    data_dir: PathBuf,

    /// gRPC listen port.
    #[arg(long, default_value_t = 9051)]
    port: u16,

    /// Vector dimensionality.
    #[arg(long, default_value_t = 768)]
    dim: usize,

    /// HNSW M parameter.
    #[arg(long, default_value_t = 16)]
    hnsw_m: usize,

    /// HNSW ef_construction.
    #[arg(long, default_value_t = 200)]
    ef_construction: usize,

    /// HNSW ef_search.
    #[arg(long, default_value_t = 128)]
    ef_search: usize,

    /// Distance metric: cosine, l2, ip.
    #[arg(long, default_value = "cosine")]
    metric: String,

    /// BM25 text fields (comma-separated).
    #[arg(long, default_value = "text")]
    text_fields: String,

    /// Prometheus metrics HTTP port (default: gRPC port + 1000).
    #[arg(long)]
    metrics_port: Option<u16>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();

    let metric = match args.metric.as_str() {
        "cosine" => DistanceMetric::Cosine,
        "l2" => DistanceMetric::L2,
        "ip" | "inner_product" => DistanceMetric::InnerProduct,
        other => {
            eprintln!("Unknown metric: {other}. Use cosine, l2, or ip.");
            std::process::exit(1);
        }
    };

    let text_fields: Vec<String> = args
        .text_fields
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    let config = ShardConfig {
        data_dir: args.data_dir.clone(),
        hnsw: HnswConfig {
            m: args.hnsw_m,
            m0: args.hnsw_m * 2,
            ef_construction: args.ef_construction,
            ef_search: args.ef_search,
            metric,
            dim: args.dim,
        },
        bm25: BM25Config {
            text_fields,
            writer_heap_size: 50_000_000,
        },
        wal_sync_interval: 1000,
        rrf_k: 60.0,
    };

    tracing::info!(
        data_dir = %args.data_dir.display(),
        port = args.port,
        dim = args.dim,
        metric = %metric,
        "starting shard server",
    );

    let service = ShardServiceImpl::from_config(config)
        .map_err(|e| format!("failed to open shard: {e}"))?;

    let addr = format!("0.0.0.0:{}", args.port).parse()?;

    // Start Prometheus metrics HTTP server
    let metrics_port = args.metrics_port.unwrap_or(args.port + 1000);
    let metrics_addr: std::net::SocketAddr =
        format!("0.0.0.0:{metrics_port}").parse()?;
    tokio::spawn(serve_metrics(metrics_addr));
    tracing::info!(%metrics_addr, "metrics HTTP server listening");

    tracing::info!(%addr, "shard server listening");

    Server::builder()
        .add_service(ShardServiceServer::new(service))
        .serve(addr)
        .await?;

    Ok(())
}

/// Serve Prometheus metrics on a simple HTTP endpoint.
async fn serve_metrics(addr: std::net::SocketAddr) {
    use http_body_util::Full;
    use hyper::body::Bytes;
    use hyper::server::conn::http1;
    use hyper::service::service_fn;
    use hyper::{Request, Response};
    use hyper_util::rt::TokioIo;

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();

    loop {
        let (stream, _) = match listener.accept().await {
            Ok(conn) => conn,
            Err(e) => {
                tracing::warn!("metrics accept error: {e}");
                continue;
            }
        };
        let io = TokioIo::new(stream);
        tokio::spawn(async move {
            let svc = service_fn(|req: Request<hyper::body::Incoming>| async move {
                if req.uri().path() == "/metrics" {
                    let body = metrics::gather_metrics();
                    Ok::<_, std::convert::Infallible>(
                        Response::builder()
                            .header("Content-Type", "text/plain; version=0.0.4")
                            .body(Full::new(Bytes::from(body)))
                            .unwrap(),
                    )
                } else {
                    Ok(Response::builder()
                        .status(404)
                        .body(Full::new(Bytes::from("Not Found")))
                        .unwrap())
                }
            });
            if let Err(e) = http1::Builder::new().serve_connection(io, svc).await {
                tracing::debug!("metrics connection error: {e}");
            }
        });
    }
}
