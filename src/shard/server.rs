//! gRPC ShardService implementation.
//!
//! Wraps the Shard engine and exposes it as a tonic gRPC server.
//! Each shard process runs one ShardServiceImpl.

use crate::index::payload::FilterCondition;
use crate::proto::s3vec::shard::{
    self as pb,
    shard_service_server::ShardService,
};
use crate::shard::engine::{Shard, ShardConfig};
use crate::types::{FusionStrategy, SparseVector};

use parking_lot::Mutex;
use serde_json::Value;
use std::sync::Arc;
use std::time::Instant;
use tonic::{Request, Response, Status};

/// The gRPC service implementation wrapping a single Shard.
pub struct ShardServiceImpl {
    shard: Arc<Mutex<Shard>>,
}

impl ShardServiceImpl {
    /// Create a new service wrapping a shard.
    pub fn new(shard: Shard) -> Self {
        Self {
            shard: Arc::new(Mutex::new(shard)),
        }
    }

    /// Create from config.
    pub fn from_config(config: ShardConfig) -> Result<Self, String> {
        let shard = Shard::open(config).map_err(|e| e.to_string())?;
        Ok(Self::new(shard))
    }
}

// ── Helpers ────────────────────────────────────────────────────────────────

fn pb_fusion_to_engine(f: i32) -> FusionStrategy {
    match pb::FusionStrategy::try_from(f) {
        Ok(pb::FusionStrategy::Dbsf) => FusionStrategy::Dbsf,
        Ok(pb::FusionStrategy::Linear) => FusionStrategy::Linear,
        _ => FusionStrategy::Rrf,
    }
}

fn pb_filter_to_engine(filter: &pb::FilterExpression) -> Option<FilterCondition> {
    match &filter.expr {
        Some(pb::filter_expression::Expr::Field(f)) => {
            let value: Value = serde_json::from_str(&f.value_json).ok()?;
            let op = pb::FilterOp::try_from(f.op).unwrap_or(pb::FilterOp::Eq);
            match op {
                pb::FilterOp::Eq => Some(FilterCondition::Eq(f.field_name.clone(), value)),
                pb::FilterOp::Ne => Some(FilterCondition::Ne(f.field_name.clone(), value)),
                pb::FilterOp::Gt => {
                    value.as_f64().map(|v| FilterCondition::Gt(f.field_name.clone(), v))
                }
                pb::FilterOp::Gte => {
                    value.as_f64().map(|v| FilterCondition::Gte(f.field_name.clone(), v))
                }
                pb::FilterOp::Lt => {
                    value.as_f64().map(|v| FilterCondition::Lt(f.field_name.clone(), v))
                }
                pb::FilterOp::Lte => {
                    value.as_f64().map(|v| FilterCondition::Lte(f.field_name.clone(), v))
                }
                pb::FilterOp::In => {
                    if let Value::Array(arr) = value {
                        Some(FilterCondition::In(f.field_name.clone(), arr))
                    } else {
                        None
                    }
                }
                pb::FilterOp::Contains => {
                    value.as_str().map(|s| FilterCondition::Contains(f.field_name.clone(), s.to_string()))
                }
            }
        }
        Some(pb::filter_expression::Expr::Composite(c)) => {
            let children: Vec<FilterCondition> = c
                .children
                .iter()
                .filter_map(pb_filter_to_engine)
                .collect();
            let op = pb::CompositeOp::try_from(c.op).unwrap_or(pb::CompositeOp::And);
            match op {
                pb::CompositeOp::And => Some(FilterCondition::And(children)),
                pb::CompositeOp::Or => Some(FilterCondition::Or(children)),
                pb::CompositeOp::Not => {
                    children.into_iter().next().map(|c| FilterCondition::Not(Box::new(c)))
                }
            }
        }
        None => None,
    }
}

// ── gRPC Service Implementation ───────────────────────────────────────────

#[tonic::async_trait]
impl ShardService for ShardServiceImpl {
    async fn search(
        &self,
        request: Request<pb::SearchRequest>,
    ) -> Result<Response<pb::SearchResponse>, Status> {
        let t0 = Instant::now();
        let req = request.into_inner();

        let k = req.top_k.max(1) as usize;
        let strategy = pb_fusion_to_engine(req.fusion);

        // Extract dense query
        let dense_vec: Option<Vec<f32>> = req.dense.as_ref().map(|d| d.vector.clone());

        // Extract sparse query
        let sparse_q: Option<SparseVector> = req.sparse.as_ref().map(|s| {
            SparseVector::new(s.indices.clone(), s.values.clone())
        });

        // Extract filter
        let filter_cond = req.filter.as_ref().and_then(pb_filter_to_engine);

        // BM25 query
        let text_query = req.bm25_query.as_deref();

        // Default weights
        let weights = vec![1.0; 3];

        let shard = self.shard.lock();
        let results = shard
            .search(
                dense_vec.as_deref(),
                sparse_q.as_ref(),
                text_query,
                k,
                strategy,
                &weights,
                filter_cond.as_ref(),
            )
            .map_err(|e| Status::internal(e.to_string()))?;

        let include_payloads = req.include_payloads;
        let scored_docs: Vec<pb::ScoredDocument> = results
            .into_iter()
            .map(|r| {
                let payload = if include_payloads {
                    shard
                        .get_payload(&r.id)
                        .ok()
                        .flatten()
                        .and_then(|v| serde_json::to_vec(&v).ok())
                } else {
                    None
                };
                pb::ScoredDocument {
                    id: r.id,
                    score: r.score,
                    payload,
                    dense_vector: None,
                    sparse_vector: None,
                }
            })
            .collect();

        let duration_us = t0.elapsed().as_micros() as u64;

        Ok(Response::new(pb::SearchResponse {
            results: scored_docs,
            vectors_scanned: 0, // TODO: track in shard
            duration_us,
        }))
    }

    type SearchStreamStream =
        tokio_stream::wrappers::ReceiverStream<Result<pb::SearchResult, Status>>;

    async fn search_stream(
        &self,
        request: Request<pb::SearchRequest>,
    ) -> Result<Response<Self::SearchStreamStream>, Status> {
        // Perform search then stream results one by one
        let search_resp = self.search(Request::new(request.into_inner())).await?;
        let results = search_resp.into_inner().results;

        let (tx, rx) = tokio::sync::mpsc::channel(results.len().max(1));
        tokio::spawn(async move {
            for doc in results {
                let _ = tx.send(Ok(pb::SearchResult { doc: Some(doc) })).await;
            }
        });

        Ok(Response::new(tokio_stream::wrappers::ReceiverStream::new(rx)))
    }

    async fn upsert(
        &self,
        request: Request<pb::UpsertRequest>,
    ) -> Result<Response<pb::UpsertResponse>, Status> {
        let req = request.into_inner();

        let dense = req.dense_vector.map(|d| d.values);
        let sparse = req.sparse_vector.map(|s| SparseVector::new(s.indices, s.values));
        let text_fields = if req.text_fields.is_empty() {
            None
        } else {
            Some(req.text_fields.into_iter().collect())
        };
        let payload: Option<Value> = req
            .payload
            .and_then(|b| serde_json::from_slice(&b).ok());

        let mut shard = self.shard.lock();
        let seq = shard
            .upsert(req.id, dense, sparse, text_fields, payload)
            .map_err(|e| Status::internal(e.to_string()))?;

        Ok(Response::new(pb::UpsertResponse {
            wal_sequence: seq,
            documents_written: 1,
        }))
    }

    async fn batch_upsert(
        &self,
        request: Request<tonic::Streaming<pb::UpsertRequest>>,
    ) -> Result<Response<pb::UpsertResponse>, Status> {
        let mut stream = request.into_inner();
        let mut count: u32 = 0;
        let mut last_seq: u64 = 0;

        while let Some(req) = stream.message().await? {
            let dense = req.dense_vector.map(|d| d.values);
            let sparse = req.sparse_vector.map(|s| SparseVector::new(s.indices, s.values));
            let text_fields = if req.text_fields.is_empty() {
                None
            } else {
                Some(req.text_fields.into_iter().collect())
            };
            let payload: Option<Value> = req
                .payload
                .and_then(|b| serde_json::from_slice(&b).ok());

            let mut shard = self.shard.lock();
            last_seq = shard
                .upsert(req.id, dense, sparse, text_fields, payload)
                .map_err(|e| Status::internal(e.to_string()))?;
            count += 1;
        }

        Ok(Response::new(pb::UpsertResponse {
            wal_sequence: last_seq,
            documents_written: count,
        }))
    }

    async fn delete(
        &self,
        request: Request<pb::DeleteRequest>,
    ) -> Result<Response<pb::DeleteResponse>, Status> {
        let req = request.into_inner();

        let mut shard = self.shard.lock();
        let seq = shard
            .delete(&req.id)
            .map_err(|e| Status::internal(e.to_string()))?;

        Ok(Response::new(pb::DeleteResponse {
            wal_sequence: seq,
            found: true,
        }))
    }

    async fn create_snapshot(
        &self,
        _request: Request<pb::SnapshotRequest>,
    ) -> Result<Response<pb::SnapshotResponse>, Status> {
        // TODO: implement snapshot serialization
        Err(Status::unimplemented("snapshot not yet implemented"))
    }

    type TailWALStream =
        tokio_stream::wrappers::ReceiverStream<Result<pb::WalEvent, Status>>;

    async fn tail_wal(
        &self,
        _request: Request<pb::TailWalRequest>,
    ) -> Result<Response<Self::TailWALStream>, Status> {
        // TODO: implement WAL tailing
        Err(Status::unimplemented("WAL tailing not yet implemented"))
    }

    async fn get_stats(
        &self,
        _request: Request<pb::StatsRequest>,
    ) -> Result<Response<pb::StatsResponse>, Status> {
        let shard = self.shard.lock();
        let stats = shard.stats();

        Ok(Response::new(pb::StatsResponse {
            dense_count: stats.dense_count as u64,
            sparse_count: stats.sparse_count as u64,
            sparse_vocab_size: stats.sparse_vocab_size as u64,
            payload_count: stats.payload_count as u64,
            wal_sequence: stats.wal_sequence,
            dim: stats.dim as u32,
        }))
    }
}
