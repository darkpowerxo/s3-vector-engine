//! Shard engine — top-level struct that owns HNSW, BM25, Sparse, Payload, WAL.
//!
//! This is the composition point: a single shard manages up to ~50M vectors
//! with all index types co-located for fast hybrid search.

use crate::index::bm25::{BM25Config, BM25Error, BM25Index};
use crate::index::hnsw::{HnswConfig, HnswIndex};
use crate::index::payload::{FilterCondition, PayloadError, PayloadStore};
use crate::index::sparse::SparseIndex;
use crate::search::fusion;
use crate::search::mmr::{self, ScoredDoc};
use crate::types::{DistanceMetric, FusionStrategy, ScoredId, SparseVector, VectorId};
use crate::wal::writer::{WalError, WalOperation, WalWriter};

use serde_json::Value;
use std::collections::HashMap;
use std::path::PathBuf;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ShardError {
    #[error("hnsw: dimension mismatch (expected {expected}, got {got})")]
    DimensionMismatch { expected: usize, got: usize },
    #[error("bm25 error: {0}")]
    BM25(#[from] BM25Error),
    #[error("payload error: {0}")]
    Payload(#[from] PayloadError),
    #[error("wal error: {0}")]
    Wal(#[from] WalError),
    #[error("shard not initialized")]
    NotInitialized,
}

/// Configuration for a shard.
#[derive(Debug, Clone)]
pub struct ShardConfig {
    /// Shard data directory.
    pub data_dir: PathBuf,
    /// HNSW index config.
    pub hnsw: HnswConfig,
    /// BM25 index config.
    pub bm25: BM25Config,
    /// WAL fsync interval (entries).
    pub wal_sync_interval: usize,
    /// RRF constant for fusion.
    pub rrf_k: f32,
}

impl Default for ShardConfig {
    fn default() -> Self {
        Self {
            data_dir: PathBuf::from("./data/shard-0"),
            hnsw: HnswConfig::default(),
            bm25: BM25Config::default(),
            wal_sync_interval: 1000,
            rrf_k: 60.0,
        }
    }
}

/// A shard: the fundamental unit of the vector engine.
/// Owns all co-located indexes for fast hybrid search.
pub struct Shard {
    config: ShardConfig,
    /// Dense ANN index (HNSW).
    hnsw: HnswIndex,
    /// BM25 full-text index (Tantivy).
    bm25: BM25Index,
    /// Sparse vector index (inverted posting lists).
    sparse: SparseIndex,
    /// Payload store (RocksDB).
    payloads: PayloadStore,
    /// Write-ahead log.
    wal: WalWriter,
}

impl Shard {
    /// Create and open a new shard.
    pub fn open(config: ShardConfig) -> Result<Self, ShardError> {
        std::fs::create_dir_all(&config.data_dir)
            .map_err(|e| ShardError::Wal(WalError::Io(e)))?;

        let hnsw = HnswIndex::new(config.hnsw.clone());

        let bm25_dir = config.data_dir.join("bm25");
        std::fs::create_dir_all(&bm25_dir).map_err(|e| ShardError::Wal(WalError::Io(e)))?;
        let bm25 = BM25Index::new(&bm25_dir, &config.bm25)?;

        let sparse = SparseIndex::new();

        let payload_dir = config.data_dir.join("payloads");
        let payloads = PayloadStore::open(&payload_dir)?;

        let wal_path = config.data_dir.join("shard.wal");
        let wal = WalWriter::open(&wal_path, config.wal_sync_interval)?;

        Ok(Self {
            config,
            hnsw,
            bm25,
            sparse,
            payloads,
            wal,
        })
    }

    /// Upsert a document into the shard.
    ///
    /// Writes all provided data to the respective indexes and records in WAL.
    pub fn upsert(
        &mut self,
        id: VectorId,
        dense_vector: Option<Vec<f32>>,
        sparse_vector: Option<SparseVector>,
        text_fields: Option<HashMap<String, String>>,
        payload: Option<Value>,
    ) -> Result<u64, ShardError> {
        // Write to WAL first (durability)
        let seq = self.wal.append(WalOperation::Upsert {
            id: id.clone(),
            dense_vector: dense_vector.clone(),
            sparse_vector: sparse_vector.clone(),
            text_fields: text_fields.clone(),
            payload: payload.as_ref().map(|v| serde_json::to_vec(v).unwrap()),
        })?;

        // Update dense index
        if let Some(vec) = dense_vector {
            if vec.len() != self.config.hnsw.dim {
                return Err(ShardError::DimensionMismatch {
                    expected: self.config.hnsw.dim,
                    got: vec.len(),
                });
            }
            self.hnsw.insert(id.clone(), vec);
        }

        // Update sparse index
        if let Some(sv) = &sparse_vector {
            self.sparse.insert(id.clone(), sv);
        }

        // Update BM25 index
        if let Some(fields) = &text_fields {
            self.bm25.insert(&id, fields)?;
            self.bm25.commit()?;
        }

        // Update payload store
        if let Some(p) = &payload {
            self.payloads.put(&id, p)?;
        }

        Ok(seq)
    }

    /// Delete a document from all indexes.
    pub fn delete(&mut self, id: &VectorId) -> Result<u64, ShardError> {
        let seq = self.wal.append(WalOperation::Delete { id: id.clone() })?;
        self.hnsw.delete(id);
        self.sparse.delete(id);
        self.bm25.delete(id)?;
        self.bm25.commit()?;
        self.payloads.delete(id)?;
        Ok(seq)
    }

    /// Hybrid search: query dense, sparse, and/or BM25 indexes, then fuse results.
    pub fn search(
        &self,
        dense_query: Option<&[f32]>,
        sparse_query: Option<&SparseVector>,
        text_query: Option<&str>,
        k: usize,
        strategy: FusionStrategy,
        weights: &[f32],
        filter: Option<&FilterCondition>,
    ) -> Result<Vec<ScoredId>, ShardError> {
        let mut result_lists: Vec<Vec<ScoredId>> = Vec::new();

        // Dense ANN search
        if let Some(query) = dense_query {
            let dense_results = self.hnsw.search(query, k);
            result_lists.push(dense_results);
        }

        // Sparse search
        if let Some(query) = sparse_query {
            let sparse_results = self.sparse.search(query, k);
            result_lists.push(sparse_results);
        }

        // BM25 search
        if let Some(query) = text_query {
            let bm25_results = self.bm25.search(query, k)?;
            result_lists.push(bm25_results);
        }

        if result_lists.is_empty() {
            return Ok(Vec::new());
        }

        // Fuse
        let mut fused = fusion::fuse_results(
            &result_lists,
            weights,
            strategy,
            k,
            self.config.rrf_k,
        );

        // Apply payload filter if provided
        if let Some(filter_cond) = filter {
            fused.retain(|item| {
                self.payloads
                    .get(&item.id)
                    .ok()
                    .flatten()
                    .map_or(false, |p| PayloadStore::matches_filter(&p, filter_cond))
            });
        }

        Ok(fused)
    }

    /// Get a document's payload.
    pub fn get_payload(&self, id: &VectorId) -> Result<Option<Value>, ShardError> {
        Ok(self.payloads.get(id)?)
    }

    /// Get statistics about the shard.
    pub fn stats(&self) -> ShardStats {
        ShardStats {
            dense_count: self.hnsw.len(),
            sparse_count: self.sparse.len(),
            sparse_vocab_size: self.sparse.vocab_size(),
            payload_count: self.payloads.count(),
            wal_sequence: self.wal.sequence(),
            dim: self.config.hnsw.dim,
        }
    }

    /// Force WAL sync.
    pub fn sync_wal(&mut self) -> Result<(), ShardError> {
        Ok(self.wal.sync()?)
    }

    /// Create a snapshot of the shard's in-memory indexes.
    pub fn create_snapshot(&self, snapshot_path: &std::path::Path) -> Result<(), ShardError> {
        super::snapshot::create_snapshot(
            &self.hnsw,
            &self.sparse,
            self.wal.sequence(),
            snapshot_path,
        )
        .map_err(|e| ShardError::Wal(WalError::Io(std::io::Error::new(
            std::io::ErrorKind::Other,
            e.to_string(),
        ))))
    }

    /// Current WAL sequence number.
    pub fn wal_sequence(&self) -> u64 {
        self.wal.sequence()
    }

    /// The shard's data directory.
    pub fn data_dir(&self) -> &std::path::Path {
        &self.config.data_dir
    }

    /// Path to the shard's WAL file.
    pub fn wal_path(&self) -> std::path::PathBuf {
        self.config.data_dir.join("shard.wal")
    }

    /// Get the dense vector for a document.
    pub fn get_vector(&self, id: &VectorId) -> Option<Vec<f32>> {
        self.hnsw.get_vector(id).map(|v| v.to_vec())
    }

    /// The distance metric used by this shard.
    pub fn distance_metric(&self) -> DistanceMetric {
        self.config.hnsw.metric
    }

    /// MMR re-ranking: search then re-rank results for diversity.
    ///
    /// Performs a standard search, then applies Maximal Marginal Relevance
    /// to balance relevance with diversity.
    pub fn search_mmr(
        &self,
        dense_query: &[f32],
        k: usize,
        fetch_k: usize,
        lambda: f32,
        filter: Option<&FilterCondition>,
    ) -> Result<Vec<ScoredId>, ShardError> {
        // Fetch more candidates than needed for MMR re-ranking
        let candidates = self.search(
            Some(dense_query),
            None,
            None,
            fetch_k,
            FusionStrategy::Rrf,
            &[1.0],
            filter,
        )?;

        // Build ScoredDoc list with vectors
        let scored_docs: Vec<ScoredDoc> = candidates
            .into_iter()
            .filter_map(|r| {
                self.hnsw.get_vector(&r.id).map(|v| ScoredDoc {
                    id: r.id,
                    score: r.score,
                    vector: v.to_vec(),
                })
            })
            .collect();

        Ok(mmr::mmr_rerank(
            &scored_docs,
            dense_query,
            lambda,
            k,
            self.config.hnsw.metric,
        ))
    }
}

/// Shard statistics.
#[derive(Debug, Clone)]
pub struct ShardStats {
    pub dense_count: usize,
    pub sparse_count: usize,
    pub sparse_vocab_size: usize,
    pub payload_count: usize,
    pub wal_sequence: u64,
    pub dim: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::DistanceMetric;
    use serde_json::json;
    use tempfile::TempDir;

    fn test_shard() -> (Shard, TempDir) {
        let dir = TempDir::new().unwrap();
        let config = ShardConfig {
            data_dir: dir.path().to_path_buf(),
            hnsw: HnswConfig {
                m: 8,
                m0: 16,
                ef_construction: 50,
                ef_search: 32,
                metric: DistanceMetric::Cosine,
                dim: 4,
            },
            bm25: BM25Config {
                text_fields: vec!["text".to_string()],
                writer_heap_size: 15_000_000,
            },
            wal_sync_interval: 1,
            rrf_k: 60.0,
        };
        let shard = Shard::open(config).unwrap();
        (shard, dir)
    }

    #[test]
    fn test_upsert_and_dense_search() {
        let (mut shard, _dir) = test_shard();

        shard
            .upsert(
                "a".to_string(),
                Some(vec![1.0, 0.0, 0.0, 0.0]),
                None,
                None,
                Some(json!({"color": "red"})),
            )
            .unwrap();

        shard
            .upsert(
                "b".to_string(),
                Some(vec![0.0, 1.0, 0.0, 0.0]),
                None,
                None,
                Some(json!({"color": "blue"})),
            )
            .unwrap();

        let results = shard
            .search(
                Some(&[1.0, 0.0, 0.0, 0.0]),
                None,
                None,
                2,
                FusionStrategy::Rrf,
                &[1.0],
                None,
            )
            .unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "a");
    }

    #[test]
    fn test_hybrid_search() {
        let (mut shard, _dir) = test_shard();

        let mut text = HashMap::new();
        text.insert("text".to_string(), "rust programming language safety".to_string());
        shard
            .upsert(
                "doc1".to_string(),
                Some(vec![1.0, 0.0, 0.0, 0.0]),
                Some(SparseVector::new(vec![0, 1], vec![1.0, 0.5])),
                Some(text),
                None,
            )
            .unwrap();

        let mut text2 = HashMap::new();
        text2.insert("text".to_string(), "python scripting easy".to_string());
        shard
            .upsert(
                "doc2".to_string(),
                Some(vec![0.0, 1.0, 0.0, 0.0]),
                Some(SparseVector::new(vec![1, 2], vec![0.8, 1.0])),
                Some(text2),
                None,
            )
            .unwrap();

        // Hybrid: dense + sparse + BM25
        let results = shard
            .search(
                Some(&[1.0, 0.0, 0.0, 0.0]),
                Some(&SparseVector::new(vec![0], vec![1.0])),
                Some("rust"),
                2,
                FusionStrategy::Rrf,
                &[1.0, 1.0, 1.0],
                None,
            )
            .unwrap();

        assert!(!results.is_empty());
        // doc1 should rank higher (matches all three dimensions)
        assert_eq!(results[0].id, "doc1");
    }

    #[test]
    fn test_filtered_search() {
        let (mut shard, _dir) = test_shard();

        shard
            .upsert(
                "a".to_string(),
                Some(vec![1.0, 0.0, 0.0, 0.0]),
                None,
                None,
                Some(json!({"color": "red"})),
            )
            .unwrap();
        shard
            .upsert(
                "b".to_string(),
                Some(vec![0.9, 0.1, 0.0, 0.0]),
                None,
                None,
                Some(json!({"color": "blue"})),
            )
            .unwrap();

        let filter = FilterCondition::Eq("color".to_string(), json!("red"));
        let results = shard
            .search(
                Some(&[1.0, 0.0, 0.0, 0.0]),
                None,
                None,
                10,
                FusionStrategy::Rrf,
                &[1.0],
                Some(&filter),
            )
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "a");
    }

    #[test]
    fn test_delete() {
        let (mut shard, _dir) = test_shard();

        shard
            .upsert("a".to_string(), Some(vec![1.0, 0.0, 0.0, 0.0]), None, None, None)
            .unwrap();
        shard.delete(&"a".to_string()).unwrap();

        let stats = shard.stats();
        assert_eq!(stats.wal_sequence, 2); // upsert + delete
    }

    #[test]
    fn test_stats() {
        let (mut shard, _dir) = test_shard();

        shard
            .upsert(
                "a".to_string(),
                Some(vec![1.0, 0.0, 0.0, 0.0]),
                Some(SparseVector::new(vec![0], vec![1.0])),
                None,
                Some(json!({"x": 1})),
            )
            .unwrap();

        let stats = shard.stats();
        assert_eq!(stats.dense_count, 1);
        assert_eq!(stats.sparse_count, 1);
        assert_eq!(stats.payload_count, 1);
        assert_eq!(stats.wal_sequence, 1);
        assert_eq!(stats.dim, 4);
    }
}
