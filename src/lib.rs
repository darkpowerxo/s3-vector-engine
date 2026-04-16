//! S3 Vector Engine — Rust shard engine for multimodal data warehouse.
//!
//! Provides:
//! - Dense ANN search (HNSW with configurable M, ef)
//! - BM25 full-text search (Tantivy)
//! - Sparse vector search (inverted posting lists)
//! - Product Quantization (PQ) compression
//! - RocksDB payload storage with filter operators
//! - Write-Ahead Log for crash recovery
//! - Hybrid search fusion (RRF, DBSF, Linear)
//!
//! The legacy `cosine_topk` / `cosine_topk_batch` functions are kept for
//! backward compatibility.

pub mod distance;
pub mod index;
pub mod proto;
pub mod search;
pub mod shard;
pub mod types;
pub mod wal;

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

/// Compute cosine similarity between a query vector and a matrix of vectors,
/// returning the top-k indices and scores.
///
/// Args:
///     query: 1D float32 array of shape (D,)
///     vectors: 2D float32 array of shape (N, D)
///     k: number of top results to return
///
/// Returns:
///     (indices, scores): tuple of 1D arrays — top-k vector indices and
///     their cosine similarity scores, sorted descending by score.
#[pyfunction]
fn cosine_topk<'py>(
    py: Python<'py>,
    query: PyReadonlyArray1<'py, f32>,
    vectors: PyReadonlyArray2<'py, f32>,
    k: usize,
) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f32>>)> {
    let q = query.as_array();
    let vecs = vectors.as_array();
    let n = vecs.nrows();
    let k = k.min(n);

    if k == 0 || n == 0 {
        return Ok((
            PyArray1::from_vec_bound(py, Vec::<i64>::new()),
            PyArray1::from_vec_bound(py, Vec::<f32>::new()),
        ));
    }

    // ── Normalize query (once) ──
    let q_norm: f32 = q.iter().map(|&x| x * x).sum::<f32>().sqrt().max(1e-10);
    let q_normalized: Vec<f32> = q.iter().map(|&x| x / q_norm).collect();

    // ── Cosine similarity for every row ──
    let mut sims: Vec<f32> = Vec::with_capacity(n);
    for i in 0..n {
        let row = vecs.row(i);
        let row_norm: f32 = row.iter().map(|&x| x * x).sum::<f32>().sqrt().max(1e-10);
        let inv_norm = 1.0 / row_norm;
        // dot(q_hat, v_hat) = sum(q_hat_i * v_i * inv_norm)
        let dot: f32 = q_normalized
            .iter()
            .zip(row.iter())
            .map(|(&q_val, &v_val)| q_val * v_val * inv_norm)
            .sum();
        sims.push(dot);
    }

    // ── Top-k selection — O(n) via partial sort ──
    let mut indices: Vec<usize> = (0..n).collect();
    if k < n {
        // Partition so that indices[n-k..] contain the k largest
        indices.select_nth_unstable_by(n - k, |&a, &b| {
            sims[a]
                .partial_cmp(&sims[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        indices.drain(..n - k);
    }

    // Sort the top-k descending by score
    indices.sort_unstable_by(|&a, &b| {
        sims[b]
            .partial_cmp(&sims[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    indices.truncate(k);

    let result_indices: Vec<i64> = indices.iter().map(|&i| i as i64).collect();
    let result_scores: Vec<f32> = indices.iter().map(|&i| sims[i]).collect();

    Ok((
        PyArray1::from_vec_bound(py, result_indices),
        PyArray1::from_vec_bound(py, result_scores),
    ))
}

/// Batch cosine similarity — process multiple queries at once.
///
/// Args:
///     queries: 2D float32 array of shape (Q, D)
///     vectors: 2D float32 array of shape (N, D)
///     k: number of top results per query
///
/// Returns:
///     (all_indices, all_scores): each of shape (Q, k)
#[pyfunction]
fn cosine_topk_batch<'py>(
    py: Python<'py>,
    queries: PyReadonlyArray2<'py, f32>,
    vectors: PyReadonlyArray2<'py, f32>,
    k: usize,
) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f32>>)> {
    let qs = queries.as_array();
    let vecs = vectors.as_array();
    let n = vecs.nrows();
    let num_queries = qs.nrows();
    let k = k.min(n);

    let mut all_indices: Vec<i64> = Vec::with_capacity(num_queries * k);
    let mut all_scores: Vec<f32> = Vec::with_capacity(num_queries * k);

    // Pre-compute row norms
    let row_inv_norms: Vec<f32> = (0..n)
        .map(|i| {
            let row = vecs.row(i);
            let norm: f32 = row.iter().map(|&x| x * x).sum::<f32>().sqrt().max(1e-10);
            1.0 / norm
        })
        .collect();

    for qi in 0..num_queries {
        let q = qs.row(qi);
        let q_norm: f32 = q.iter().map(|&x| x * x).sum::<f32>().sqrt().max(1e-10);
        let q_normalized: Vec<f32> = q.iter().map(|&x| x / q_norm).collect();

        let mut sims: Vec<f32> = Vec::with_capacity(n);
        for i in 0..n {
            let row = vecs.row(i);
            let dot: f32 = q_normalized
                .iter()
                .zip(row.iter())
                .map(|(&q_val, &v_val)| q_val * v_val * row_inv_norms[i])
                .sum();
            sims.push(dot);
        }

        let mut idxs: Vec<usize> = (0..n).collect();
        if k < n {
            idxs.select_nth_unstable_by(n - k, |&a, &b| {
                sims[a]
                    .partial_cmp(&sims[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            idxs.drain(..n - k);
        }
        idxs.sort_unstable_by(|&a, &b| {
            sims[b]
                .partial_cmp(&sims[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        idxs.truncate(k);

        for &i in &idxs {
            all_indices.push(i as i64);
            all_scores.push(sims[i]);
        }
        // Pad if fewer than k results
        for _ in idxs.len()..k {
            all_indices.push(-1);
            all_scores.push(0.0);
        }
    }

    Ok((
        PyArray1::from_vec_bound(py, all_indices),
        PyArray1::from_vec_bound(py, all_scores),
    ))
}

#[pymodule]
fn _engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Legacy functions
    m.add_function(wrap_pyfunction!(cosine_topk, m)?)?;
    m.add_function(wrap_pyfunction!(cosine_topk_batch, m)?)?;

    // New shard engine
    m.add_class::<PyShard>()?;
    m.add_class::<PyShardStats>()?;
    Ok(())
}

// ─── PyO3 Shard Wrapper ───────────────────────────────────────────────────

use shard::engine::{Shard, ShardConfig};
use index::hnsw::HnswConfig;
use index::bm25::BM25Config;
use types::{DistanceMetric, FusionStrategy, SparseVector};
use serde_json::Value;

/// Python-facing shard engine.
#[pyclass]
struct PyShard {
    inner: Shard,
}

#[pymethods]
impl PyShard {
    /// Create a new shard.
    ///
    /// Args:
    ///     data_dir: path to shard data directory
    ///     dim: vector dimensionality (default: 768)
    ///     m: HNSW M parameter (default: 16)
    ///     ef_construction: HNSW build-time ef (default: 200)
    ///     ef_search: HNSW search-time ef (default: 128)
    ///     metric: "cosine", "l2", or "ip" (default: "cosine")
    ///     text_fields: list of BM25 text field names (default: ["text"])
    #[new]
    #[pyo3(signature = (data_dir, dim=768, m=16, ef_construction=200, ef_search=128, metric="cosine", text_fields=None))]
    fn new(
        data_dir: &str,
        dim: usize,
        m: usize,
        ef_construction: usize,
        ef_search: usize,
        metric: &str,
        text_fields: Option<Vec<String>>,
    ) -> PyResult<Self> {
        let dist_metric = match metric {
            "cosine" => DistanceMetric::Cosine,
            "l2" => DistanceMetric::L2,
            "ip" | "inner_product" => DistanceMetric::InnerProduct,
            _ => return Err(pyo3::exceptions::PyValueError::new_err(
                format!("unknown metric: {metric}. use cosine, l2, or ip")
            )),
        };

        let config = ShardConfig {
            data_dir: std::path::PathBuf::from(data_dir),
            hnsw: HnswConfig {
                m,
                m0: m * 2,
                ef_construction,
                ef_search,
                metric: dist_metric,
                dim,
            },
            bm25: BM25Config {
                text_fields: text_fields.unwrap_or_else(|| vec!["text".to_string()]),
                writer_heap_size: 50_000_000,
            },
            wal_sync_interval: 1000,
            rrf_k: 60.0,
        };

        let inner = Shard::open(config).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("failed to open shard: {e}"))
        })?;

        Ok(Self { inner })
    }

    /// Upsert a document.
    ///
    /// Args:
    ///     id: document ID
    ///     dense_vector: optional float list (dense embedding)
    ///     sparse_indices: optional int list (sparse token IDs)
    ///     sparse_values: optional float list (sparse weights)
    ///     text_fields: optional dict of field_name → text
    ///     payload: optional dict (JSON metadata)
    ///
    /// Returns:
    ///     WAL sequence number
    #[pyo3(signature = (id, dense_vector=None, sparse_indices=None, sparse_values=None, text_fields=None, payload=None))]
    fn upsert(
        &mut self,
        id: String,
        dense_vector: Option<Vec<f32>>,
        sparse_indices: Option<Vec<u32>>,
        sparse_values: Option<Vec<f32>>,
        text_fields: Option<HashMap<String, String>>,
        payload: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<u64> {
        let sparse = match (sparse_indices, sparse_values) {
            (Some(idx), Some(val)) => Some(SparseVector::new(idx, val)),
            (None, None) => None,
            _ => return Err(pyo3::exceptions::PyValueError::new_err(
                "sparse_indices and sparse_values must both be provided or both be None"
            )),
        };

        let payload_value: Option<Value> = payload
            .map(|d| py_dict_to_json(d));

        self.inner
            .upsert(id, dense_vector, sparse, text_fields, payload_value)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Delete a document by ID.
    fn delete(&mut self, id: String) -> PyResult<u64> {
        self.inner
            .delete(&id)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Search the shard.
    ///
    /// Args:
    ///     dense_query: optional float list
    ///     sparse_indices: optional int list
    ///     sparse_values: optional float list
    ///     text_query: optional string for BM25
    ///     k: number of results (default: 10)
    ///     fusion: "rrf", "dbsf", or "linear" (default: "rrf")
    ///     weights: per-index weights (default: equal)
    ///
    /// Returns:
    ///     list of (id, score) tuples
    #[pyo3(signature = (dense_query=None, sparse_indices=None, sparse_values=None, text_query=None, k=10, fusion="rrf", weights=None))]
    fn search(
        &self,
        dense_query: Option<Vec<f32>>,
        sparse_indices: Option<Vec<u32>>,
        sparse_values: Option<Vec<f32>>,
        text_query: Option<&str>,
        k: usize,
        fusion: &str,
        weights: Option<Vec<f32>>,
    ) -> PyResult<Vec<(String, f32)>> {
        let strategy = match fusion {
            "rrf" => FusionStrategy::Rrf,
            "dbsf" => FusionStrategy::Dbsf,
            "linear" => FusionStrategy::Linear,
            _ => return Err(pyo3::exceptions::PyValueError::new_err(
                format!("unknown fusion: {fusion}. use rrf, dbsf, or linear")
            )),
        };

        let sparse_q = match (sparse_indices, sparse_values) {
            (Some(idx), Some(val)) => Some(SparseVector::new(idx, val)),
            (None, None) => None,
            _ => return Err(pyo3::exceptions::PyValueError::new_err(
                "sparse_indices and sparse_values must both be provided or both be None"
            )),
        };

        let w = weights.unwrap_or_else(|| vec![1.0; 3]);

        let results = self
            .inner
            .search(
                dense_query.as_deref(),
                sparse_q.as_ref(),
                text_query,
                k,
                strategy,
                &w,
                None,
            )
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        Ok(results.into_iter().map(|r| (r.id, r.score)).collect())
    }

    /// Get shard statistics.
    fn stats(&self) -> PyShardStats {
        let s = self.inner.stats();
        PyShardStats {
            dense_count: s.dense_count,
            sparse_count: s.sparse_count,
            sparse_vocab_size: s.sparse_vocab_size,
            payload_count: s.payload_count,
            wal_sequence: s.wal_sequence,
            dim: s.dim,
        }
    }

    /// Force WAL sync to disk.
    fn sync_wal(&mut self) -> PyResult<()> {
        self.inner
            .sync_wal()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }
}

#[pyclass]
#[derive(Debug, Clone)]
struct PyShardStats {
    #[pyo3(get)]
    dense_count: usize,
    #[pyo3(get)]
    sparse_count: usize,
    #[pyo3(get)]
    sparse_vocab_size: usize,
    #[pyo3(get)]
    payload_count: usize,
    #[pyo3(get)]
    wal_sequence: u64,
    #[pyo3(get)]
    dim: usize,
}

/// Convert a PyDict to serde_json::Value (basic implementation).
fn py_dict_to_json(dict: &Bound<'_, PyDict>) -> Value {
    let mut map = serde_json::Map::new();
    for (key, value) in dict.iter() {
        let k: String = key.extract().unwrap_or_default();
        let v = py_to_json(&value);
        map.insert(k, v);
    }
    Value::Object(map)
}

fn py_to_json(obj: &Bound<'_, pyo3::PyAny>) -> Value {
    if let Ok(v) = obj.extract::<bool>() {
        Value::Bool(v)
    } else if let Ok(v) = obj.extract::<i64>() {
        Value::Number(serde_json::Number::from(v))
    } else if let Ok(v) = obj.extract::<f64>() {
        serde_json::Number::from_f64(v)
            .map(Value::Number)
            .unwrap_or(Value::Null)
    } else if let Ok(v) = obj.extract::<String>() {
        Value::String(v)
    } else if let Ok(list) = obj.downcast::<pyo3::types::PyList>() {
        Value::Array(list.iter().map(|item| py_to_json(&item)).collect())
    } else if let Ok(dict) = obj.downcast::<PyDict>() {
        py_dict_to_json(dict)
    } else if obj.is_none() {
        Value::Null
    } else {
        Value::String(obj.str().map(|s| s.to_string()).unwrap_or_default())
    }
}
