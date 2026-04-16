//! Common types shared across the engine.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Unique identifier for a vector/document.
pub type VectorId = String;

/// A scored search result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScoredId {
    pub id: VectorId,
    pub score: f32,
}

/// Distance metric used for dense vector search.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DistanceMetric {
    Cosine,
    L2,
    InnerProduct,
}

impl Default for DistanceMetric {
    fn default() -> Self {
        Self::Cosine
    }
}

impl fmt::Display for DistanceMetric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DistanceMetric::Cosine => write!(f, "cosine"),
            DistanceMetric::L2 => write!(f, "l2"),
            DistanceMetric::InnerProduct => write!(f, "ip"),
        }
    }
}

/// Sparse vector: parallel arrays of token indices and float weights.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseVector {
    pub indices: Vec<u32>,
    pub values: Vec<f32>,
}

impl SparseVector {
    pub fn new(indices: Vec<u32>, values: Vec<f32>) -> Self {
        assert_eq!(indices.len(), values.len());
        Self { indices, values }
    }
}

/// Strategy for fusing results from multiple index types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FusionStrategy {
    /// Reciprocal Rank Fusion: score = Σ 1/(k + rank_i)
    Rrf,
    /// Distribution-Based Score Fusion: normalize to [0,1] then weighted sum
    Dbsf,
    /// Linear weighted combination of raw scores
    Linear,
}

impl Default for FusionStrategy {
    fn default() -> Self {
        Self::Rrf
    }
}
