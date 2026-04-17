//! Shard snapshot — serialize and restore in-memory indexes.
//!
//! Persists HNSW graph and sparse index to disk so a shard can be
//! restored without replaying the full WAL. RocksDB (payloads) and
//! Tantivy (BM25) are already disk-backed and don't need snapshot logic.

use crate::index::hnsw::{HnswConfig, HnswIndex};
use crate::index::sparse::SparseIndex;
use crate::types::{SparseVector, VectorId};

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SnapshotError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("serialization error: {0}")]
    Serde(#[from] bincode::Error),
}

// ── Serializable representations ───────────────────────────────────────────

/// Serializable form of an HNSW node.
#[derive(Serialize, Deserialize)]
pub struct HnswNodeSnapshot {
    pub id: VectorId,
    pub vector: Vec<f32>,
    pub neighbors: Vec<Vec<usize>>,
}

/// Serializable form of the full HNSW index.
#[derive(Serialize, Deserialize)]
pub(crate) struct HnswSnapshot {
    m: usize,
    m0: usize,
    ef_construction: usize,
    ef_search: usize,
    metric: u8, // 0=Cosine, 1=L2, 2=InnerProduct
    dim: usize,
    entry_point: Option<usize>,
    max_layer: usize,
    nodes: Vec<HnswNodeSnapshot>,
}

/// Serializable form of a sparse index posting.
#[derive(Serialize, Deserialize)]
pub struct SparseDocSnapshot {
    pub id: VectorId,
    pub indices: Vec<u32>,
    pub values: Vec<f32>,
}

/// Serializable form of the sparse index.
#[derive(Serialize, Deserialize)]
pub(crate) struct SparseSnapshot {
    docs: Vec<SparseDocSnapshot>,
}

/// Full shard snapshot (in-memory indexes only).
#[derive(Serialize, Deserialize)]
pub(crate) struct ShardSnapshot {
    hnsw: HnswSnapshot,
    sparse: SparseSnapshot,
    wal_sequence: u64,
}

// ── Public API ─────────────────────────────────────────────────────────────

/// Create a snapshot of the shard's in-memory indexes.
pub fn create_snapshot(
    hnsw: &HnswIndex,
    sparse: &SparseIndex,
    wal_sequence: u64,
    snapshot_path: &Path,
) -> Result<(), SnapshotError> {
    let hnsw_snap = hnsw.to_snapshot();
    let sparse_snap = sparse.to_snapshot();

    let snapshot = ShardSnapshot {
        hnsw: hnsw_snap,
        sparse: sparse_snap,
        wal_sequence,
    };

    let data = bincode::serialize(&snapshot)?;
    fs::write(snapshot_path, &data)?;

    Ok(())
}

/// Restore in-memory indexes from a snapshot.
///
/// Returns (HnswIndex, SparseIndex, wal_sequence_at_snapshot).
pub fn restore_snapshot(
    snapshot_path: &Path,
) -> Result<(HnswIndex, SparseIndex, u64), SnapshotError> {
    let data = fs::read(snapshot_path)?;
    let snapshot: ShardSnapshot = bincode::deserialize(&data)?;

    let hnsw = HnswIndex::from_snapshot(snapshot.hnsw);
    let sparse = SparseIndex::from_snapshot(snapshot.sparse);

    Ok((hnsw, sparse, snapshot.wal_sequence))
}

// ── Trait-like methods on indexes ──────────────────────────────────────────
// We add these as inherent methods via internal types to avoid orphan rule issues.

impl HnswIndex {
    /// Serialize the index to a snapshot representation.
    pub(crate) fn to_snapshot(&self) -> HnswSnapshot {
        let nodes: Vec<HnswNodeSnapshot> = self
            .nodes_snapshot_iter()
            .collect();

        HnswSnapshot {
            m: self.config().m,
            m0: self.config().m0,
            ef_construction: self.config().ef_construction,
            ef_search: self.config().ef_search,
            metric: match self.config().metric {
                crate::types::DistanceMetric::Cosine => 0,
                crate::types::DistanceMetric::L2 => 1,
                crate::types::DistanceMetric::InnerProduct => 2,
            },
            dim: self.config().dim,
            entry_point: self.entry_point(),
            max_layer: self.max_layer(),
            nodes,
        }
    }

    /// Restore from a snapshot representation.
    pub(crate) fn from_snapshot(snap: HnswSnapshot) -> Self {
        use crate::types::DistanceMetric;

        let metric = match snap.metric {
            1 => DistanceMetric::L2,
            2 => DistanceMetric::InnerProduct,
            _ => DistanceMetric::Cosine,
        };

        let config = HnswConfig {
            m: snap.m,
            m0: snap.m0,
            ef_construction: snap.ef_construction,
            ef_search: snap.ef_search,
            metric,
            dim: snap.dim,
        };

        let mut index = Self::new(config);
        index.restore_nodes(snap.nodes, snap.entry_point, snap.max_layer);
        index
    }
}

impl SparseIndex {
    /// Serialize the index to a snapshot representation.
    pub(crate) fn to_snapshot(&self) -> SparseSnapshot {
        let docs = self.docs_snapshot_iter().collect();
        SparseSnapshot { docs }
    }

    /// Restore from a snapshot representation.
    pub(crate) fn from_snapshot(snap: SparseSnapshot) -> Self {
        let mut index = Self::new();
        for doc in snap.docs {
            let sv = SparseVector::new(doc.indices, doc.values);
            index.insert(doc.id, &sv);
        }
        index
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::DistanceMetric;
    use tempfile::TempDir;

    #[test]
    fn test_snapshot_roundtrip() {
        let mut hnsw = HnswIndex::new(HnswConfig {
            m: 8,
            m0: 16,
            ef_construction: 50,
            ef_search: 32,
            metric: DistanceMetric::Cosine,
            dim: 4,
        });
        hnsw.insert("a".into(), vec![1.0, 0.0, 0.0, 0.0]);
        hnsw.insert("b".into(), vec![0.0, 1.0, 0.0, 0.0]);
        hnsw.insert("c".into(), vec![0.5, 0.5, 0.0, 0.0]);

        let mut sparse = SparseIndex::new();
        sparse.insert("a".into(), &SparseVector::new(vec![0, 1], vec![1.0, 0.5]));
        sparse.insert("b".into(), &SparseVector::new(vec![1, 2], vec![0.3, 0.8]));

        let dir = TempDir::new().unwrap();
        let snap_path = dir.path().join("snapshot.bin");

        // Create
        create_snapshot(&hnsw, &sparse, 42, &snap_path).unwrap();
        assert!(snap_path.exists());

        // Restore
        let (hnsw2, sparse2, seq) = restore_snapshot(&snap_path).unwrap();
        assert_eq!(seq, 42);
        assert_eq!(hnsw2.len(), 3);
        assert_eq!(sparse2.len(), 2);

        // Search restored HNSW
        let results = hnsw2.search(&[1.0, 0.0, 0.0, 0.0], 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "a");

        // Search restored sparse
        let results = sparse2.search(&SparseVector::new(vec![1], vec![1.0]), 2);
        assert!(!results.is_empty());
    }
}
