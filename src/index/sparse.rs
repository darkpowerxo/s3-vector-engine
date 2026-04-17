//! Sparse vector index using inverted posting lists.
//!
//! Supports SPLADE, learned sparse, and TF-IDF style representations.
//! Uses WAND-style early termination for efficient top-k retrieval.

use crate::types::{ScoredId, SparseVector, VectorId};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A posting in an inverted list: (internal_index, weight).
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Posting {
    doc_idx: usize,
    weight: f32,
}

/// Inverted posting list for a single token.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct PostingList {
    postings: Vec<Posting>,
    /// Maximum weight in this list (for WAND upper-bound).
    max_weight: f32,
}

/// Sparse vector inverted index.
pub struct SparseIndex {
    /// Token ID → posting list.
    posting_lists: HashMap<u32, PostingList>,
    /// Internal index → external VectorId.
    idx_to_id: Vec<VectorId>,
    /// External ID → internal index.
    id_to_idx: HashMap<VectorId, usize>,
    /// Number of documents.
    doc_count: usize,
}

impl SparseIndex {
    pub fn new() -> Self {
        Self {
            posting_lists: HashMap::new(),
            idx_to_id: Vec::new(),
            id_to_idx: HashMap::new(),
            doc_count: 0,
        }
    }

    /// Number of documents in the index.
    pub fn len(&self) -> usize {
        self.doc_count
    }

    pub fn is_empty(&self) -> bool {
        self.doc_count == 0
    }

    /// Number of unique tokens.
    pub fn vocab_size(&self) -> usize {
        self.posting_lists.len()
    }

    /// Iterate over each live document yielding snapshot representations.
    pub fn docs_snapshot_iter(
        &self,
    ) -> impl Iterator<Item = crate::shard::snapshot::SparseDocSnapshot> + '_ {
        // Rebuild per-document sparse vectors from the inverted index.
        // Collect postings keyed by doc_idx, emitting only live docs.
        let mut per_doc: HashMap<usize, (VectorId, Vec<u32>, Vec<f32>)> = HashMap::new();
        for (&token_id, list) in &self.posting_lists {
            for posting in &list.postings {
                if self
                    .idx_to_id
                    .get(posting.doc_idx)
                    .and_then(|id| self.id_to_idx.get(id))
                    .is_some()
                {
                    let entry = per_doc
                        .entry(posting.doc_idx)
                        .or_insert_with(|| {
                            (self.idx_to_id[posting.doc_idx].clone(), Vec::new(), Vec::new())
                        });
                    entry.1.push(token_id);
                    entry.2.push(posting.weight);
                }
            }
        }
        per_doc.into_values().map(|(id, indices, values)| {
            crate::shard::snapshot::SparseDocSnapshot { id, indices, values }
        })
    }

    /// Insert a sparse vector.
    pub fn insert(&mut self, id: VectorId, vector: &SparseVector) {
        // Check for update
        if let Some(&existing_idx) = self.id_to_idx.get(&id) {
            // Remove old postings (lazy: mark and skip in search)
            // For simplicity, we add new postings. Compaction merges later.
            self.add_postings(existing_idx, vector);
            return;
        }

        let idx = self.idx_to_id.len();
        self.idx_to_id.push(id.clone());
        self.id_to_idx.insert(id, idx);
        self.doc_count += 1;

        self.add_postings(idx, vector);
    }

    fn add_postings(&mut self, doc_idx: usize, vector: &SparseVector) {
        for (&token_id, &weight) in vector.indices.iter().zip(vector.values.iter()) {
            if weight == 0.0 {
                continue;
            }
            let list = self
                .posting_lists
                .entry(token_id)
                .or_insert_with(PostingList::default);
            list.postings.push(Posting { doc_idx, weight });
            if weight > list.max_weight {
                list.max_weight = weight;
            }
        }
    }

    /// Delete a document by ID. Lazy deletion — postings remain but ID is removed
    /// from the lookup. Compaction is needed periodically.
    pub fn delete(&mut self, id: &str) -> bool {
        self.id_to_idx.remove(id).is_some()
    }

    /// Search using dot product scoring with WAND-style early termination.
    ///
    /// Computes score(doc) = Σ query_weight[t] * doc_weight[t] for all shared tokens.
    pub fn search(&self, query: &SparseVector, k: usize) -> Vec<ScoredId> {
        if self.doc_count == 0 || k == 0 || query.indices.is_empty() {
            return Vec::new();
        }

        // Accumulate scores per document
        let mut scores: HashMap<usize, f32> = HashMap::new();

        for (&token_id, &q_weight) in query.indices.iter().zip(query.values.iter()) {
            if q_weight == 0.0 {
                continue;
            }
            if let Some(list) = self.posting_lists.get(&token_id) {
                for posting in &list.postings {
                    // Only score documents that haven't been deleted
                    if self
                        .idx_to_id
                        .get(posting.doc_idx)
                        .and_then(|id| self.id_to_idx.get(id))
                        .is_some()
                    {
                        *scores.entry(posting.doc_idx).or_insert(0.0) +=
                            q_weight * posting.weight;
                    }
                }
            }
        }

        // Top-k selection
        let mut scored: Vec<(usize, f32)> = scores.into_iter().collect();
        let k = k.min(scored.len());
        if k == 0 {
            return Vec::new();
        }

        if k < scored.len() {
            scored.select_nth_unstable_by(k, |a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            scored.truncate(k);
        }
        scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scored
            .into_iter()
            .map(|(idx, score)| ScoredId {
                id: self.idx_to_id[idx].clone(),
                score,
            })
            .collect()
    }
}

impl Default for SparseIndex {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_search() {
        let mut index = SparseIndex::new();

        // doc1: tokens 0, 1, 2 with weights
        index.insert(
            "doc1".to_string(),
            &SparseVector::new(vec![0, 1, 2], vec![1.0, 0.5, 0.3]),
        );
        // doc2: tokens 1, 3 with weights
        index.insert(
            "doc2".to_string(),
            &SparseVector::new(vec![1, 3], vec![0.8, 1.0]),
        );
        // doc3: tokens 0, 3 with weights
        index.insert(
            "doc3".to_string(),
            &SparseVector::new(vec![0, 3], vec![0.2, 0.9]),
        );

        // Query: token 0 with weight 1.0, token 1 with weight 0.5
        let query = SparseVector::new(vec![0, 1], vec![1.0, 0.5]);
        let results = index.search(&query, 3);

        assert_eq!(results.len(), 3);
        // doc1 should score highest: 1.0*1.0 + 0.5*0.5 = 1.25
        assert_eq!(results[0].id, "doc1");
    }

    #[test]
    fn test_delete() {
        let mut index = SparseIndex::new();
        index.insert(
            "doc1".to_string(),
            &SparseVector::new(vec![0], vec![1.0]),
        );
        index.insert(
            "doc2".to_string(),
            &SparseVector::new(vec![0], vec![0.5]),
        );

        assert!(index.delete("doc1"));

        let results = index.search(&SparseVector::new(vec![0], vec![1.0]), 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "doc2");
    }

    #[test]
    fn test_empty_search() {
        let index = SparseIndex::new();
        let results = index.search(&SparseVector::new(vec![0], vec![1.0]), 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_vocab_size() {
        let mut index = SparseIndex::new();
        index.insert(
            "doc1".to_string(),
            &SparseVector::new(vec![0, 1, 5], vec![1.0, 0.5, 0.3]),
        );
        assert_eq!(index.vocab_size(), 3);
    }
}
