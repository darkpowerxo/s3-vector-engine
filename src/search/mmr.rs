//! Maximal Marginal Relevance (MMR) re-ranking.
//!
//! MMR balances relevance and diversity:
//!   MMR = argmax_{d ∈ R\S} [ λ · sim(d, q) - (1-λ) · max_{d_j ∈ S} sim(d, d_j) ]
//!
//! where λ controls the relevance/diversity trade-off (1.0 = pure relevance,
//! 0.0 = pure diversity).

use crate::distance;
use crate::types::{DistanceMetric, ScoredId};

/// A scored document with its vector for MMR computation.
pub struct ScoredDoc {
    pub id: String,
    pub score: f32,
    pub vector: Vec<f32>,
}

/// Apply MMR re-ranking to a list of scored documents.
///
/// # Arguments
/// * `docs` — candidate documents with vectors
/// * `query_vector` — the original query vector
/// * `lambda` — relevance/diversity trade-off (0.0–1.0, default 0.5)
/// * `k` — number of results to return
/// * `metric` — distance metric to use
///
/// # Returns
/// Top-k documents re-ranked by MMR score.
pub fn mmr_rerank(
    docs: &[ScoredDoc],
    query_vector: &[f32],
    lambda: f32,
    k: usize,
    metric: DistanceMetric,
) -> Vec<ScoredId> {
    if docs.is_empty() || k == 0 {
        return Vec::new();
    }

    let k = k.min(docs.len());
    let lambda = lambda.clamp(0.0, 1.0);

    // Pre-compute query similarities
    let query_sims: Vec<f32> = docs
        .iter()
        .map(|d| distance::distance(&d.vector, query_vector, metric))
        .collect();

    // Normalize query similarities to [0,1] for consistent MMR scoring
    let (min_q, max_q) = query_sims.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(mn, mx), &s| {
        (mn.min(s), mx.max(s))
    });
    let range_q = (max_q - min_q).max(1e-10);
    let norm_query_sims: Vec<f32> = query_sims.iter().map(|&s| (s - min_q) / range_q).collect();

    let mut selected: Vec<usize> = Vec::with_capacity(k);
    let mut remaining: Vec<usize> = (0..docs.len()).collect();
    let mut result: Vec<ScoredId> = Vec::with_capacity(k);

    for _ in 0..k {
        if remaining.is_empty() {
            break;
        }

        let mut best_idx_in_remaining = 0;
        let mut best_mmr = f32::NEG_INFINITY;

        for (ri, &doc_idx) in remaining.iter().enumerate() {
            let relevance = norm_query_sims[doc_idx];

            // Max similarity to any already-selected document
            let max_sim_to_selected = if selected.is_empty() {
                0.0
            } else {
                selected
                    .iter()
                    .map(|&sel_idx| {
                        distance::distance(&docs[doc_idx].vector, &docs[sel_idx].vector, metric)
                    })
                    .fold(f32::NEG_INFINITY, f32::max)
            };

            // Normalize inter-doc similarity roughly to [0,1]
            let norm_max_sim = if selected.is_empty() {
                0.0
            } else {
                // Cosine and IP naturally in [-1, 1], shift to [0, 1]
                (max_sim_to_selected + 1.0) / 2.0
            };

            let mmr_score = lambda * relevance - (1.0 - lambda) * norm_max_sim;

            if mmr_score > best_mmr {
                best_mmr = mmr_score;
                best_idx_in_remaining = ri;
            }
        }

        let chosen = remaining.swap_remove(best_idx_in_remaining);
        selected.push(chosen);
        result.push(ScoredId {
            id: docs[chosen].id.clone(),
            score: best_mmr,
        });
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_doc(id: &str, score: f32, vector: Vec<f32>) -> ScoredDoc {
        ScoredDoc {
            id: id.to_string(),
            score,
            vector,
        }
    }

    #[test]
    fn test_mmr_pure_relevance() {
        // λ=1.0 should return results in relevance order
        let query = vec![1.0, 0.0, 0.0];
        let docs = vec![
            make_doc("a", 0.9, vec![0.9, 0.1, 0.0]),  // most similar to query
            make_doc("b", 0.5, vec![0.5, 0.5, 0.0]),
            make_doc("c", 0.3, vec![0.1, 0.9, 0.0]),  // least similar
        ];

        let results = mmr_rerank(&docs, &query, 1.0, 3, DistanceMetric::Cosine);
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].id, "a");
    }

    #[test]
    fn test_mmr_diversity() {
        // λ=0.0 should maximize diversity (avoid similar docs)
        let query = vec![1.0, 0.0];
        let docs = vec![
            make_doc("a", 0.9, vec![1.0, 0.0]),
            make_doc("b", 0.85, vec![0.99, 0.01]), // very similar to "a"
            make_doc("c", 0.5, vec![0.0, 1.0]),    // very different
        ];

        let results = mmr_rerank(&docs, &query, 0.0, 3, DistanceMetric::Cosine);
        assert_eq!(results.len(), 3);
        // First pick doesn't care about diversity (no selected set), picks highest sim to query
        // Second pick: should pick "c" (most diverse from "a")
        assert_eq!(results[1].id, "c");
    }

    #[test]
    fn test_mmr_empty() {
        let results = mmr_rerank(&[], &[1.0, 0.0], 0.5, 10, DistanceMetric::Cosine);
        assert!(results.is_empty());
    }

    #[test]
    fn test_mmr_k_larger_than_docs() {
        let query = vec![1.0, 0.0];
        let docs = vec![
            make_doc("a", 0.9, vec![1.0, 0.0]),
            make_doc("b", 0.5, vec![0.0, 1.0]),
        ];

        let results = mmr_rerank(&docs, &query, 0.5, 10, DistanceMetric::Cosine);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_mmr_balanced() {
        // λ=0.5 balances relevance and diversity
        let query = vec![1.0, 0.0, 0.0];
        let docs = vec![
            make_doc("a", 0.9, vec![1.0, 0.0, 0.0]),
            make_doc("b", 0.85, vec![0.95, 0.05, 0.0]),
            make_doc("c", 0.7, vec![0.0, 1.0, 0.0]),
            make_doc("d", 0.6, vec![0.0, 0.0, 1.0]),
        ];

        let results = mmr_rerank(&docs, &query, 0.5, 4, DistanceMetric::Cosine);
        assert_eq!(results.len(), 4);
        // First should be "a" (most relevant)
        assert_eq!(results[0].id, "a");
    }
}
