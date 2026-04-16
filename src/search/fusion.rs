//! Hybrid search fusion: RRF, DBSF, Linear.
//!
//! Takes scored result lists from multiple indexes and produces a single
//! merged ranked list.

use crate::types::{FusionStrategy, ScoredId};
use std::collections::HashMap;

/// Fuse results from multiple index searches into a single ranked list.
///
/// # Arguments
/// * `result_lists` — scored results from each index (dense, sparse, bm25, etc.)
/// * `weights` — per-list weight (used by DBSF and Linear; ignored by RRF)
/// * `strategy` — fusion algorithm
/// * `k` — number of results to return
/// * `rrf_k` — RRF constant (default: 60)
pub fn fuse_results(
    result_lists: &[Vec<ScoredId>],
    weights: &[f32],
    strategy: FusionStrategy,
    k: usize,
    rrf_k: f32,
) -> Vec<ScoredId> {
    if result_lists.is_empty() || k == 0 {
        return Vec::new();
    }

    // If only one list, just return top-k from it
    if result_lists.len() == 1 {
        let mut results = result_lists[0].clone();
        results.truncate(k);
        return results;
    }

    match strategy {
        FusionStrategy::Rrf => fuse_rrf(result_lists, k, rrf_k),
        FusionStrategy::Dbsf => fuse_dbsf(result_lists, weights, k),
        FusionStrategy::Linear => fuse_linear(result_lists, weights, k),
    }
}

/// Reciprocal Rank Fusion: score = Σ 1/(rrf_k + rank_i)
///
/// Rank-based fusion, immune to score magnitude differences between indexes.
fn fuse_rrf(result_lists: &[Vec<ScoredId>], k: usize, rrf_k: f32) -> Vec<ScoredId> {
    let mut scores: HashMap<String, f32> = HashMap::new();

    for list in result_lists {
        for (rank, item) in list.iter().enumerate() {
            *scores.entry(item.id.clone()).or_insert(0.0) += 1.0 / (rrf_k + rank as f32 + 1.0);
        }
    }

    top_k_from_map(scores, k)
}

/// Distribution-Based Score Fusion: normalize each list to [0,1], then weighted sum.
fn fuse_dbsf(result_lists: &[Vec<ScoredId>], weights: &[f32], k: usize) -> Vec<ScoredId> {
    let mut scores: HashMap<String, f32> = HashMap::new();

    for (i, list) in result_lists.iter().enumerate() {
        let w = weights.get(i).copied().unwrap_or(1.0);
        if list.is_empty() {
            continue;
        }

        // Find min/max scores for normalization
        let min_score = list
            .iter()
            .map(|s| s.score)
            .fold(f32::INFINITY, f32::min);
        let max_score = list
            .iter()
            .map(|s| s.score)
            .fold(f32::NEG_INFINITY, f32::max);
        let range = (max_score - min_score).max(1e-10);

        for item in list {
            let normalized = (item.score - min_score) / range;
            *scores.entry(item.id.clone()).or_insert(0.0) += w * normalized;
        }
    }

    top_k_from_map(scores, k)
}

/// Linear weighted fusion: score = Σ weight_i * raw_score_i
fn fuse_linear(result_lists: &[Vec<ScoredId>], weights: &[f32], k: usize) -> Vec<ScoredId> {
    let mut scores: HashMap<String, f32> = HashMap::new();

    for (i, list) in result_lists.iter().enumerate() {
        let w = weights.get(i).copied().unwrap_or(1.0);
        for item in list {
            *scores.entry(item.id.clone()).or_insert(0.0) += w * item.score;
        }
    }

    top_k_from_map(scores, k)
}

/// Extract top-k from a score map, sorted descending.
fn top_k_from_map(scores: HashMap<String, f32>, k: usize) -> Vec<ScoredId> {
    let mut items: Vec<ScoredId> = scores
        .into_iter()
        .map(|(id, score)| ScoredId { id, score })
        .collect();

    let k = k.min(items.len());
    if k < items.len() {
        items.select_nth_unstable_by(k, |a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        items.truncate(k);
    }
    items.sort_unstable_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    items
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_list(ids_scores: &[(&str, f32)]) -> Vec<ScoredId> {
        ids_scores
            .iter()
            .map(|(id, score)| ScoredId {
                id: id.to_string(),
                score: *score,
            })
            .collect()
    }

    #[test]
    fn test_rrf_basic() {
        let dense = make_list(&[("a", 0.9), ("b", 0.8), ("c", 0.7)]);
        let sparse = make_list(&[("b", 1.5), ("a", 1.0), ("d", 0.5)]);

        let results = fuse_results(
            &[dense, sparse],
            &[1.0, 1.0],
            FusionStrategy::Rrf,
            3,
            60.0,
        );

        assert_eq!(results.len(), 3);
        // "a" rank 1+2=3, "b" rank 2+1=3 → both high. "a" and "b" should be top-2.
        let top_ids: Vec<&str> = results.iter().map(|r| r.id.as_str()).collect();
        assert!(top_ids.contains(&"a"));
        assert!(top_ids.contains(&"b"));
    }

    #[test]
    fn test_rrf_single_list() {
        let dense = make_list(&[("a", 0.9), ("b", 0.8)]);
        let results = fuse_results(&[dense], &[1.0], FusionStrategy::Rrf, 2, 60.0);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "a");
    }

    #[test]
    fn test_dbsf() {
        let dense = make_list(&[("a", 0.9), ("b", 0.5)]);
        let bm25 = make_list(&[("b", 15.0), ("a", 5.0)]);

        let results = fuse_results(
            &[dense, bm25],
            &[1.0, 1.0],
            FusionStrategy::Dbsf,
            2,
            60.0,
        );

        assert_eq!(results.len(), 2);
        // After normalization: dense a=1.0, b=0.0; bm25 b=1.0, a=0.0
        // Combined: a=1.0, b=1.0 → tied
    }

    #[test]
    fn test_linear() {
        let dense = make_list(&[("a", 0.9), ("b", 0.5)]);
        let sparse = make_list(&[("b", 2.0), ("c", 1.5)]);

        let results = fuse_results(
            &[dense, sparse],
            &[1.0, 0.5],
            FusionStrategy::Linear,
            3,
            60.0,
        );

        assert_eq!(results.len(), 3);
        // a: 0.9*1.0 = 0.9, b: 0.5*1.0 + 2.0*0.5 = 1.5, c: 1.5*0.5 = 0.75
        assert_eq!(results[0].id, "b");
    }

    #[test]
    fn test_empty_lists() {
        let results = fuse_results(&[], &[], FusionStrategy::Rrf, 10, 60.0);
        assert!(results.is_empty());
    }

    #[test]
    fn test_k_larger_than_results() {
        let dense = make_list(&[("a", 0.9)]);
        let results = fuse_results(&[dense], &[1.0], FusionStrategy::Rrf, 100, 60.0);
        assert_eq!(results.len(), 1);
    }
}
