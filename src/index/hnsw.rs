//! HNSW (Hierarchical Navigable Small World) graph index for dense ANN search.
//!
//! Supports configurable M, ef_construction, ef_search parameters.
//! Integrates with PQ for compressed storage mode.

use crate::distance::{self, normalize};
use crate::types::{DistanceMetric, ScoredId, VectorId};

use ordered_float::OrderedFloat;
use rand::Rng;
use std::collections::{BinaryHeap, HashMap, HashSet};

/// Configuration for the HNSW index.
#[derive(Debug, Clone)]
pub struct HnswConfig {
    /// Max edges per node per layer (default: 16).
    pub m: usize,
    /// Max edges for layer 0 (typically 2*M).
    pub m0: usize,
    /// ef parameter used during construction (default: 200).
    pub ef_construction: usize,
    /// Default ef parameter for search (default: 128).
    pub ef_search: usize,
    /// Distance metric.
    pub metric: DistanceMetric,
    /// Vector dimensionality.
    pub dim: usize,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            m: 16,
            m0: 32,
            ef_construction: 200,
            ef_search: 128,
            metric: DistanceMetric::Cosine,
            dim: 768,
        }
    }
}

/// Internal node in the HNSW graph.
struct HnswNode {
    /// The vector data (full precision).
    vector: Vec<f32>,
    /// Neighbors at each layer. neighbors[layer] = vec of node indices.
    neighbors: Vec<Vec<usize>>,
    /// External ID mapping.
    id: VectorId,
}

/// HNSW index.
pub struct HnswIndex {
    config: HnswConfig,
    nodes: Vec<HnswNode>,
    /// Maps external VectorId to internal index.
    id_to_idx: HashMap<VectorId, usize>,
    /// Entry point (index of the top-layer node).
    entry_point: Option<usize>,
    /// Maximum layer currently in the graph.
    max_layer: usize,
    /// Inverse of ln(M) — used for random level generation.
    level_mult: f64,
}

impl HnswIndex {
    pub fn new(config: HnswConfig) -> Self {
        let level_mult = 1.0 / (config.m as f64).ln();
        Self {
            config,
            nodes: Vec::new(),
            id_to_idx: HashMap::new(),
            entry_point: None,
            max_layer: 0,
            level_mult,
        }
    }

    /// Number of vectors in the index.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Returns the dimensionality.
    pub fn dim(&self) -> usize {
        self.config.dim
    }

    /// Generate a random level for a new node.
    fn random_level(&self) -> usize {
        let mut rng = rand::thread_rng();
        let r: f64 = rng.gen::<f64>();
        (-r.ln() * self.level_mult).floor() as usize
    }

    /// Distance between two internal nodes.
    #[allow(dead_code)]
    fn dist(&self, a: usize, b: usize) -> f32 {
        distance::distance(
            &self.nodes[a].vector,
            &self.nodes[b].vector,
            self.config.metric,
        )
    }

    /// Distance from a query vector to an internal node.
    fn dist_to_query(&self, query: &[f32], node_idx: usize) -> f32 {
        distance::distance(query, &self.nodes[node_idx].vector, self.config.metric)
    }

    /// Insert a vector into the index.
    pub fn insert(&mut self, id: VectorId, mut vector: Vec<f32>) -> bool {
        if vector.len() != self.config.dim {
            return false;
        }
        // Deduplicate: if ID exists, update in place
        if let Some(&existing) = self.id_to_idx.get(&id) {
            if self.config.metric == DistanceMetric::Cosine {
                normalize(&mut vector);
            }
            self.nodes[existing].vector = vector;
            return true;
        }

        if self.config.metric == DistanceMetric::Cosine {
            normalize(&mut vector);
        }

        let new_level = self.random_level();
        let new_idx = self.nodes.len();

        // Create node with empty neighbor lists for each layer up to new_level
        let neighbors = (0..=new_level).map(|_| Vec::new()).collect();
        self.nodes.push(HnswNode {
            vector,
            neighbors,
            id: id.clone(),
        });
        self.id_to_idx.insert(id, new_idx);

        // First node: just set as entry point
        if self.nodes.len() == 1 {
            self.entry_point = Some(new_idx);
            self.max_layer = new_level;
            return true;
        }

        let ep = self.entry_point.unwrap();
        let mut current_ep = ep;

        // Phase 1: Greedy traverse from top layer down to new_level+1
        let top = self.max_layer;
        for layer in (new_level + 1..=top).rev() {
            current_ep = self.search_layer_greedy(&self.nodes[new_idx].vector, current_ep, layer);
        }

        // Phase 2: Insert at layers [min(new_level, max_layer)..=0]
        let insert_top = new_level.min(self.max_layer);
        for layer in (0..=insert_top).rev() {
            let max_neighbors = if layer == 0 {
                self.config.m0
            } else {
                self.config.m
            };

            // Find candidates
            let candidates = self.search_layer(
                &self.nodes[new_idx].vector,
                current_ep,
                self.config.ef_construction,
                layer,
            );

            // Select best neighbors
            let neighbors: Vec<usize> = candidates
                .iter()
                .take(max_neighbors)
                .map(|&(idx, _)| idx)
                .collect();

            // Add bidirectional edges
            self.nodes[new_idx].neighbors[layer] = neighbors.clone();
            for &neighbor in &neighbors {
                if layer < self.nodes[neighbor].neighbors.len() {
                    self.nodes[neighbor].neighbors[layer].push(new_idx);
                    // Prune if over capacity
                    if self.nodes[neighbor].neighbors[layer].len() > max_neighbors {
                        self.prune_neighbors(neighbor, layer, max_neighbors);
                    }
                }
            }

            if !candidates.is_empty() {
                current_ep = candidates[0].0;
            }
        }

        // Update entry point if new node has a higher level
        if new_level > self.max_layer {
            self.entry_point = Some(new_idx);
            self.max_layer = new_level;
        }

        true
    }

    /// Greedy search on a single layer — returns the closest node.
    fn search_layer_greedy(&self, query: &[f32], entry: usize, layer: usize) -> usize {
        let mut current = entry;
        let mut current_dist = self.dist_to_query(query, current);

        loop {
            let mut improved = false;
            if layer < self.nodes[current].neighbors.len() {
                for &neighbor in &self.nodes[current].neighbors[layer] {
                    let d = self.dist_to_query(query, neighbor);
                    if d > current_dist {
                        // Higher similarity = better
                        current_dist = d;
                        current = neighbor;
                        improved = true;
                    }
                }
            }
            if !improved {
                break;
            }
        }
        current
    }

    /// Search a single layer with ef candidates. Returns sorted (idx, score) pairs, best first.
    fn search_layer(
        &self,
        query: &[f32],
        entry: usize,
        ef: usize,
        layer: usize,
    ) -> Vec<(usize, f32)> {
        let entry_dist = self.dist_to_query(query, entry);

        // Max-heap for candidates (best = highest similarity)
        let mut candidates: BinaryHeap<(OrderedFloat<f32>, usize)> = BinaryHeap::new();
        // Min-heap for result set (worst = lowest similarity on top)
        let mut results: BinaryHeap<std::cmp::Reverse<(OrderedFloat<f32>, usize)>> =
            BinaryHeap::new();
        let mut visited: HashSet<usize> = HashSet::new();

        candidates.push((OrderedFloat(entry_dist), entry));
        results.push(std::cmp::Reverse((OrderedFloat(entry_dist), entry)));
        visited.insert(entry);

        while let Some((OrderedFloat(c_dist), c_idx)) = candidates.pop() {
            // If the best candidate is worse than the worst result, stop
            let worst_result = results.peek().unwrap().0 .0 .0;
            if c_dist < worst_result && results.len() >= ef {
                break;
            }

            if layer < self.nodes[c_idx].neighbors.len() {
                for &neighbor in &self.nodes[c_idx].neighbors[layer] {
                    if visited.contains(&neighbor) {
                        continue;
                    }
                    visited.insert(neighbor);

                    let n_dist = self.dist_to_query(query, neighbor);
                    let worst_result = results.peek().unwrap().0 .0 .0;

                    if n_dist > worst_result || results.len() < ef {
                        candidates.push((OrderedFloat(n_dist), neighbor));
                        results.push(std::cmp::Reverse((OrderedFloat(n_dist), neighbor)));
                        if results.len() > ef {
                            results.pop(); // Remove worst
                        }
                    }
                }
            }
        }

        // Extract and sort by score descending
        let mut result_vec: Vec<(usize, f32)> = results
            .into_iter()
            .map(|std::cmp::Reverse((OrderedFloat(score), idx))| (idx, score))
            .collect();
        result_vec.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        result_vec
    }

    /// Prune a node's neighbors on a given layer to at most max_count.
    fn prune_neighbors(&mut self, node: usize, layer: usize, max_count: usize) {
        let query_vec = self.nodes[node].vector.clone();
        let neighbor_indices: Vec<usize> = self.nodes[node].neighbors[layer].clone();

        let mut scored: Vec<(usize, f32)> = neighbor_indices
            .iter()
            .map(|&n| {
                let d = distance::distance(
                    &query_vec,
                    &self.nodes[n].vector,
                    self.config.metric,
                );
                (n, d)
            })
            .collect();
        scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scored.truncate(max_count);
        self.nodes[node].neighbors[layer] = scored.into_iter().map(|(idx, _)| idx).collect();
    }

    /// Search the index for the k nearest neighbors.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<ScoredId> {
        self.search_with_ef(query, k, self.config.ef_search)
    }

    /// Search with a custom ef parameter (higher ef = more accurate, slower).
    pub fn search_with_ef(&self, query: &[f32], k: usize, ef: usize) -> Vec<ScoredId> {
        if self.nodes.is_empty() || k == 0 {
            return Vec::new();
        }

        let mut search_query;
        let query = if self.config.metric == DistanceMetric::Cosine {
            search_query = query.to_vec();
            normalize(&mut search_query);
            &search_query
        } else {
            query
        };

        let ep = self.entry_point.unwrap();
        let mut current_ep = ep;

        // Traverse from top layer down to layer 1
        for layer in (1..=self.max_layer).rev() {
            current_ep = self.search_layer_greedy(query, current_ep, layer);
        }

        // Search layer 0 with ef candidates
        let ef = ef.max(k);
        let candidates = self.search_layer(query, current_ep, ef, 0);

        candidates
            .into_iter()
            .filter(|(idx, _)| self.id_to_idx.contains_key(&self.nodes[*idx].id))
            .take(k)
            .map(|(idx, score)| ScoredId {
                id: self.nodes[idx].id.clone(),
                score,
            })
            .collect()
    }

    /// Delete a vector by ID. Marks it as deleted (lazy deletion).
    pub fn delete(&mut self, id: &str) -> bool {
        // Simple approach: remove from id_to_idx so it's not returned.
        // The node remains in the graph but its ID won't be in results.
        // Full graph compaction can be done periodically.
        self.id_to_idx.remove(id).is_some()
    }

    /// Get a vector by ID.
    pub fn get_vector(&self, id: &str) -> Option<&[f32]> {
        self.id_to_idx
            .get(id)
            .map(|&idx| self.nodes[idx].vector.as_slice())
    }

    /// Get all IDs in the index.
    pub fn ids(&self) -> Vec<&VectorId> {
        self.id_to_idx.keys().collect()
    }

    // ── Snapshot accessors ─────────────────────────────────────────────

    /// Returns a reference to the index configuration.
    pub fn config(&self) -> &HnswConfig {
        &self.config
    }

    /// Returns the entry point index, if any.
    pub fn entry_point(&self) -> Option<usize> {
        self.entry_point
    }

    /// Returns the maximum HNSW layer in the graph.
    pub fn max_layer(&self) -> usize {
        self.max_layer
    }

    /// Iterate over nodes yielding snapshot-friendly tuples.
    pub fn nodes_snapshot_iter(
        &self,
    ) -> impl Iterator<Item = crate::shard::snapshot::HnswNodeSnapshot> + '_ {
        self.nodes.iter().map(|n| crate::shard::snapshot::HnswNodeSnapshot {
            id: n.id.clone(),
            vector: n.vector.clone(),
            neighbors: n.neighbors.clone(),
        })
    }

    /// Restore nodes from snapshot data (used by `from_snapshot`).
    pub fn restore_nodes(
        &mut self,
        nodes: Vec<crate::shard::snapshot::HnswNodeSnapshot>,
        entry_point: Option<usize>,
        max_layer: usize,
    ) {
        for (idx, n) in nodes.into_iter().enumerate() {
            self.id_to_idx.insert(n.id.clone(), idx);
            self.nodes.push(HnswNode {
                id: n.id,
                vector: n.vector,
                neighbors: n.neighbors,
            });
        }
        self.entry_point = entry_point;
        self.max_layer = max_layer;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(dim: usize) -> HnswConfig {
        HnswConfig {
            m: 8,
            m0: 16,
            ef_construction: 100,
            ef_search: 64,
            metric: DistanceMetric::Cosine,
            dim,
        }
    }

    #[test]
    fn test_insert_and_search() {
        let mut index = HnswIndex::new(make_config(4));
        index.insert("a".into(), vec![1.0, 0.0, 0.0, 0.0]);
        index.insert("b".into(), vec![0.0, 1.0, 0.0, 0.0]);
        index.insert("c".into(), vec![1.0, 0.1, 0.0, 0.0]);

        let results = index.search(&[1.0, 0.0, 0.0, 0.0], 2);
        assert_eq!(results.len(), 2);
        // "a" should be the top result (exact match)
        assert_eq!(results[0].id, "a");
        // "c" should be second (close to query)
        assert_eq!(results[1].id, "c");
    }

    #[test]
    fn test_empty_index() {
        let index = HnswIndex::new(make_config(4));
        let results = index.search(&[1.0, 0.0, 0.0, 0.0], 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_delete() {
        let mut index = HnswIndex::new(make_config(4));
        index.insert("a".into(), vec![1.0, 0.0, 0.0, 0.0]);
        index.insert("b".into(), vec![0.0, 1.0, 0.0, 0.0]);
        assert!(index.delete("a"));
        assert!(!index.delete("nonexistent"));
    }

    #[test]
    fn test_recall_100_vectors() {
        let dim = 32;
        let n = 100;
        let mut index = HnswIndex::new(make_config(dim));

        let mut rng = rand::thread_rng();
        let mut vectors: Vec<Vec<f32>> = Vec::new();

        for i in 0..n {
            let v: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() - 0.5).collect();
            index.insert(format!("v{i}"), v.clone());
            vectors.push(v);
        }

        // Query with the first vector — it should find itself as top-1
        let results = index.search(&vectors[0], 1);
        assert_eq!(results[0].id, "v0");
    }
}
