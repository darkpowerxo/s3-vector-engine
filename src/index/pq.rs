//! Product Quantization (PQ) — compress high-dimensional vectors into compact codes.
//!
//! Splits a D-dimensional vector into M sub-vectors, each quantized to one of
//! 256 centroids (fits in u8). A 768d vector becomes 48 bytes with M=48.
//!
//! Training uses K-means on a representative sample. Distance computation uses
//! Asymmetric Distance Computation (ADC): pre-compute a distance table from the
//! query to each centroid, then sum table lookups per code byte.

use rand::seq::SliceRandom;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// A PQ code: M bytes, one per sub-vector.
pub type PqCode = Vec<u8>;

/// Pre-computed distance table for ADC: [sub_vector][centroid] → distance.
pub type DistanceTable = Vec<Vec<f32>>;

/// Product Quantization codebook.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PqCodebook {
    /// Number of sub-vectors (M).
    pub n_subvectors: usize,
    /// Number of centroids per sub-vector (always 256 for u8 codes).
    pub n_centroids: usize,
    /// Centroid data: [sub_vector][centroid][sub_dim].
    pub centroids: Vec<Vec<Vec<f32>>>,
    /// Dimensionality of the original vector.
    pub dim: usize,
    /// Sub-vector dimensionality (dim / n_subvectors).
    pub sub_dim: usize,
}

impl PqCodebook {
    /// Train a PQ codebook using K-means on a sample of vectors.
    ///
    /// # Arguments
    /// * `vectors` — training vectors, each of length `dim`
    /// * `n_subvectors` — number of sub-vector splits (M)
    /// * `n_centroids` — centroids per sub-vector (256 for u8)
    /// * `n_iterations` — K-means iterations (default: 20)
    pub fn train(
        vectors: &[&[f32]],
        dim: usize,
        n_subvectors: usize,
        n_centroids: usize,
        n_iterations: usize,
    ) -> Self {
        assert!(dim % n_subvectors == 0, "dim must be divisible by n_subvectors");
        assert!(n_centroids <= 256, "n_centroids must be <= 256 for u8 codes");
        assert!(!vectors.is_empty(), "need at least one training vector");

        let sub_dim = dim / n_subvectors;

        // Train each sub-vector independently (parallelized)
        let centroids: Vec<Vec<Vec<f32>>> = (0..n_subvectors)
            .into_par_iter()
            .map(|m| {
                // Extract sub-vectors for this partition
                let sub_vecs: Vec<Vec<f32>> = vectors
                    .iter()
                    .map(|v| v[m * sub_dim..(m + 1) * sub_dim].to_vec())
                    .collect();
                kmeans(&sub_vecs, n_centroids, n_iterations)
            })
            .collect();

        Self {
            n_subvectors,
            n_centroids,
            centroids,
            dim,
            sub_dim,
        }
    }

    /// Encode a vector into a PQ code.
    pub fn encode(&self, vector: &[f32]) -> PqCode {
        assert_eq!(vector.len(), self.dim);
        let mut code = Vec::with_capacity(self.n_subvectors);

        for m in 0..self.n_subvectors {
            let sub = &vector[m * self.sub_dim..(m + 1) * self.sub_dim];
            // Find nearest centroid
            let mut best_idx = 0u8;
            let mut best_dist = f32::MAX;
            for (c, centroid) in self.centroids[m].iter().enumerate() {
                let d = l2_squared(sub, centroid);
                if d < best_dist {
                    best_dist = d;
                    best_idx = c as u8;
                }
            }
            code.push(best_idx);
        }
        code
    }

    /// Encode multiple vectors in parallel.
    pub fn encode_batch(&self, vectors: &[&[f32]]) -> Vec<PqCode> {
        vectors.par_iter().map(|v| self.encode(v)).collect()
    }

    /// Build an ADC distance table from a query vector.
    /// table[m][c] = distance from query sub-vector m to centroid c.
    pub fn build_distance_table(&self, query: &[f32]) -> DistanceTable {
        assert_eq!(query.len(), self.dim);
        (0..self.n_subvectors)
            .map(|m| {
                let sub_q = &query[m * self.sub_dim..(m + 1) * self.sub_dim];
                self.centroids[m]
                    .iter()
                    .map(|centroid| l2_squared(sub_q, centroid))
                    .collect()
            })
            .collect()
    }

    /// Compute asymmetric distance from a pre-built table to a PQ code.
    /// Result is the sum of squared L2 distances per sub-vector (lower = closer).
    #[inline]
    pub fn asymmetric_distance(&self, table: &DistanceTable, code: &PqCode) -> f32 {
        let mut dist = 0.0f32;
        for m in 0..self.n_subvectors {
            dist += table[m][code[m] as usize];
        }
        dist
    }

    /// Search PQ codes for the k nearest to the query. Returns (index, distance) pairs
    /// sorted ascending by distance (lower = closer).
    pub fn search(&self, query: &[f32], codes: &[PqCode], k: usize) -> Vec<(usize, f32)> {
        let table = self.build_distance_table(query);
        let mut scored: Vec<(usize, f32)> = codes
            .iter()
            .enumerate()
            .map(|(i, code)| (i, self.asymmetric_distance(&table, code)))
            .collect();

        // Partial sort for top-k (smallest distances)
        let k = k.min(scored.len());
        if k < scored.len() {
            scored.select_nth_unstable_by(k, |a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            scored.truncate(k);
        }
        scored.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        scored
    }

    /// Serialization size estimate (bytes).
    pub fn size_bytes(&self) -> usize {
        self.n_subvectors * self.n_centroids * self.sub_dim * std::mem::size_of::<f32>()
    }
}

/// Squared L2 distance (no sqrt — used for comparison).
#[inline]
fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

/// Simple K-means clustering.
fn kmeans(data: &[Vec<f32>], k: usize, max_iter: usize) -> Vec<Vec<f32>> {
    let n = data.len();
    let dim = data[0].len();
    let k = k.min(n);

    // Initialize centroids with random selection (k distinct)
    let mut rng = rand::thread_rng();
    let mut indices: Vec<usize> = (0..n).collect();
    indices.shuffle(&mut rng);
    let mut centroids: Vec<Vec<f32>> = indices[..k].iter().map(|&i| data[i].clone()).collect();

    let mut assignments = vec![0usize; n];

    for _ in 0..max_iter {
        // Assign each point to nearest centroid
        let mut changed = false;
        for i in 0..n {
            let mut best = 0;
            let mut best_dist = f32::MAX;
            for (c, centroid) in centroids.iter().enumerate() {
                let d = l2_squared(&data[i], centroid);
                if d < best_dist {
                    best_dist = d;
                    best = c;
                }
            }
            if assignments[i] != best {
                assignments[i] = best;
                changed = true;
            }
        }

        if !changed {
            break;
        }

        // Recompute centroids
        let mut sums = vec![vec![0.0f32; dim]; k];
        let mut counts = vec![0usize; k];
        for i in 0..n {
            let c = assignments[i];
            counts[c] += 1;
            for d in 0..dim {
                sums[c][d] += data[i][d];
            }
        }
        for c in 0..k {
            if counts[c] > 0 {
                let inv = 1.0 / counts[c] as f32;
                for d in 0..dim {
                    centroids[c][d] = sums[c][d] * inv;
                }
            }
        }
    }

    centroids
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    #[test]
    fn test_train_and_encode() {
        let dim = 16;
        let n_subvectors = 4;
        let n_centroids = 8; // Small for testing
        let mut rng = rand::thread_rng();

        let vectors: Vec<Vec<f32>> = (0..100)
            .map(|_| (0..dim).map(|_| rng.gen::<f32>() - 0.5).collect())
            .collect();
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

        let codebook = PqCodebook::train(&refs, dim, n_subvectors, n_centroids, 10);
        assert_eq!(codebook.n_subvectors, n_subvectors);
        assert_eq!(codebook.sub_dim, dim / n_subvectors);

        let code = codebook.encode(&vectors[0]);
        assert_eq!(code.len(), n_subvectors);

        // All codes should be < n_centroids
        for &c in &code {
            assert!((c as usize) < n_centroids);
        }
    }

    #[test]
    fn test_adc_search() {
        let dim = 16;
        let n_subvectors = 4;
        let n_centroids = 8;
        let mut rng = rand::thread_rng();

        let vectors: Vec<Vec<f32>> = (0..50)
            .map(|_| (0..dim).map(|_| rng.gen::<f32>() - 0.5).collect())
            .collect();
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

        let codebook = PqCodebook::train(&refs, dim, n_subvectors, n_centroids, 10);
        let codes: Vec<PqCode> = refs.iter().map(|v| codebook.encode(v)).collect();

        // Search for the first vector itself — should return index 0 in top results
        let results = codebook.search(&vectors[0], &codes, 5);
        assert_eq!(results.len(), 5);
        // The closest hit should be index 0 (itself)
        assert_eq!(results[0].0, 0);
    }

    #[test]
    fn test_distance_table_symmetry() {
        use rand::Rng;
        let dim = 8;
        let n_subvectors = 2;
        let n_centroids = 4;
        let mut rng = rand::thread_rng();

        // Need enough training vectors for k-means to converge with 4 centroids
        let vectors: Vec<Vec<f32>> = (0..40)
            .map(|_| (0..dim).map(|_| rng.gen::<f32>() - 0.5).collect())
            .collect();
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

        let codebook = PqCodebook::train(&refs, dim, n_subvectors, n_centroids, 5);
        let table = codebook.build_distance_table(&vectors[0]);

        assert_eq!(table.len(), n_subvectors);
        assert_eq!(table[0].len(), n_centroids);
    }
}
