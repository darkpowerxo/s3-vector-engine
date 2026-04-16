//! Distance metric implementations with SIMD-friendly patterns.

use crate::types::DistanceMetric;

/// Compute distance between two vectors using the specified metric.
/// Returns a *similarity* score (higher = more similar) for all metrics.
#[inline]
pub fn distance(a: &[f32], b: &[f32], metric: DistanceMetric) -> f32 {
    match metric {
        DistanceMetric::Cosine => cosine_similarity(a, b),
        DistanceMetric::L2 => negative_l2(a, b),
        DistanceMetric::InnerProduct => inner_product(a, b),
    }
}

/// Cosine similarity: dot(a, b) / (|a| * |b|)
#[inline]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    let denom = (norm_a * norm_b).sqrt().max(1e-10);
    dot / denom
}

/// Negative L2 distance (so higher = closer, consistent with similarity ordering).
#[inline]
pub fn negative_l2(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        sum += d * d;
    }
    -sum.sqrt()
}

/// Inner product (dotproduct).
#[inline]
pub fn inner_product(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
    }
    dot
}

/// Normalize a vector in-place. Returns the original norm.
#[inline]
pub fn normalize(v: &mut [f32]) -> f32 {
    let norm: f32 = v.iter().map(|&x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        let inv = 1.0 / norm;
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
    norm
}
