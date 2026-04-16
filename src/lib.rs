//! Rust shard scanner for the S3-native vector engine.
//!
//! This is the hot path. Each shard's vectors are fetched from S3 by Python,
//! passed into this Rust extension for cosine similarity + top-k selection,
//! then discarded. No persistent RAM allocation.
//!
//! Performance advantages over numpy:
//!   - No GIL during computation
//!   - SIMD auto-vectorization with -C target-cpu=native
//!   - Cache-friendly single-pass scan
//!   - O(n) partial sort for top-k via select_nth_unstable

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

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
    m.add_function(wrap_pyfunction!(cosine_topk, m)?)?;
    m.add_function(wrap_pyfunction!(cosine_topk_batch, m)?)?;
    Ok(())
}
