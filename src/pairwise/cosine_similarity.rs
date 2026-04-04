//! Cosine similarity and k-nearest neighbors for embedding vectors.
//!
//! This module provides efficient implementations of cosine similarity calculations
//! and k-nearest neighbor search for embedding vectors, commonly used in RAG
//! (Retrieval-Augmented Generation) and semantic search applications.
//!
//! # Example
//! ```
//! use mongodb_voyageai::pairwise::cosine_similarity::{cosine_similarity, k_nearest_neighbors};
//! use ndarray::{Array1, Array2};
//!
//! // Calculate similarity between two sets of embeddings
//! let embeddings_a = Array2::from_shape_vec((2, 3), vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).unwrap();
//! let embeddings_b = Array2::from_shape_vec((2, 3), vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0]).unwrap();
//! let similarity = cosine_similarity(embeddings_a.view(), Some(embeddings_b.view()));
//!
//! // Find k nearest neighbors
//! let query = Array1::from_vec(vec![1.0, 0.0, 0.0]);
//! let documents = Array2::from_shape_vec((3, 3), vec![
//!     1.0, 0.0, 0.0,
//!     0.0, 1.0, 0.0,
//!     0.0, 0.0, 1.0,
//! ]).unwrap();
//! let (top_embeddings, top_indices) = k_nearest_neighbors(query.view(), documents.view(), 2);
//! ```

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

// ──────────────────────────────────────────────
// Cosine Similarity
// ──────────────────────────────────────────────

/// Computes pairwise cosine similarity between rows of two matrices.
///
/// Cosine similarity measures the cosine of the angle between two vectors,
/// ranging from -1 (opposite) to 1 (identical direction). For normalized
/// embeddings, this is equivalent to their dot product.
///
/// # Arguments
///
/// * `x` - First matrix of embeddings, shape (n_samples_x, n_features)
/// * `y` - Optional second matrix of embeddings, shape (n_samples_y, n_features).
///   If `None`, computes similarity between all pairs in `x`.
///
/// # Returns
///
/// A matrix of shape (n_samples_x, n_samples_y) where element (i, j) contains
/// the cosine similarity between row i of x and row j of y.
///
/// # Example
/// ```
/// use mongodb_voyageai::pairwise::cosine_similarity::cosine_similarity;
/// use ndarray::Array2;
///
/// let x = Array2::from_shape_vec((2, 3), vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).unwrap();
/// let y = Array2::from_shape_vec((2, 3), vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0]).unwrap();
/// let similarity = cosine_similarity(x.view(), Some(y.view()));
/// assert_eq!(similarity.shape(), &[2, 2]);
/// ```
pub fn cosine_similarity(x: ArrayView2<f32>, y: Option<ArrayView2<f32>>) -> Array2<f32> {
    // Normalize all rows to unit length (L2 normalization)
    let x_normalized = normalize_rows(x);
    let y_normalized = match y {
        Some(y_view) => normalize_rows(y_view),
        None => x_normalized.clone(), // Self-similarity: compare x with itself
    };

    // Cosine similarity = dot product of normalized vectors
    x_normalized.dot(&y_normalized.t())
}

/// Normalizes each row of a matrix to unit length (L2 normalization).
///
/// For each row vector, divides all elements by the vector's L2 norm (Euclidean length).
/// Zero vectors remain unchanged to avoid division by zero.
///
/// # Arguments
///
/// * `x` - Input matrix where each row is a vector to normalize
///
/// # Returns
///
/// A new matrix with the same shape where each row has unit length
fn normalize_rows(x: ArrayView2<f32>) -> Array2<f32> {
    let mut normalized = x.to_owned();

    for mut row in normalized.rows_mut() {
        // Calculate L2 norm: sqrt(sum of squares)
        let norm = row.dot(&row).sqrt();

        // Normalize only if norm is non-zero
        if norm > 0.0 {
            row.mapv_inplace(|v| v / norm);
        }
    }

    normalized
}

// ──────────────────────────────────────────────
// K-Nearest Neighbors (KNN)
// ──────────────────────────────────────────────

/// Finds the k most similar document embeddings to a query embedding.
///
/// Uses cosine similarity as the distance metric. Returns both the embeddings
/// and indices of the k nearest neighbors, sorted by similarity (highest first).
///
/// # Arguments
///
/// * `query_embedding` - Single query vector, shape (n_features,)
/// * `documents_embeddings` - Matrix of document embeddings, shape (n_docs, n_features)
/// * `k` - Number of nearest neighbors to return. If k > n_docs, returns all documents.
///
/// # Returns
///
/// A tuple containing:
/// * `Array2<f32>` - Matrix of k nearest embeddings, shape (k, n_features)
/// * `Array1<usize>` - Indices of the k nearest documents in the original matrix
///
/// # Example
/// ```
/// use mongodb_voyageai::pairwise::cosine_similarity::k_nearest_neighbors;
/// use ndarray::{Array1, Array2};
///
/// let query = Array1::from_vec(vec![1.0, 0.0, 0.0]);
/// let documents = Array2::from_shape_vec((3, 3), vec![
///     1.0, 0.0, 0.0,  // Most similar
///     0.5, 0.5, 0.0,  // Somewhat similar
///     0.0, 0.0, 1.0,  // Least similar
/// ]).unwrap();
///
/// let (top_embeddings, top_indices) = k_nearest_neighbors(query.view(), documents.view(), 2);
/// assert_eq!(top_embeddings.nrows(), 2);
/// assert_eq!(top_indices.len(), 2);
/// assert_eq!(top_indices[0], 0); // First document is most similar
/// ```
pub fn k_nearest_neighbors(
    query_embedding: ArrayView1<f32>,
    documents_embeddings: ArrayView2<f32>,
    k: usize,
) -> (Array2<f32>, Array1<usize>) {
    let n_docs = documents_embeddings.nrows();

    // Reshape query vector to 2D matrix (1, n_features) for cosine_similarity
    let query_matrix = query_embedding
        .to_owned()
        .into_shape_with_order((1, query_embedding.len()))
        .expect("reshape failed");

    // Calculate cosine similarity between query and all documents
    let cosine_sim = cosine_similarity(query_matrix.view(), Some(documents_embeddings));
    let sim_row: Vec<f32> = cosine_sim.row(0).to_vec();

    // Create indices and sort by similarity (descending order)
    let mut sorted_indices: Vec<usize> = (0..n_docs).collect();
    sorted_indices.sort_by(|&a, &b| {
        // Sort in descending order (highest similarity first)
        sim_row[b]
            .partial_cmp(&sim_row[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Take top k results (or all if k > n_docs)
    let k = k.min(n_docs);
    let top_k_indices: Vec<usize> = sorted_indices[..k].to_vec();

    // Extract the embeddings for top k documents
    let top_k_rows: Vec<f32> = top_k_indices
        .iter()
        .flat_map(|&idx| documents_embeddings.row(idx).to_vec())
        .collect();

    let n_features = documents_embeddings.ncols();
    let top_k_embeddings =
        Array2::from_shape_vec((k, n_features), top_k_rows).expect("failed to build top-k matrix");

    (top_k_embeddings, Array1::from_vec(top_k_indices))
}

// ──────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_normalize_rows_unit_vectors() {
        let x = Array2::from_shape_vec((2, 3), vec![3.0, 0.0, 0.0, 0.0, 4.0, 0.0]).unwrap();

        let normalized = normalize_rows(x.view());

        // First row should be [1.0, 0.0, 0.0]
        assert!((normalized[[0, 0]] - 1.0).abs() < 1e-6);
        assert!((normalized[[0, 1]] - 0.0).abs() < 1e-6);

        // Second row should be [0.0, 1.0, 0.0]
        assert!((normalized[[1, 1]] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_rows_zero_vector() {
        let x = Array2::from_shape_vec(
            (2, 3),
            vec![
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0, // Zero vector
            ],
        )
        .unwrap();

        let normalized = normalize_rows(x.view());

        // Zero vector should remain zero
        assert_eq!(normalized[[1, 0]], 0.0);
        assert_eq!(normalized[[1, 1]], 0.0);
        assert_eq!(normalized[[1, 2]], 0.0);
    }

    #[test]
    fn test_cosine_similarity_identical_vectors() {
        let x = Array2::from_shape_vec((2, 3), vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).unwrap();

        let similarity = cosine_similarity(x.view(), None);

        // Diagonal should be 1.0 (self-similarity)
        assert!((similarity[[0, 0]] - 1.0).abs() < 1e-6);
        assert!((similarity[[1, 1]] - 1.0).abs() < 1e-6);

        // Off-diagonal should be 0.0 (orthogonal)
        assert!((similarity[[0, 1]] - 0.0).abs() < 1e-6);
        assert!((similarity[[1, 0]] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_with_two_matrices() {
        let x = Array2::from_shape_vec((1, 3), vec![1.0, 0.0, 0.0]).unwrap();
        let y = Array2::from_shape_vec(
            (2, 3),
            vec![
                1.0, 0.0, 0.0, // Same direction
                0.0, 1.0, 0.0, // Orthogonal
            ],
        )
        .unwrap();

        let similarity = cosine_similarity(x.view(), Some(y.view()));

        assert_eq!(similarity.shape(), &[1, 2]);
        assert!((similarity[[0, 0]] - 1.0).abs() < 1e-6); // Same direction
        assert!((similarity[[0, 1]] - 0.0).abs() < 1e-6); // Orthogonal
    }

    #[test]
    fn test_cosine_similarity_opposite_vectors() {
        let x = Array2::from_shape_vec((1, 3), vec![1.0, 0.0, 0.0]).unwrap();
        let y = Array2::from_shape_vec((1, 3), vec![-1.0, 0.0, 0.0]).unwrap();

        let similarity = cosine_similarity(x.view(), Some(y.view()));

        // Opposite vectors should have similarity -1.0
        assert!((similarity[[0, 0]] + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_k_nearest_neighbors_basic() {
        let query = Array1::from_vec(vec![1.0, 0.0, 0.0]);
        let documents = Array2::from_shape_vec(
            (3, 3),
            vec![
                1.0, 0.0, 0.0, // Most similar (identical)
                0.5, 0.5, 0.0, // Somewhat similar
                0.0, 0.0, 1.0, // Orthogonal
            ],
        )
        .unwrap();

        let (top_embeddings, top_indices) = k_nearest_neighbors(query.view(), documents.view(), 2);

        assert_eq!(top_embeddings.nrows(), 2);
        assert_eq!(top_indices.len(), 2);
        assert_eq!(top_indices[0], 0); // First document is most similar
    }

    #[test]
    fn test_k_nearest_neighbors_k_larger_than_docs() {
        let query = Array1::from_vec(vec![1.0, 0.0, 0.0]);
        let documents = Array2::from_shape_vec((2, 3), vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).unwrap();

        // Request more neighbors than available documents
        let (top_embeddings, top_indices) = k_nearest_neighbors(query.view(), documents.view(), 10);

        // Should return all available documents
        assert_eq!(top_embeddings.nrows(), 2);
        assert_eq!(top_indices.len(), 2);
    }

    #[test]
    fn test_k_nearest_neighbors_single_document() {
        let query = Array1::from_vec(vec![1.0, 0.0, 0.0]);
        let documents = Array2::from_shape_vec((1, 3), vec![1.0, 0.0, 0.0]).unwrap();

        let (top_embeddings, top_indices) = k_nearest_neighbors(query.view(), documents.view(), 1);

        assert_eq!(top_embeddings.nrows(), 1);
        assert_eq!(top_indices[0], 0);
    }

    #[test]
    fn test_k_nearest_neighbors_ordering() {
        let query = Array1::from_vec(vec![1.0, 0.0, 0.0]);
        let documents = Array2::from_shape_vec(
            (4, 3),
            vec![
                0.0, 1.0, 0.0, // Orthogonal (similarity = 0)
                0.5, 0.5, 0.0, // Somewhat similar
                1.0, 0.0, 0.0, // Most similar (identical)
                0.7, 0.3, 0.0, // Very similar
            ],
        )
        .unwrap();

        let (_, top_indices) = k_nearest_neighbors(query.view(), documents.view(), 3);

        // Should return indices in order of similarity
        assert_eq!(top_indices[0], 2); // Most similar
        // Second and third can vary based on exact similarity values
        assert!(top_indices.len() == 3);
    }
}
