//! Example: Text normalization and chunking for RAG pipelines
//!
//! Run with: cargo run --example chunk-normalize

use mongodb_voyageai::{
    pairwise::k_nearest_neighbors,
    chunk::{
        chunking::{chunk_recursive, ChunkConfig},
        normalizer::{normalize, NormalizerConfig},
    },
    Client,
    model,
};
use ndarray::{Array1, Array2};
use std::fs::read_to_string;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::from_env();

    // ── Long document example ──────────────────────────────────────────
    let long_document = read_to_string("examples/sample-doc.txt")
        .expect("Could not open sample-doc.txt — make sure it exists in the project root");

    println!("=== Text Normalization and Chunking Example ===\n");
    println!("Original document length: {} chars\n", long_document.len());

    // Recursive chunking (best strategy for RAG)
    let config = ChunkConfig {
        chunk_size: 500,   // ~150 tokens — adjusted to model
        chunk_overlap: 20,
    };

    // Normalize text: remove extra whitespace, fix line breaks
    let clean = normalize(&long_document, &NormalizerConfig::prose());
    println!("Normalized text length: {} chars\n", clean.len());

    // Split into chunks respecting paragraph boundaries
    let chunks = chunk_recursive(&clean, &config);
    println!("Number of chunks created: {}\n", chunks.len());

    // Display each chunk
    // for (i, chunk) in chunks.iter().enumerate() {
    //     println!("--- Chunk {} ({} chars) ---", i + 1, chunk.len());
    //     println!("{}\n", chunk);
    // }

    // ── Generate embeddings for each chunk ────────────────────────────
    println!("=== Generating Embeddings ===\n");

    let embed_response = client
        .embed(&chunks)
        .model(model::VOYAGE_4_LARGE)
        .input_type("document")
        .output_dimension(512)
        .send()
        .await?;

    println!("Model used: {}", embed_response.model);
    println!("Total embeddings: {}", embed_response.embeddings.len());
    println!(
        "Embedding dimensions: {}",
        embed_response.embedding(0).unwrap().len()
    );
    println!("Total tokens used: {}", embed_response.usage.total_tokens);

    let n_docs = embed_response.embeddings.len();
    let n_features = embed_response.embeddings[0].len();

    // Flatten embeddings into a 2D ndarray matrix (n_docs × n_features),
    // casting to f64 for ndarray compatibility.
    let flat: Vec<f64> = embed_response
        .embeddings
        .iter()
        .flatten()
        .map(|&v| v as f64)
        .collect();
    let documents_embeddings = Array2::from_shape_vec((n_docs, n_features), flat)?;

    // ── Asymmetric retrieval: query with voyage-4-lite ─────────────────
    // let query_text = "What foods are part of a healthy diet?";
    let query_text = "Why is Pakistan's relationship with Saudi Arabia described as an 'awkward position' regarding the conflict with Iran?";

    let query_embeds = client
        .embed(vec![query_text])
        .model(model::VOYAGE_4_LITE)
        .input_type("query")
        .output_dimension(512)  // must match document embedding dimension
        .send()
        .await?;

    let query_embedding: Array1<f64> = Array1::from_vec(
        query_embeds.embeddings[0]
            .iter()
            .map(|&v| v as f64)
            .collect(),
    );

    println!("Query: \"{}\"\n", query_text);

    // KNN — retrieve all chunks as initial candidates for reranking
    let k_final = 3;
    println!("KNN — retrieving top {} candidates...\n", chunks.len());

    let (_, top_indices) = k_nearest_neighbors(
        query_embedding.view(),
        documents_embeddings.view(),
        chunks.len(),
    );

    // Map KNN indices back to text slices for the reranker
    let candidate_texts: Vec<&str> = top_indices
        .iter()
        .map(|&idx| chunks[idx].as_str())
        .collect();

    // Rerank with cross-encoder — refine down to top k_final results
    println!("Rerank — refining to top {}...\n", k_final);

    let rerank = client
        .rerank(query_text, candidate_texts)
        .model("rerank-2.5")
        .top_k(k_final)
        .send()
        .await?;

    // Display final ranked results
    for (rank, result) in rerank.results.iter().enumerate() {
        let original_idx = top_indices[result.index];
        println!(
            "Rank {} (score: {:.4}):\n{}\n",
            rank + 1,
            result.relevance_score,
            chunks[original_idx]
        );
    }

    Ok(())
}