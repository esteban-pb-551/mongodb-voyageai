//! Semantic search with embedding + rerank pipeline.
//!
//! This example demonstrates a two-stage retrieval approach:
//!
//! 1. **Embed** — convert all documents into vectors once and store them.
//! 2. **Search** — for each query, embed it and find the nearest neighbours
//!    using Euclidean distance (fast, runs locally).
//! 3. **Rerank** — send the top-K candidates to the reranker, which scores
//!    them with a cross-encoder model for higher precision.
//!
//! # Why two stages?
//!
//! Embedding-based nearest-neighbour search is fast but approximate.
//! The reranker is slower yet more accurate, so running it only on the
//! top-K candidates from stage 1 gives the best of both worlds.

use mongodb_voyageai::{Client, Config, model};

/// A document paired with its pre-computed embedding vector.
struct Entry {
    document: String,
    embedding: Vec<f64>,
}

/// Computes the Euclidean (L2) distance between two vectors.
///
/// Lower values mean the vectors are closer in the embedding space,
/// i.e. the texts are semantically more similar.
fn euclidean_distance(src: &[f64], dst: &[f64]) -> f64 {
    src.iter()
        .zip(dst)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt()
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::new(&Config::new())?;

    // ── Stage 0: build the document store ────────────────────────────────────
    // In a real application these would come from a database; here we use a
    // small in-memory corpus to keep the example self-contained.
    let documents = vec![
        "John is a musician.",
        "Paul is a plumber.",
        "George is a teacher.",
        "Ringo is a doctor.",
        "Lisa is a lawyer.",
        "Stuart is a painter.",
        "Brian is a writer.",
        "Jane is a chef.",
        "Bill is a nurse.",
        "Susan is a carpenter.",
    ];

    // ── Stage 1: embed all documents once ────────────────────────────────────
    // We pass `&documents` as a slice so the original vec stays accessible
    // for building the `Entry` list below. The `input_type("document")` hint
    // tells the model these are passages to be retrieved, not queries.
    let embed_result = client
        .embed(&documents)
        .model(model::VOYAGE_4)
        .input_type("document")
        .send()
        .await?;

    // Pair each document string with its embedding vector.
    let entries: Vec<Entry> = documents
        .iter()
        .zip(&embed_result.embeddings)
        .map(|(doc, emb)| Entry {
            document: doc.to_string(),
            embedding: emb.clone(),
        })
        .collect();

    // Run two example queries against the store.
    search(&client, &entries, "What do George and Ringo do?").await?;
    search(&client, &entries, "Who works in the medical field?").await?;

    Ok(())
}

/// Runs a semantic search query against the pre-embedded document store.
///
/// # Pipeline
///
/// 1. Embed the query with `input_type("query")`.
/// 2. Rank all entries by Euclidean distance and keep the top 4.
/// 3. Rerank those 4 candidates with the cross-encoder and print the best.
///
/// # Arguments
///
/// * `client`  — Shared VoyageAI client.
/// * `entries` — Pre-embedded document store.
/// * `query`   — Natural-language search query.
async fn search(
    client: &Client,
    entries: &[Entry],
    query: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // ── Step 1: embed the query ───────────────────────────────────────────────
    // Using `input_type("query")` is important for asymmetric search: the model
    // applies a different transformation to queries than to documents, which
    // improves retrieval quality.
    let query_embed = client
        .embed(query)
        .model(model::VOYAGE_4)
        .input_type("query")
        .send()
        .await?;
    let query_embedding = query_embed.embedding(0).expect("no embedding returned");

    // ── Step 2: nearest-neighbour retrieval ──────────────────────────────────
    // Sort all entries by Euclidean distance to the query vector and take the
    // top 4. This is a brute-force O(n) scan; for large corpora use an ANN
    // index (e.g. MongoDB Atlas Vector Search).
    let mut sorted: Vec<&Entry> = entries.iter().collect();
    sorted.sort_by(|a, b| {
        euclidean_distance(&a.embedding, query_embedding)
            .partial_cmp(&euclidean_distance(&b.embedding, query_embedding))
            .unwrap()
    });

    // Collect the top-4 candidates as string slices.
    // Using `&str` avoids cloning the document strings just to pass them to
    // the reranker; `nearest` stays accessible for indexing after the call.
    let nearest: Vec<&str> = sorted.iter().take(4).map(|e| e.document.as_str()).collect();

    // ── Step 3: rerank the candidates ────────────────────────────────────────
    // The cross-encoder reranker scores each (query, document) pair jointly,
    // which is more accurate than embedding similarity alone. We pass `&nearest`
    // as a slice so `nearest` is still available for indexing the result below.
    let rerank_result = client
        .rerank(query, &nearest)
        .model(model::RERANK)
        .top_k(1)
        .send()
        .await?;

    // Print the best result. `result.index` refers to the position within
    // `nearest`, not the original `entries` slice.
    println!("query={query:?}");
    for result in &rerank_result.results {
        println!(
            "document={:?} relevance_score={}",
            nearest[result.index], result.relevance_score
        );
    }

    Ok(())
}