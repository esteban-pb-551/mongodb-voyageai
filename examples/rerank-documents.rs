//! Reranking example.
//!
//! Shows the three most common reranking patterns:
//!
//! 1. **Full ranking** — score all documents and return them in order.
//! 2. **Top-K** — return only the N highest-scoring documents.
//! 3. **Alternate model** — swap in a lighter model to trade accuracy for speed.
//!
//! # What is reranking?
//!
//! A reranker is a *cross-encoder*: it scores each (query, document) pair
//! jointly rather than comparing independent vectors. This is slower than
//! embedding similarity but significantly more accurate, making it ideal as a
//! second stage after a fast nearest-neighbour retrieval step.
//!

use mongodb_voyageai::{Client, model};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Reads VOYAGEAI_API_KEY from the environment
    let client = Client::try_from_env().unwrap_or_else(|e| {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    });

    let query = "Who should I call if my pipes are leaking?";

    // The corpus is defined once as a `Vec<&str>`.
    // All three rerank calls below pass `&documents` as a slice, so this
    // binding stays alive and can be used to look up the original text via
    // `result.index` after each call — no `.clone()` needed.
    let documents = vec![
        "John is a musician who plays guitar in a local band.",
        "Paul is a licensed plumber with 15 years of experience.",
        "George is a high school math teacher.",
        "Ringo is a doctor specializing in cardiology.",
        "Lisa is a corporate lawyer at a downtown firm.",
        "Stuart is a painter who exhibits in galleries.",
    ];

    // ── Example 1: full ranking ───────────────────────────────────────────────
    // Without `.top_k()` the API returns all documents sorted by relevance.
    // The default model is `model::RERANK`; omitting `.model()` is equivalent
    // to calling `.model(model::RERANK)` explicitly.
    println!("=== Full ranking ===");
    let rerank = client.rerank(query, &documents).send().await?;

    println!("Query:       {:?}", query);
    println!("Model:       {}", rerank.model);
    println!("Tokens used: {}", rerank.usage.total_tokens);
    println!();
    for result in &rerank.results {
        // `result.index` is the position of the document in the original slice,
        // so we can recover the text without storing it a second time.
        println!(
            "  [{}] score={:.4}  {:?}",
            result.index, result.relevance_score, documents[result.index]
        );
    }

    // ── Example 2: top-K filter ───────────────────────────────────────────────
    // `.top_k(n)` asks the API to return only the n highest-scoring results.
    // Useful when you only care about the best matches and want to save on
    // response size and downstream processing.
    println!("\n=== Top 2 only ===");
    let top2 = client
        .rerank(query, &documents)
        .model(model::RERANK)
        .top_k(2)
        .send()
        .await?;

    for result in &top2.results {
        println!(
            "  [{}] score={:.4}  {:?}",
            result.index, result.relevance_score, documents[result.index]
        );
    }

    // ── Example 3: lite model ─────────────────────────────────────────────────
    // `model::RERANK_2_5_LITE` is a smaller, faster variant that trades a small
    // amount of accuracy for lower latency and cost. A good choice when reranking
    // large candidate sets or in latency-sensitive pipelines.
    println!("\n=== Using rerank-2.5-lite ===");
    let lite = client
        .rerank(query, &documents)
        .model(model::RERANK_2_5_LITE)
        .top_k(3)
        .send()
        .await?;

    println!("Model: {}", lite.model);
    for result in &lite.results {
        println!(
            "  [{}] score={:.4}  {:?}",
            result.index, result.relevance_score, documents[result.index]
        );
    }

    Ok(())
}