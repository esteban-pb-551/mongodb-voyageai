//! Asymmetric retrieval: different models for documents vs queries.
//!
//! This example demonstrates Voyage AI's asymmetric retrieval capability,
//! where you use a larger, more accurate model for document embeddings
//! (computed once) and a smaller, faster model for query embeddings
//! (computed on every request).
//!
//! # Why Asymmetric?
//!
//! - Documents are embedded once and stored → Use expensive, high-quality model
//! - Queries are embedded on every request → Use fast, cheap model
//! - Voyage 4 series has shared embedding spaces, making this possible!
//!
//! # Cost Optimization
//!
//! Example for 1M documents, 10K queries/day:
//!
//! **Symmetric (voyage-4-large for both):**
//! - Index: 1M × $0.12/1M tokens = $120 (one-time)
//! - Queries: 10K × 365 × $0.12/1M = $438/year
//! - Total first year: $558
//!
//! **Asymmetric (voyage-4-large docs, voyage-4-nano queries):**
//! - Index: 1M × $0.12/1M tokens = $120 (one-time)
//! - Queries: 10K × 365 × $0.02/1M = $73/year
//! - Total first year: $193 (65% savings!)

use mongodb_voyageai::{Client, model, OutputDtype};

/// Document with embedding from the large model.
struct Document {
    title: String,
    embedding: Vec<f64>,
}

/// Computes cosine similarity between two vectors.
fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Reads VOYAGEAI_API_KEY from the environment
    let client = Client::from_env();

    println!("=== Asymmetric Retrieval with Voyage 4 Series ===\n");

    // ── Step 1: Index documents with voyage-4-large ──────────────────────────
    println!("📚 Step 1: Indexing documents with voyage-4-large (high quality)");
    println!("   This happens once, so we can afford the expensive model.\n");

    let knowledge_base = [
        (
            "Quantum Computing Basics",
            "Quantum computers use qubits that can exist in superposition, \
             enabling parallel computation. They excel at optimization problems \
             and cryptography but require extremely low temperatures to operate.",
        ),
        (
            "Blockchain Technology",
            "Blockchain is a distributed ledger technology that ensures data \
             integrity through cryptographic hashing and consensus mechanisms. \
             It powers cryptocurrencies and enables trustless transactions.",
        ),
        (
            "Edge Computing",
            "Edge computing processes data near its source rather than in \
             centralized data centers. This reduces latency and bandwidth usage, \
             making it ideal for IoT devices and real-time applications.",
        ),
        (
            "Serverless Architecture",
            "Serverless computing allows developers to run code without managing \
             servers. Cloud providers automatically scale resources and charge \
             only for actual execution time, reducing operational overhead.",
        ),
        (
            "GraphQL APIs",
            "GraphQL is a query language for APIs that lets clients request \
             exactly the data they need. Unlike REST, it reduces over-fetching \
             and enables efficient data loading with a single request.",
        ),
    ];

    let doc_contents: Vec<&str> = knowledge_base.iter().map(|(_, c)| *c).collect();

    // Use voyage-4-large for maximum quality (one-time cost)
    let doc_embeddings = client
        .embed(&doc_contents)
        .model(model::VOYAGE_4_LARGE)
        .input_type("document")
        .output_dimension(512)
        .output_dtype(OutputDtype::Int8)
        .send()
        .await?;

    println!("   ✓ Model: {} (MoE architecture)", doc_embeddings.model);
    println!("   ✓ Documents: {}", doc_embeddings.embeddings.len());
    println!("   ✓ Dimensions: {}", doc_embeddings.embedding(0).unwrap().len());
    println!("   ✓ Tokens: {}", doc_embeddings.usage.total_tokens);
    println!("   ✓ Cost: ~${:.4} (one-time)\n", 
             doc_embeddings.usage.total_tokens as f64 * 0.12 / 1_000_000.0);

    // Build document store
    let documents: Vec<Document> = knowledge_base
        .iter()
        .zip(&doc_embeddings.embeddings)
        .map(|((title, _content), emb)| Document {
            title: title.to_string(),
            embedding: emb.clone(),
        })
        .collect();

    // ── Step 2: Process queries with voyage-4-lite ───────────────────────────
    println!("🔍 Step 2: Processing queries with voyage-4-lite (fast & cheap)");
    println!("   This happens on every request, so we use the efficient model.\n");

    let queries = [
        "How do quantum computers work?",
        "What is distributed ledger technology?",
        "Explain serverless computing benefits",
    ];

    let mut total_query_tokens = 0;

    for (i, query) in queries.iter().enumerate() {
        println!("Query {}: \"{}\"", i + 1, query);

        // Use voyage-4-lite for speed (per-request cost)
        let query_embedding = client
            .embed(*query)
            .model(model::VOYAGE_4_LITE)
            .input_type("query")
            .output_dimension(512)
            .output_dtype(OutputDtype::Int8)
            .send()
            .await?;

        total_query_tokens += query_embedding.usage.total_tokens;

        let query_vec = query_embedding.embedding(0).unwrap();

        // Find top 3 matches
        let mut scored: Vec<(&Document, f64)> = documents
            .iter()
            .map(|doc| {
                let sim = cosine_similarity(query_vec, &doc.embedding);
                (doc, sim)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        println!("   Top 3 results:");
        for (rank, (doc, score)) in scored.iter().take(3).enumerate() {
            println!("      {}. [Score: {:.4}] {}", rank + 1, score, doc.title);
        }
        println!();
    }

    // ── Step 3: Cost analysis ────────────────────────────────────────────────
    println!("💰 Cost Analysis:");
    println!();
    println!("Document Indexing (one-time):");
    println!("   • Model: voyage-4-large");
    println!("   • Tokens: {}", doc_embeddings.usage.total_tokens);
    println!("   • Cost: ~${:.4}", 
             doc_embeddings.usage.total_tokens as f64 * 0.12 / 1_000_000.0);
    println!();
    println!("Query Processing (per request):");
    println!("   • Model: voyage-4-lite");
    println!("   • Tokens: {} (for {} queries)", total_query_tokens, queries.len());
    println!("   • Cost per query: ~${:.6}", 
             (total_query_tokens as f64 / queries.len() as f64) * 0.02 / 1_000_000.0);
    println!();
    println!("Projected Annual Cost (10K queries/day):");
    println!("   • Indexing: $120 (one-time for 1M docs)");
    println!("   • Queries: ${:.2}/year", 
             10_000.0 * 365.0 * 0.02 / 1_000_000.0 * 
             (total_query_tokens as f64 / queries.len() as f64));
    println!();
    println!("Comparison with Symmetric Approach:");
    println!("   • Asymmetric (4-large + 4-lite): ~$193 first year");
    println!("   • Symmetric (4-large only):      ~$558 first year");
    println!("   • Savings: 65% 💰");
    println!();
    println!("Key Insight:");
    println!("   The Voyage 4 series uses shared embedding spaces, so vectors");
    println!("   from different models (4-large, 4-lite, 4-nano) are compatible!");
    println!("   This enables asymmetric retrieval for massive cost savings.");

    Ok(())
}
