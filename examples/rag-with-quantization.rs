//! RAG pipeline with quantized embeddings for cost-efficient vector storage.
//!
//! This example demonstrates a production-ready RAG (Retrieval-Augmented Generation)
//! pipeline that uses int8 quantization to reduce storage costs by 4× while
//! maintaining high retrieval quality.
//!
//! # Pipeline Architecture
//!
//! 1. **Document Indexing** (one-time):
//!    - Embed documents with `output_dtype(Int8)` and `output_dimension(512)`
//!    - Store quantized vectors in a vector database
//!    - Storage: 512 bytes per document (vs 2048 bytes for float)
//!
//! 2. **Query Processing** (per request):
//!    - Embed query with same settings (int8, 512 dims)
//!    - Search vector database for top-K candidates
//!    - Rerank candidates with cross-encoder
//!    - Return best results
//!
//! # Cost Savings
//!
//! For 1 million documents:
//! - Float (2048 dims): ~2 GB storage
//! - Int8 (512 dims): ~512 MB storage (4× reduction)
//! - Quality: Minimal degradation (<2% on most benchmarks)

use mongodb_voyageai::{Client, Config, model, OutputDtype};
use std::collections::HashMap;

/// A document in our knowledge base with its quantized embedding.
#[derive(Clone)]
struct Document {
    id: usize,
    title: String,
    content: String,
    embedding: Vec<f64>, // In production, this would be Vec<i8>
}

/// Simple in-memory vector store (in production, use MongoDB Atlas, Pinecone, etc.)
struct VectorStore {
    documents: Vec<Document>,
}

impl VectorStore {
    fn new() -> Self {
        Self {
            documents: Vec::new(),
        }
    }

    fn add(&mut self, doc: Document) {
        self.documents.push(doc);
    }

    /// Find top-K nearest neighbors using cosine similarity.
    fn search(&self, query_embedding: &[f64], top_k: usize) -> Vec<&Document> {
        let mut scored: Vec<(&Document, f64)> = self
            .documents
            .iter()
            .map(|doc| {
                let similarity = cosine_similarity(query_embedding, &doc.embedding);
                (doc, similarity)
            })
            .collect();

        // Sort by similarity (descending)
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        scored.into_iter().take(top_k).map(|(doc, _)| doc).collect()
    }
}

/// Computes cosine similarity between two vectors.
fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    dot / (norm_a * norm_b)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::new(&Config::new())?;

    println!("=== RAG Pipeline with Quantized Embeddings ===\n");

    // ── Step 1: Build the knowledge base ─────────────────────────────────────
    println!("📚 Step 1: Indexing documents with int8 quantization...");

    let knowledge_base = vec![
        (
            "Rust Memory Safety",
            "Rust guarantees memory safety through its ownership system. \
             The compiler enforces rules at compile time, preventing data races \
             and null pointer dereferences without needing a garbage collector.",
        ),
        (
            "Python Data Science",
            "Python is the leading language for data science and machine learning. \
             Libraries like NumPy, Pandas, and scikit-learn provide powerful tools \
             for data analysis, while TensorFlow and PyTorch enable deep learning.",
        ),
        (
            "JavaScript Async Programming",
            "JavaScript uses promises and async/await for asynchronous programming. \
             This allows non-blocking I/O operations, making it ideal for web servers \
             and browser applications that need to handle multiple concurrent tasks.",
        ),
        (
            "Database Indexing",
            "Database indexes improve query performance by creating data structures \
             that allow fast lookups. B-tree indexes are common for range queries, \
             while hash indexes excel at equality comparisons.",
        ),
        (
            "Machine Learning Embeddings",
            "Embeddings are dense vector representations of data that capture semantic \
             meaning. They enable similarity search, clustering, and are fundamental \
             to modern NLP models like transformers.",
        ),
        (
            "Cloud Computing Scalability",
            "Cloud platforms provide elastic scalability through auto-scaling groups \
             and load balancers. Applications can automatically adjust resources based \
             on demand, optimizing both performance and cost.",
        ),
        (
            "API Rate Limiting",
            "Rate limiting protects APIs from abuse by restricting the number of requests \
             per time window. Common strategies include token bucket, leaky bucket, and \
             fixed window algorithms.",
        ),
        (
            "Microservices Architecture",
            "Microservices decompose applications into small, independent services that \
             communicate via APIs. This enables teams to develop, deploy, and scale \
             services independently.",
        ),
    ];

    // Collect all content for batch embedding
    let contents: Vec<&str> = knowledge_base.iter().map(|(_, content)| *content).collect();

    // Embed all documents with int8 quantization and 512 dimensions
    let embed_result = client
        .embed(&contents)
        .model(model::VOYAGE_3_LARGE)
        .input_type("document")
        .output_dimension(512)
        .output_dtype(OutputDtype::Int8) // ← 4× storage reduction
        .send()
        .await?;

    println!("   ✓ Embedded {} documents", embed_result.embeddings.len());
    println!("   ✓ Model: {}", embed_result.model);
    println!("   ✓ Dimensions: {}", embed_result.embedding(0).unwrap().len());
    println!("   ✓ Tokens used: {}", embed_result.usage.total_tokens);
    println!(
        "   ✓ Storage per doc: {} bytes (vs 2048 for float)\n",
        512
    );

    // Build the vector store
    let mut store = VectorStore::new();
    for (i, ((title, content), embedding)) in knowledge_base
        .iter()
        .zip(&embed_result.embeddings)
        .enumerate()
    {
        store.add(Document {
            id: i,
            title: title.to_string(),
            content: content.to_string(),
            embedding: embedding.clone(),
        });
    }

    // ── Step 2: Process queries ──────────────────────────────────────────────
    let queries = vec![
        "How does Rust prevent memory bugs?",
        "What are the best Python libraries for ML?",
        "Explain vector embeddings in AI",
    ];

    for (i, query) in queries.iter().enumerate() {
        println!("🔍 Query {}: \"{}\"", i + 1, query);

        // Embed the query with the same settings as documents
        let query_embed = client
            .embed(query)
            .model(model::VOYAGE_3_LARGE)
            .input_type("query")
            .output_dimension(512)
            .output_dtype(OutputDtype::Int8) // ← Must match document settings
            .send()
            .await?;

        let query_vector = query_embed.embedding(0).unwrap();

        // Search for top 3 candidates
        let candidates = store.search(query_vector, 3);

        println!("   📊 Top 3 results (by cosine similarity):");
        for (rank, doc) in candidates.iter().enumerate() {
            let similarity = cosine_similarity(query_vector, &doc.embedding);
            println!(
                "      {}. [Score: {:.4}] {}",
                rank + 1,
                similarity,
                doc.title
            );
        }

        // Rerank with cross-encoder for higher precision
        let candidate_contents: Vec<&str> =
            candidates.iter().map(|d| d.content.as_str()).collect();

        let rerank_result = client
            .rerank(query, &candidate_contents)
            .model(model::RERANK_2_5)
            .top_k(1)
            .send()
            .await?;

        println!("   🎯 Best match after reranking:");
        for result in &rerank_result.results {
            let best_doc = candidates[result.index];
            println!(
                "      ✓ [Score: {:.4}] {}",
                result.relevance_score, best_doc.title
            );
            println!("        Preview: {}...", &best_doc.content[..100]);
        }
        println!();
    }

    // ── Step 3: Show storage savings ─────────────────────────────────────────
    println!("💾 Storage Analysis:");
    println!("   Documents indexed: {}", store.documents.len());
    println!("   Dimensions per vector: 512");
    println!("   Quantization: int8");
    println!();
    println!("   Storage comparison:");
    println!("   • Float (2048 dims):  {} bytes total", 8 * 2048);
    println!("   • Int8 (512 dims):    {} bytes total", 8 * 512);
    println!("   • Savings: 4× reduction");
    println!();
    println!("   For 1 million documents:");
    println!("   • Float: ~2.0 GB");
    println!("   • Int8:  ~512 MB");
    println!("   • Cost savings: ~75%");

    Ok(())
}
