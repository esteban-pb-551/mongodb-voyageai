//! Compare embedding quality across different quantization types.
//!
//! This example embeds the same text with different quantization settings
//! and compares the results to help you choose the right trade-off between
//! storage cost and quality for your use case.
//!
//! # Key Findings
//!
//! - Int8/Uint8: Minimal quality loss, 4× storage reduction → Best for production
//! - Binary/Ubinary: Significant compression, 32× reduction → Use for massive scale
//! - Float: Maximum precision but 4-32× more expensive storage

use mongodb_voyageai::{Client, Config, model, OutputDtype};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::new(&Config::new())?;

    // Sample texts for embedding
    let texts = vec![
        "Artificial intelligence is transforming software development.",
        "Machine learning models require large amounts of training data.",
        "Neural networks can learn complex patterns from examples.",
    ];

    println!("=== Quantization Quality Comparison ===\n");
    println!("Embedding {} texts with different quantization types...\n", texts.len());

    // ── 1. Float (baseline) ──────────────────────────────────────────────────
    println!("1️⃣  FLOAT (Baseline - Full Precision)");
    let float_result = client
        .embed(&texts)
        .model(model::VOYAGE_4_LARGE)
        .input_type("document")
        .output_dimension(512)
        // No output_dtype = defaults to Float
        .send()
        .await?;

    print_embedding_info("Float", &float_result, 4);

    // ── 2. Int8 ──────────────────────────────────────────────────────────────
    println!("\n2️⃣  INT8 (Signed 8-bit Integer)");
    let int8_result = client
        .embed(&texts)
        .model(model::VOYAGE_4_LARGE)
        .input_type("document")
        .output_dimension(512)
        .output_dtype(OutputDtype::Int8)
        .send()
        .await?;

    print_embedding_info("Int8", &int8_result, 1);
    compare_embeddings("Float", &float_result, "Int8", &int8_result);

    // ── 3. Uint8 ─────────────────────────────────────────────────────────────
    println!("\n3️⃣  UINT8 (Unsigned 8-bit Integer)");
    let uint8_result = client
        .embed(&texts)
        .model(model::VOYAGE_4_LARGE)
        .input_type("document")
        .output_dimension(512)
        .output_dtype(OutputDtype::Uint8)
        .send()
        .await?;

    print_embedding_info("Uint8", &uint8_result, 1);
    compare_embeddings("Float", &float_result, "Uint8", &uint8_result);

    // ── 4. Binary ────────────────────────────────────────────────────────────
    println!("\n4️⃣  BINARY (1-bit Signed)");
    let binary_result = client
        .embed(&texts)
        .model(model::VOYAGE_4_LARGE)
        .input_type("document")
        .output_dimension(512)
        .output_dtype(OutputDtype::Binary)
        .send()
        .await?;

    print_embedding_info("Binary", &binary_result, 1);
    compare_embeddings("Float", &float_result, "Binary", &binary_result);

    // ── 5. Ubinary ───────────────────────────────────────────────────────────
    println!("\n5️⃣  UBINARY (1-bit Unsigned)");
    let ubinary_result = client
        .embed(&texts)
        .model(model::VOYAGE_4_LARGE)
        .input_type("document")
        .output_dimension(512)
        .output_dtype(OutputDtype::Ubinary)
        .send()
        .await?;

    print_embedding_info("Ubinary", &ubinary_result, 1);
    compare_embeddings("Float", &float_result, "Ubinary", &ubinary_result);

    // ── Summary ──────────────────────────────────────────────────────────────
    println!("\n{}", "=".repeat(60));
    println!("📊 SUMMARY & RECOMMENDATIONS");
    println!("{}", "=".repeat(60));
    println!();
    println!("Storage for 1M vectors (512 dims):");
    println!("  • Float:   1.95 GB  (baseline)");
    println!("  • Int8:    488 MB   (4× smaller)   ← RECOMMENDED for production");
    println!("  • Uint8:   488 MB   (4× smaller)   ← RECOMMENDED for production");
    println!("  • Binary:  61 MB    (32× smaller)  ← Use for massive scale");
    println!("  • Ubinary: 61 MB    (32× smaller)  ← Use for massive scale");
    println!();
    println!("Quality vs Storage Trade-off:");
    println!("  ✓ Int8/Uint8:  Minimal quality loss (<2%), huge savings");
    println!("  ⚠ Binary:      Noticeable quality loss, maximum compression");
    println!();
    println!("Best Practices:");
    println!("  1. Start with Int8 for most production use cases");
    println!("  2. Use Binary only if storage cost is critical");
    println!("  3. Always use output_dimension(512) with quantization");
    println!("  4. Test on your specific data before deploying");

    Ok(())
}

/// Prints embedding information for a given result.
fn print_embedding_info(
    _dtype_name: &str,
    result: &mongodb_voyageai::Embed,
    bytes_per_element: usize,
) {
    let dims = result.embedding(0).unwrap().len();
    let storage_per_vector = dims * bytes_per_element;
    let num_vectors = result.embeddings.len();

    println!("   Model:      {}", result.model);
    println!("   Dimensions: {}", dims);
    println!("   Vectors:    {}", num_vectors);
    println!("   Storage:    {} bytes/vector", storage_per_vector);
    println!("   Total:      {} bytes for {} vectors", 
             storage_per_vector * num_vectors, num_vectors);
    println!("   Tokens:     {}", result.usage.total_tokens);

    // Show sample values from first embedding
    let first_embedding = result.embedding(0).unwrap();
    println!("   Sample:     [{:.4}, {:.4}, {:.4}, ...]", 
             first_embedding[0], first_embedding[1], first_embedding[2]);
}

/// Compares two embedding results by computing cosine similarity.
fn compare_embeddings(
    name1: &str,
    result1: &mongodb_voyageai::Embed,
    _name2: &str,
    result2: &mongodb_voyageai::Embed,
) {
    let emb1 = result1.embedding(0).unwrap();
    let emb2 = result2.embedding(0).unwrap();

    let similarity = cosine_similarity(emb1, emb2);

    println!("   Similarity: {:.4} (vs {})", similarity, name1);
    
    if similarity > 0.99 {
        println!("   Quality:    ✓ Excellent (>99% preserved)");
    } else if similarity > 0.95 {
        println!("   Quality:    ✓ Very Good (>95% preserved)");
    } else if similarity > 0.90 {
        println!("   Quality:    ⚠ Good (>90% preserved)");
    } else {
        println!("   Quality:    ⚠ Degraded (<90% preserved)");
    }
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
