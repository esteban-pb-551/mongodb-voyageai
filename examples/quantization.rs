//! Embedding quantization for storage optimization.
//!
//! This example demonstrates how to use Voyage AI's Quantization-Aware Training
//! to dramatically reduce storage costs while maintaining quality.
//!
//! # Storage Comparison
//!
//! For a 512-dimensional embedding:
//!
//! | Type     | Bytes | Compression | Use Case                          |
//! |----------|-------|-------------|-----------------------------------|
//! | Float    | 2048  | 1×          | Maximum precision required        |
//! | Int8     | 512   | 4×          | Production (minimal quality loss) |
//! | Uint8    | 512   | 4×          | Production (minimal quality loss) |
//! | Binary   | 64    | 32×         | Large-scale (storage critical)    |
//! | Ubinary  | 64    | 32×         | Large-scale (storage critical)    |
//!
//! # Model Support
//!
//! Only models with Quantization-Aware Training support non-float types:
//! - voyage-3-large
//! - voyage-4, voyage-4-lite, voyage-4-large
//!
//! # Performance
//!
//! According to Voyage AI benchmarks, voyage-3-large with int8 at 512 dimensions
//! outperforms OpenAI-v3-large by 8.56% while using only 1/24 the storage.

use mongodb_voyageai::{Client, Config, model, OutputDtype};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::new(&Config::new())?;

    let texts = vec![
        "Rust is a systems programming language.",
        "Python is great for data science.",
        "JavaScript runs in the browser.",
    ];

    println!("=== Embedding Quantization Comparison ===\n");

    // ── Float (default) ──────────────────────────────────────────────────────
    println!("1. Float (full precision)");
    let float_embed = client
        .embed(&texts)
        .model(model::VOYAGE_3_LARGE)
        .output_dimension(512)
        .send()
        .await?;

    let float_size = calculate_storage_bytes(&float_embed, 4); // 4 bytes per f32
    println!("   Model: {}", float_embed.model);
    println!("   Tokens: {}", float_embed.usage.total_tokens);
    println!("   Dimensions: {}", float_embed.embedding(0).unwrap().len());
    println!("   Storage: {} bytes per vector", float_size);
    println!("   Total: {} bytes for {} vectors\n", float_size * texts.len(), texts.len());

    // ── Int8 (4× compression) ────────────────────────────────────────────────
    println!("2. Int8 (4× compression)");
    let int8_embed = client
        .embed(&texts)
        .model(model::VOYAGE_3_LARGE)
        .output_dimension(512)
        .output_dtype(OutputDtype::Int8)
        .send()
        .await?;

    let int8_size = calculate_storage_bytes(&int8_embed, 1); // 1 byte per int8
    println!("   Model: {}", int8_embed.model);
    println!("   Tokens: {}", int8_embed.usage.total_tokens);
    println!("   Dimensions: {}", int8_embed.embedding(0).unwrap().len());
    println!("   Storage: {} bytes per vector", int8_size);
    println!("   Total: {} bytes for {} vectors", int8_size * texts.len(), texts.len());
    println!("   Compression: {:.1}× vs float\n", float_size as f64 / int8_size as f64);

    // ── Uint8 (4× compression) ───────────────────────────────────────────────
    println!("3. Uint8 (4× compression)");
    let uint8_embed = client
        .embed(&texts)
        .model(model::VOYAGE_3_LARGE)
        .output_dimension(512)
        .output_dtype(OutputDtype::Uint8)
        .send()
        .await?;

    let uint8_size = calculate_storage_bytes(&uint8_embed, 1); // 1 byte per uint8
    println!("   Model: {}", uint8_embed.model);
    println!("   Tokens: {}", uint8_embed.usage.total_tokens);
    println!("   Dimensions: {}", uint8_embed.embedding(0).unwrap().len());
    println!("   Storage: {} bytes per vector", uint8_size);
    println!("   Total: {} bytes for {} vectors", uint8_size * texts.len(), texts.len());
    println!("   Compression: {:.1}× vs float\n", float_size as f64 / uint8_size as f64);

    // ── Binary (32× compression) ─────────────────────────────────────────────
    println!("4. Binary (32× compression)");
    let binary_embed = client
        .embed(&texts)
        .model(model::VOYAGE_3_LARGE)
        .output_dimension(512)
        .output_dtype(OutputDtype::Binary)
        .send()
        .await?;

    let binary_size = calculate_storage_bytes(&binary_embed, 1) / 8; // 1 bit per dimension
    println!("   Model: {}", binary_embed.model);
    println!("   Tokens: {}", binary_embed.usage.total_tokens);
    println!("   Dimensions: {}", binary_embed.embedding(0).unwrap().len());
    println!("   Storage: {} bytes per vector", binary_size);
    println!("   Total: {} bytes for {} vectors", binary_size * texts.len(), texts.len());
    println!("   Compression: {:.1}× vs float\n", float_size as f64 / binary_size as f64);

    // ── Ubinary (32× compression) ────────────────────────────────────────────
    println!("5. Ubinary (32× compression)");
    let ubinary_embed = client
        .embed(&texts)
        .model(model::VOYAGE_3_LARGE)
        .output_dimension(512)
        .output_dtype(OutputDtype::Ubinary)
        .send()
        .await?;

    let ubinary_size = calculate_storage_bytes(&ubinary_embed, 1) / 8; // 1 bit per dimension
    println!("   Model: {}", ubinary_embed.model);
    println!("   Tokens: {}", ubinary_embed.usage.total_tokens);
    println!("   Dimensions: {}", ubinary_embed.embedding(0).unwrap().len());
    println!("   Storage: {} bytes per vector", ubinary_size);
    println!("   Total: {} bytes for {} vectors", ubinary_size * texts.len(), texts.len());
    println!("   Compression: {:.1}× vs float\n", float_size as f64 / ubinary_size as f64);

    // ── Summary ──────────────────────────────────────────────────────────────
    println!("=== Summary ===");
    println!("For a corpus of 1 million 512-dimensional vectors:");
    println!("  Float:   {:.2} GB", (float_size * 1_000_000) as f64 / 1_073_741_824.0);
    println!("  Int8:    {:.2} GB (4× smaller)", (int8_size * 1_000_000) as f64 / 1_073_741_824.0);
    println!("  Binary:  {:.2} GB (32× smaller)", (binary_size * 1_000_000) as f64 / 1_073_741_824.0);
    println!("\nRecommendation: Use Int8 for production (best quality/cost trade-off)");

    Ok(())
}

/// Calculates storage size in bytes for a single embedding vector.
fn calculate_storage_bytes(embed: &mongodb_voyageai::Embed, bytes_per_element: usize) -> usize {
    embed.embedding(0).unwrap().len() * bytes_per_element
}
