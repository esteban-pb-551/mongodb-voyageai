//! Example: Contextualized chunk embeddings
//!
//! This example demonstrates how to use contextualized chunk embeddings
//! to maintain document context when embedding chunks.
//!
//! Run with:
//! ```sh
//! cargo run --example contextualized-embeddings
//! ```

use mongodb_voyageai::{Client, OutputDtype};

#[tokio::main]
async fn main() -> Result<(), mongodb_voyageai::Error> {
    // Create client from environment variable VOYAGEAI_API_KEY
    let client = Client::from_env();

    // Example 1: Document chunks with context
    println!("Example 1: Document chunks with context");
    let documents = vec![
        vec![
            "This is the SEC filing on Greenery Corp.'s Q2 2024 performance.",
            "The company's revenue increased by 7% compared to the previous quarter.",
        ],
        vec![
            "This is the SEC filing on Leafy Inc.'s Q2 2024 performance.",
            "The company's revenue increased by 15% compared to the previous quarter.",
        ],
    ];

    let embed = client
        .contextualized_embed(documents)
        .model("voyage-context-3")
        .input_type("document")
        .send()
        .await?;

    println!("Model: {}", embed.model);
    println!("Total tokens: {}", embed.usage.total_tokens);
    println!("Number of documents: {}", embed.results.len());

    for (i, result) in embed.results.iter().enumerate() {
        println!(
            "  Document {}: {} chunks, {} dimensions",
            i,
            result.embeddings().len(),
            result.embeddings()[0].len()
        );
    }

    // Example 2: Single query (context-agnostic behavior)
    println!("\nExample 2: Single query");
    let query = vec![vec![
        "What was the revenue growth for Leafy Inc. in Q2 2024?",
    ]];

    let query_embed = client
        .contextualized_embed(query)
        .model("voyage-context-3")
        .input_type("query")
        .send()
        .await?;

    println!(
        "Query embedding dimensions: {}",
        query_embed.results[0].embeddings()[0].len()
    );

    // Example 3: With quantization for efficient storage
    println!("\nExample 3: With quantization");
    let chunks = vec![vec!["First chunk of document", "Second chunk of document"]];

    let quantized_embed = client
        .contextualized_embed(chunks)
        .model("voyage-context-3")
        .input_type("document")
        .output_dimension(512)
        .output_dtype(OutputDtype::Int8)
        .send()
        .await?;

    println!(
        "Quantized embedding dimensions: {}",
        quantized_embed.results[0].embeddings()[0].len()
    );
    println!("Total tokens used: {}", quantized_embed.usage.total_tokens);

    Ok(())
}
