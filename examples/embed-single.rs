//! Basic embedding example.
//!
//! Shows the two most common embedding patterns:
//!
//! 1. **Default model** — embed a single piece of text with no extra options.
//! 2. **Explicit model + input type** — embed a query with a specific model,
//!    using `input_type("query")` for better asymmetric retrieval quality.
//!
//! # When to use `input_type`
//!
//! VoyageAI models support asymmetric search: the vector space is optimised
//! differently depending on whether the text is a *query* (short, intent-heavy)
//! or a *document* (longer, content-heavy). Always pass `input_type("query")`
//! when embedding search queries and `input_type("document")` when embedding
//! passages you intend to store and retrieve later.

use mongodb_voyageai::{Client, model};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Reads VOYAGEAI_API_KEY from the environment
    let client = Client::from_env();

    // ── Example 1: single &str, default model ────────────────────────────────
    // `embed` accepts a bare `&str` — no `vec![]` or `.into()` needed.
    // The default model is `model::VOYAGE`; omitting `.model()` is equivalent
    // to calling `.model(model::VOYAGE)` explicitly.
    let text = "Rust is a systems programming language focused on safety and performance.";
    let embed = client.embed(text).send().await?;

    println!("Model:        {}", embed.model);
    println!("Tokens used:  {}", embed.usage.total_tokens);

    // `embedding(i)` returns a `&[f64]` for the i-th input.
    // It returns `None` if the index is out of range.
    let embedding = embed.embedding(0).expect("no embedding returned");
    println!("Dimensions:   {}", embedding.len());
    println!("First 5 values: {:?}", &embedding[..5.min(embedding.len())]);

    // ── Example 2: explicit model + input_type ───────────────────────────────
    // Using `input_type("query")` tells the model this text is a search query,
    // which improves retrieval quality when the stored documents were embedded
    // with `input_type("document")`.
    let embed_query = client
        .embed("What is Rust?")
        .model(model::VOYAGE_3)
        .input_type("query")
        .send()
        .await?;

    println!("\nQuery model:      {}", embed_query.model);
    println!(
        "Query dimensions: {}",
        embed_query.embedding(0).unwrap().len()
    );

    Ok(())
}