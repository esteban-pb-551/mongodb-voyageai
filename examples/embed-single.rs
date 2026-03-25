use mongodb_voyageai::{Client, Config, model};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::new(&Config::new())?;

    // Embed a single piece of text
    let text = "Rust is a systems programming language focused on safety and performance.";
    let embed = client.embed(vec![text.into()]).send().await?;

    println!("Model: {}", embed.model);
    println!("Tokens used: {}", embed.usage.total_tokens);

    let embedding = embed.embedding(0).expect("no embedding returned");
    println!("Dimensions: {}", embedding.len());
    println!("First 5 values: {:?}", &embedding[..5.min(embedding.len())]);

    // Embed with a specific model and input type
    let embed_query = client
        .embed(vec!["What is Rust?".into()])
        .model(model::VOYAGE_3)
        .input_type("query")
        .send()
        .await?;

    println!("\nQuery embedding model: {}", embed_query.model);
    println!(
        "Query embedding dimensions: {}",
        embed_query.embedding(0).unwrap().len()
    );

    Ok(())
}
