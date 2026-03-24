use mongodb_voyageai::{Client, Config, model};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::new(&Config::new())?;

    let query = "Who should I call if my pipes are leaking?";

    let documents = vec![
        "John is a musician who plays guitar in a local band.".into(),
        "Paul is a licensed plumber with 15 years of experience.".into(),
        "George is a high school math teacher.".into(),
        "Ringo is a doctor specializing in cardiology.".into(),
        "Lisa is a corporate lawyer at a downtown firm.".into(),
        "Stuart is a painter who exhibits in galleries.".into(),
    ];

    // Rerank all documents — no top_k filter
    println!("=== Full ranking ===");
    let rerank = client
        .rerank(query, documents.clone(), None, None, None)
        .await?;

    println!("Query: {:?}", query);
    println!("Model: {}", rerank.model);
    println!("Tokens used: {}", rerank.usage.total_tokens);
    println!();
    for result in &rerank.results {
        let doc = &documents[result.index];
        println!(
            "  [{}] score={:.4}  {:?}",
            result.index, result.relevance_score, doc
        );
    }

    // Rerank with top_k=2 to get only the best matches
    println!("\n=== Top 2 only ===");
    let top2 = client
        .rerank(query, documents.clone(), None, Some(2), None)
        .await?;

    for result in &top2.results {
        let doc = &documents[result.index];
        println!(
            "  [{}] score={:.4}  {:?}",
            result.index, result.relevance_score, doc
        );
    }

    // Rerank with the lite model
    println!("\n=== Using rerank-2-lite ===");
    let lite = client
        .rerank(
            query,
            documents.clone(),
            Some(model::RERANK_LITE),
            Some(3),
            None,
        )
        .await?;

    println!("Model: {}", lite.model);
    for result in &lite.results {
        let doc = &documents[result.index];
        println!(
            "  [{}] score={:.4}  {:?}",
            result.index, result.relevance_score, doc
        );
    }

    Ok(())
}
