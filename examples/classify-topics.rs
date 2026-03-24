use mongodb_voyageai::{Client, Config};

fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::new(&Config::new())?;

    // Define topic labels
    let topics = [
        "Technology and programming",
        "Cooking and food",
        "Sports and athletics",
        "Music and entertainment",
        "Science and research",
    ];

    // Embed the topic labels
    let topic_embeds = client
        .embed(
            topics.iter().map(|s| s.to_string()).collect(),
            None,
            Some("document"),
            None,
            None,
        )
        .await?;

    // Texts to classify
    let texts = [
        "Rust's borrow checker prevents data races at compile time.",
        "The sourdough bread needs to proof for at least 12 hours.",
        "She finished the marathon in under three hours.",
        "The new album features a blend of jazz and electronic music.",
        "The researchers published their findings on CRISPR gene editing.",
        "Python is great for machine learning prototyping.",
        "Add a pinch of saffron to the risotto for extra flavor.",
        "The goalkeeper made an incredible save in the final minute.",
    ];

    // Embed the texts
    let text_embeds = client
        .embed(
            texts.iter().map(|s| s.to_string()).collect(),
            None,
            Some("document"),
            None,
            None,
        )
        .await?;

    println!(
        "Tokens used: {}\n",
        text_embeds.usage.total_tokens + topic_embeds.usage.total_tokens
    );

    // Classify each text by finding the most similar topic
    for (i, text) in texts.iter().enumerate() {
        let text_emb = text_embeds.embedding(i).unwrap();

        let mut best_topic = "";
        let mut best_score = f64::NEG_INFINITY;

        for (j, topic) in topics.iter().enumerate() {
            let topic_emb = topic_embeds.embedding(j).unwrap();
            let sim = cosine_similarity(text_emb, topic_emb);
            if sim > best_score {
                best_score = sim;
                best_topic = topic;
            }
        }

        println!("[{:.4}] {:<45} => {}", best_score, text, best_topic);
    }

    Ok(())
}
