use voyageai::{Client, Config};

struct Entry {
    document: String,
    embedding: Vec<f64>,
}

fn euclidean_distance(src: &[f64], dst: &[f64]) -> f64 {
    src.iter()
        .zip(dst.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt()
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = Config::new();
    let client = Client::new(&config)?;

    let documents: Vec<String> = vec![
        "John is a musician.",
        "Paul is a plumber.",
        "George is a teacher.",
        "Ringo is a doctor.",
        "Lisa is a lawyer.",
        "Stuart is a painter.",
        "Brian is a writer.",
        "Jane is a chef.",
        "Bill is a nurse.",
        "Susan is a carpenter.",
    ]
    .into_iter()
    .map(String::from)
    .collect();

    let embed_result = client
        .embed(documents.clone(), None, Some("document"), None, None)
        .await?;

    let entries: Vec<Entry> = documents
        .iter()
        .zip(embed_result.embeddings.iter())
        .map(|(doc, emb)| Entry {
            document: doc.clone(),
            embedding: emb.clone(),
        })
        .collect();

    search(&client, &entries, "What do George and Ringo do?").await?;
    search(&client, &entries, "Who works in the medical field?").await?;

    Ok(())
}

async fn search(
    client: &Client,
    entries: &[Entry],
    query: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let query_embed = client
        .embed(vec![query.to_string()], None, Some("query"), None, None)
        .await?;
    let query_embedding = query_embed.embedding(0).expect("no embedding returned");

    let mut sorted: Vec<&Entry> = entries.iter().collect();
    sorted.sort_by(|a, b| {
        let da = euclidean_distance(&a.embedding, query_embedding);
        let db = euclidean_distance(&b.embedding, query_embedding);
        da.partial_cmp(&db).unwrap()
    });

    let nearest: Vec<String> = sorted.iter().take(4).map(|e| e.document.clone()).collect();

    let rerank_result = client
        .rerank(query, nearest.clone(), None, Some(2), None)
        .await?;

    println!("query={:?}", query);
    for reranking in &rerank_result.results {
        let document = &nearest[reranking.index];
        println!(
            "document={:?} relevance_score={}",
            document, reranking.relevance_score
        );
    }

    Ok(())
}
