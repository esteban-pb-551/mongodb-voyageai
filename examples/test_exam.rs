use mongodb_voyageai::Embed;

#[tokio::main]
async fn main() -> Result<(), mongodb_voyageai::Error> {

    let json = r#"{
        "object": "list",
        "data": [
            {"object": "embedding", "embedding": [0.1, 0.2, 0.3], "index": 0}
        ],
        "model": "voyage-4",
        "usage": {"total_tokens": 5}
    }"#;

    let embed = Embed::parse(json).unwrap();
    assert_eq!(embed.model, "voyage-4");
    assert_eq!(embed.usage.total_tokens, 5);
    assert_eq!(embed.embeddings.len(), 1);
    assert_eq!(embed.embedding(0).unwrap(), &vec![0.1, 0.2, 0.3]);
    Ok(())
}