use mongodb_voyageai::{Client, Config, Error, model};

fn mock_config(server_url: String) -> Config {
    Config {
        api_key: Some("test_api_key".into()),
        host: server_url,
        version: "v1".into(),
        timeout: None,
    }
}

#[tokio::test]
async fn embed_success() {
    let mut server = mockito::Server::new_async().await;
    let mock = server
        .mock("POST", "/v1/embeddings")
        .match_header("authorization", "Bearer test_api_key")
        .match_header("content-type", "application/json")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            r#"{
            "object": "list",
            "data": [{"object": "embedding", "embedding": [0.0, 1.0, 2.0, 3.0]}],
            "model": "voyage-3.5",
            "usage": {"total_tokens": 4}
        }"#,
        )
        .create_async()
        .await;

    let client = Client::new(&mock_config(server.url())).unwrap();
    let embed = client
        .embed(vec!["Greetings!".into()], None, None, None, None)
        .await
        .unwrap();

    assert_eq!(embed.model, "voyage-3.5");
    assert_eq!(embed.usage.total_tokens, 4);
    assert_eq!(embed.embeddings, vec![vec![0.0, 1.0, 2.0, 3.0]]);
    mock.assert_async().await;
}

#[tokio::test]
async fn embed_with_all_params() {
    let mut server = mockito::Server::new_async().await;
    let mock = server
        .mock("POST", "/v1/embeddings")
        .match_body(mockito::Matcher::Json(serde_json::json!({
            "input": ["Test input"],
            "model": "voyage-3-large",
            "input_type": "document",
            "truncation": true,
            "output_dimension": 512
        })))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            r#"{
            "object": "list",
            "data": [{"object": "embedding", "embedding": [0.1, 0.2]}],
            "model": "voyage-3-large",
            "usage": {"total_tokens": 5}
        }"#,
        )
        .create_async()
        .await;

    let client = Client::new(&mock_config(server.url())).unwrap();
    let embed = client
        .embed(
            vec!["Test input".into()],
            Some(model::VOYAGE_3_LARGE),
            Some("document"),
            Some(true),
            Some(512),
        )
        .await
        .unwrap();

    assert_eq!(embed.model, "voyage-3-large");
    assert_eq!(embed.embeddings[0], vec![0.1, 0.2]);
    mock.assert_async().await;
}

#[tokio::test]
async fn embed_multiple_inputs() {
    let mut server = mockito::Server::new_async().await;
    let mock = server
        .mock("POST", "/v1/embeddings")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            r#"{
            "object": "list",
            "data": [
                {"object": "embedding", "embedding": [0.1], "index": 0},
                {"object": "embedding", "embedding": [0.2], "index": 1},
                {"object": "embedding", "embedding": [0.3], "index": 2}
            ],
            "model": "voyage-3.5",
            "usage": {"total_tokens": 15}
        }"#,
        )
        .create_async()
        .await;

    let client = Client::new(&mock_config(server.url())).unwrap();
    let embed = client
        .embed(
            vec!["one".into(), "two".into(), "three".into()],
            None,
            None,
            None,
            None,
        )
        .await
        .unwrap();

    assert_eq!(embed.embeddings.len(), 3);
    assert_eq!(embed.usage.total_tokens, 15);
    assert_eq!(embed.embedding(0), Some(&vec![0.1]));
    assert_eq!(embed.embedding(1), Some(&vec![0.2]));
    assert_eq!(embed.embedding(2), Some(&vec![0.3]));
    mock.assert_async().await;
}

#[tokio::test]
async fn embed_custom_model() {
    let mut server = mockito::Server::new_async().await;
    let mock = server
        .mock("POST", "/v1/embeddings")
        .match_body(mockito::Matcher::PartialJson(serde_json::json!({
            "model": "voyage-code-2"
        })))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            r#"{
            "object": "list",
            "data": [{"object": "embedding", "embedding": [1.0]}],
            "model": "voyage-code-2",
            "usage": {"total_tokens": 1}
        }"#,
        )
        .create_async()
        .await;

    let client = Client::new(&mock_config(server.url())).unwrap();
    let embed = client
        .embed(
            vec!["fn main() {}".into()],
            Some(model::VOYAGE_CODE),
            None,
            None,
            None,
        )
        .await
        .unwrap();

    assert_eq!(embed.model, "voyage-code-2");
    mock.assert_async().await;
}

#[tokio::test]
async fn embed_failure_500() {
    let mut server = mockito::Server::new_async().await;
    let mock = server
        .mock("POST", "/v1/embeddings")
        .with_status(500)
        .with_header("content-type", "application/json")
        .with_body(r#"{"error": "An unknown error occurred."}"#)
        .create_async()
        .await;

    let client = Client::new(&mock_config(server.url())).unwrap();
    let result = client
        .embed(vec!["test".into()], None, None, None, None)
        .await;

    match result.unwrap_err() {
        Error::RequestError { status, body } => {
            assert_eq!(status, 500);
            assert!(body.contains("unknown error"));
        }
        other => panic!("expected RequestError, got: {other:?}"),
    }
    mock.assert_async().await;
}

#[tokio::test]
async fn embed_failure_401() {
    let mut server = mockito::Server::new_async().await;
    let mock = server
        .mock("POST", "/v1/embeddings")
        .with_status(401)
        .with_header("content-type", "application/json")
        .with_body(r#"{"error": "Invalid API key."}"#)
        .create_async()
        .await;

    let client = Client::new(&mock_config(server.url())).unwrap();
    match client
        .embed(vec!["test".into()], None, None, None, None)
        .await
        .unwrap_err()
    {
        Error::RequestError { status, body } => {
            assert_eq!(status, 401);
            assert!(body.contains("Invalid API key"));
        }
        other => panic!("expected RequestError, got: {other:?}"),
    }
    mock.assert_async().await;
}

#[tokio::test]
async fn embed_failure_429() {
    let mut server = mockito::Server::new_async().await;
    let mock = server
        .mock("POST", "/v1/embeddings")
        .with_status(429)
        .with_header("content-type", "application/json")
        .with_body(r#"{"error": "Rate limit exceeded."}"#)
        .create_async()
        .await;

    let client = Client::new(&mock_config(server.url())).unwrap();
    match client
        .embed(vec!["test".into()], None, None, None, None)
        .await
        .unwrap_err()
    {
        Error::RequestError { status, .. } => assert_eq!(status, 429),
        other => panic!("expected RequestError, got: {other:?}"),
    }
    mock.assert_async().await;
}

#[tokio::test]
async fn embed_uses_custom_version_in_url() {
    let mut server = mockito::Server::new_async().await;
    let mock = server
        .mock("POST", "/v2/embeddings")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            r#"{
            "object": "list",
            "data": [{"object": "embedding", "embedding": [1.0]}],
            "model": "voyage-3.5",
            "usage": {"total_tokens": 1}
        }"#,
        )
        .create_async()
        .await;

    let config = Config {
        api_key: Some("key".into()),
        host: server.url(),
        version: "v2".into(),
        timeout: None,
    };
    let client = Client::new(&config).unwrap();
    assert!(
        client
            .embed(vec!["test".into()], None, None, None, None)
            .await
            .is_ok()
    );
    mock.assert_async().await;
}
