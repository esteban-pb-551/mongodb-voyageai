use voyageai::{Client, Config, Error, model};

fn mock_config(server_url: String) -> Config {
    Config {
        api_key: Some("test_api_key".into()),
        host: server_url,
        version: "v1".into(),
        timeout: None,
    }
}

#[tokio::test]
async fn rerank_success() {
    let mut server = mockito::Server::new_async().await;
    let mock = server
        .mock("POST", "/v1/rerank")
        .match_header("authorization", "Bearer test_api_key")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            r#"{
            "object": "list",
            "data": [{"index": 0, "relevance_score": 0.5}],
            "model": "rerank-2",
            "usage": {"total_tokens": 8}
        }"#,
        )
        .create_async()
        .await;

    let client = Client::new(&mock_config(server.url())).unwrap();
    let rerank = client
        .rerank("Welcome!", vec!["Greetings!".into()], None, None, None)
        .await
        .unwrap();

    assert_eq!(rerank.model, "rerank-2");
    assert_eq!(rerank.usage.total_tokens, 8);
    assert_eq!(rerank.results.len(), 1);
    assert_eq!(rerank.results[0].index, 0);
    assert_eq!(rerank.results[0].relevance_score, 0.5);
    mock.assert_async().await;
}

#[tokio::test]
async fn rerank_with_all_params() {
    let mut server = mockito::Server::new_async().await;
    let mock = server
        .mock("POST", "/v1/rerank")
        .match_body(mockito::Matcher::Json(serde_json::json!({
            "query": "query",
            "documents": ["doc a", "doc b", "doc c"],
            "model": "rerank-2-lite",
            "top_k": 2,
            "truncation": true
        })))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            r#"{
            "object": "list",
            "data": [
                {"index": 1, "relevance_score": 0.9},
                {"index": 0, "relevance_score": 0.4}
            ],
            "model": "rerank-2-lite",
            "usage": {"total_tokens": 12}
        }"#,
        )
        .create_async()
        .await;

    let client = Client::new(&mock_config(server.url())).unwrap();
    let rerank = client
        .rerank(
            "query",
            vec!["doc a".into(), "doc b".into(), "doc c".into()],
            Some(model::RERANK_LITE),
            Some(2),
            Some(true),
        )
        .await
        .unwrap();

    assert_eq!(rerank.model, "rerank-2-lite");
    assert_eq!(rerank.results.len(), 2);
    assert_eq!(rerank.results[0].index, 1);
    assert_eq!(rerank.results[0].relevance_score, 0.9);
    mock.assert_async().await;
}

#[tokio::test]
async fn rerank_multiple_documents() {
    let mut server = mockito::Server::new_async().await;
    let mock = server
        .mock("POST", "/v1/rerank")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            r#"{
            "object": "list",
            "data": [
                {"index": 2, "relevance_score": 0.95},
                {"index": 0, "relevance_score": 0.80},
                {"index": 3, "relevance_score": 0.60},
                {"index": 1, "relevance_score": 0.20}
            ],
            "model": "rerank-2",
            "usage": {"total_tokens": 50}
        }"#,
        )
        .create_async()
        .await;

    let client = Client::new(&mock_config(server.url())).unwrap();
    let rerank = client
        .rerank(
            "What is Rust?",
            vec![
                "Python is a language.".into(),
                "Java is verbose.".into(),
                "Rust is a systems language.".into(),
                "Go is simple.".into(),
            ],
            None,
            None,
            None,
        )
        .await
        .unwrap();

    assert_eq!(rerank.results.len(), 4);
    assert_eq!(rerank.results[0].index, 2);
    assert!(rerank.results[0].relevance_score > rerank.results[1].relevance_score);
    mock.assert_async().await;
}

#[tokio::test]
async fn rerank_failure_500() {
    let mut server = mockito::Server::new_async().await;
    let mock = server
        .mock("POST", "/v1/rerank")
        .with_status(500)
        .with_header("content-type", "application/json")
        .with_body(r#"{"error": "An unknown error occurred."}"#)
        .create_async()
        .await;

    let client = Client::new(&mock_config(server.url())).unwrap();
    match client
        .rerank("query", vec!["doc".into()], None, None, None)
        .await
        .unwrap_err()
    {
        Error::RequestError { status, body } => {
            assert_eq!(status, 500);
            assert!(body.contains("unknown error"));
        }
        other => panic!("expected RequestError, got: {other:?}"),
    }
    mock.assert_async().await;
}

#[tokio::test]
async fn rerank_failure_422() {
    let mut server = mockito::Server::new_async().await;
    let mock = server
        .mock("POST", "/v1/rerank")
        .with_status(422)
        .with_header("content-type", "application/json")
        .with_body(r#"{"error": "Invalid model."}"#)
        .create_async()
        .await;

    let client = Client::new(&mock_config(server.url())).unwrap();
    match client
        .rerank("q", vec!["d".into()], Some("bad-model"), None, None)
        .await
        .unwrap_err()
    {
        Error::RequestError { status, body } => {
            assert_eq!(status, 422);
            assert!(body.contains("Invalid model"));
        }
        other => panic!("expected RequestError, got: {other:?}"),
    }
    mock.assert_async().await;
}
