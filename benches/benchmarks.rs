use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use mongodb_voyageai::{Client, Config, Embed, Rerank, Reranking, Usage, client, model};

// ---------------------------------------------------------------------------
// JSON fixtures
// ---------------------------------------------------------------------------

fn embed_json_single() -> String {
    r#"{
        "object": "list",
        "data": [{"object": "embedding", "embedding": [0.1, 0.2, 0.3, 0.4, 0.5], "index": 0}],
        "model": "voyage-3.5",
        "usage": {"total_tokens": 5}
    }"#
    .to_string()
}

fn embed_json_multi(n: usize) -> String {
    let entries: Vec<String> = (0..n)
        .map(|i| {
            format!(
                r#"{{"object": "embedding", "embedding": [{}], "index": {}}}"#,
                (0..128)
                    .map(|j| format!("{:.6}", (i * 128 + j) as f64 * 0.001))
                    .collect::<Vec<_>>()
                    .join(", "),
                i
            )
        })
        .collect();
    format!(
        r#"{{"object": "list", "data": [{}], "model": "voyage-3.5", "usage": {{"total_tokens": {}}}}}"#,
        entries.join(", "),
        n * 10
    )
}

fn rerank_json(n: usize) -> String {
    let entries: Vec<String> = (0..n)
        .map(|i| {
            format!(
                r#"{{"index": {}, "document": "Document number {}", "relevance_score": {:.4}}}"#,
                i,
                i,
                1.0 - (i as f64 * 0.1)
            )
        })
        .collect();
    format!(
        r#"{{"object": "list", "data": [{}], "model": "rerank-2", "usage": {{"total_tokens": {}}}}}"#,
        entries.join(", "),
        n * 5
    )
}

fn usage_json() -> String {
    r#"{"total_tokens": 42}"#.to_string()
}

fn reranking_json() -> String {
    r#"{"index": 0, "document": "Sample document text", "relevance_score": 0.875}"#.to_string()
}

// ---------------------------------------------------------------------------
// Parsing benchmarks
// ---------------------------------------------------------------------------

fn bench_usage_parse(c: &mut Criterion) {
    let json = usage_json();
    c.bench_function("usage_parse", |b| {
        b.iter(|| {
            let _: Usage = serde_json::from_str(black_box(&json)).unwrap();
        });
    });
}

fn bench_reranking_parse(c: &mut Criterion) {
    let json = reranking_json();
    c.bench_function("reranking_parse", |b| {
        b.iter(|| {
            let _: Reranking = serde_json::from_str(black_box(&json)).unwrap();
        });
    });
}

fn bench_embed_parse(c: &mut Criterion) {
    let mut group = c.benchmark_group("embed_parse");

    let json_1 = embed_json_single();
    group.bench_function("1_embedding", |b| {
        b.iter(|| Embed::parse(black_box(&json_1)).unwrap());
    });

    for n in [10, 50, 100] {
        let json = embed_json_multi(n);
        group.bench_with_input(BenchmarkId::new("n_embeddings", n), &json, |b, json| {
            b.iter(|| Embed::parse(black_box(json)).unwrap());
        });
    }

    group.finish();
}

fn bench_rerank_parse(c: &mut Criterion) {
    let mut group = c.benchmark_group("rerank_parse");

    for n in [1, 10, 50, 100] {
        let json = rerank_json(n);
        group.bench_with_input(BenchmarkId::new("n_results", n), &json, |b, json| {
            b.iter(|| Rerank::parse(black_box(json)).unwrap());
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Serialization benchmarks
// ---------------------------------------------------------------------------

fn bench_embed_input_serialize(c: &mut Criterion) {
    let mut group = c.benchmark_group("embed_input_serialize");

    let minimal = client::EmbedInput {
        input: vec!["hello world".into()],
        model: model::VOYAGE.into(),
        input_type: None,
        truncation: None,
        output_dimension: None,
    };
    group.bench_function("minimal", |b| {
        b.iter(|| serde_json::to_string(black_box(&minimal)).unwrap());
    });

    let full = client::EmbedInput {
        input: vec!["hello world".into()],
        model: model::VOYAGE_3_LARGE.into(),
        input_type: Some("document".into()),
        truncation: Some(true),
        output_dimension: Some(512),
    };
    group.bench_function("full_params", |b| {
        b.iter(|| serde_json::to_string(black_box(&full)).unwrap());
    });

    let large_batch: Vec<String> = (0..100).map(|i| format!("Document number {i}")).collect();
    let batch = client::EmbedInput {
        input: large_batch,
        model: model::VOYAGE.into(),
        input_type: Some("document".into()),
        truncation: None,
        output_dimension: None,
    };
    group.bench_function("100_inputs", |b| {
        b.iter(|| serde_json::to_string(black_box(&batch)).unwrap());
    });

    group.finish();
}

fn bench_rerank_input_serialize(c: &mut Criterion) {
    let mut group = c.benchmark_group("rerank_input_serialize");

    let small = client::RerankInput {
        query: "search query".into(),
        documents: vec!["doc A".into(), "doc B".into()],
        model: model::RERANK.into(),
        top_k: None,
        truncation: None,
    };
    group.bench_function("2_docs", |b| {
        b.iter(|| serde_json::to_string(black_box(&small)).unwrap());
    });

    let large_docs: Vec<String> = (0..50)
        .map(|i| format!("This is document number {i} with some additional content for reranking"))
        .collect();
    let large = client::RerankInput {
        query: "What is the most relevant document?".into(),
        documents: large_docs,
        model: model::RERANK.into(),
        top_k: Some(10),
        truncation: Some(true),
    };
    group.bench_function("50_docs", |b| {
        b.iter(|| serde_json::to_string(black_box(&large)).unwrap());
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Client construction benchmark
// ---------------------------------------------------------------------------

fn bench_client_construction(c: &mut Criterion) {
    let config = Config {
        api_key: Some("pa-benchmark-key-12345".into()),
        ..Config::default()
    };
    c.bench_function("client_new", |b| {
        b.iter(|| Client::new(black_box(&config)).unwrap());
    });
}

// ---------------------------------------------------------------------------
// Embed accessor benchmark
// ---------------------------------------------------------------------------

fn bench_embed_accessor(c: &mut Criterion) {
    let json = embed_json_multi(100);
    let embed = Embed::parse(&json).unwrap();

    c.bench_function("embed_accessor_100", |b| {
        b.iter(|| {
            for i in 0..100 {
                black_box(embed.embedding(i));
            }
        });
    });
}

// ---------------------------------------------------------------------------
// Async HTTP round-trip benchmarks (mockito)
// ---------------------------------------------------------------------------

fn bench_client_embed_roundtrip(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    c.bench_function("client_embed_roundtrip", |b| {
        b.to_async(&rt).iter(|| async {
            let mut server = mockito::Server::new_async().await;
            let _mock = server
                .mock("POST", "/v1/embeddings")
                .with_status(200)
                .with_header("content-type", "application/json")
                .with_body(
                    r#"{
                    "object": "list",
                    "data": [{"object": "embedding", "embedding": [0.1, 0.2, 0.3]}],
                    "model": "voyage-3.5",
                    "usage": {"total_tokens": 3}
                }"#,
                )
                .create_async()
                .await;

            let config = Config {
                api_key: Some("bench-key".into()),
                host: server.url(),
                version: "v1".into(),
                timeout: None,
            };
            let client = Client::new(&config).unwrap();
            let result = client
                .embed(vec!["benchmark".into()], None, None, None, None)
                .await;
            black_box(result.unwrap());
        });
    });
}

fn bench_client_rerank_roundtrip(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    c.bench_function("client_rerank_roundtrip", |b| {
        b.to_async(&rt).iter(|| async {
            let mut server = mockito::Server::new_async().await;
            let _mock = server
                .mock("POST", "/v1/rerank")
                .with_status(200)
                .with_header("content-type", "application/json")
                .with_body(
                    r#"{
                    "object": "list",
                    "data": [{"index": 0, "relevance_score": 0.9}],
                    "model": "rerank-2",
                    "usage": {"total_tokens": 5}
                }"#,
                )
                .create_async()
                .await;

            let config = Config {
                api_key: Some("bench-key".into()),
                host: server.url(),
                version: "v1".into(),
                timeout: None,
            };
            let client = Client::new(&config).unwrap();
            let result = client
                .rerank("query", vec!["document".into()], None, None, None)
                .await;
            black_box(result.unwrap());
        });
    });
}

// ---------------------------------------------------------------------------
// Criterion groups
// ---------------------------------------------------------------------------

criterion_group!(
    parsing,
    bench_usage_parse,
    bench_reranking_parse,
    bench_embed_parse,
    bench_rerank_parse,
);

criterion_group!(
    serialization,
    bench_embed_input_serialize,
    bench_rerank_input_serialize,
);

criterion_group!(
    client,
    bench_client_construction,
    bench_embed_accessor,
    bench_client_embed_roundtrip,
    bench_client_rerank_roundtrip,
);

criterion_main!(parsing, serialization, client);
