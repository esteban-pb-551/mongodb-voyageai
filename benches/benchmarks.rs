use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use mongodb_voyageai::{
    Client, Config, ContextualizedEmbed, Embed, OutputDtype, Rerank, Reranking, Usage, client,
    model,
};
use mongodb_voyageai::chunk::chunking::{ChunkConfig, chunk_by_sentences, chunk_fixed_size, chunk_recursive};
use mongodb_voyageai::chunk::normalizer::{NormalizerConfig, normalize};
use mongodb_voyageai::pairwise::cosine_similarity::{cosine_similarity, k_nearest_neighbors};
use ndarray::{Array1, Array2};

// ---------------------------------------------------------------------------
// JSON fixtures
// ---------------------------------------------------------------------------

fn embed_json_single() -> String {
    r#"{
        "object": "list",
        "data": [{"object": "embedding", "embedding": [0.1, 0.2, 0.3, 0.4, 0.5], "index": 0}],
        "model": "voyage-4",
        "usage": {"total_tokens": 5}
    }"#
    .to_string()
}

fn embed_json_multi(n: usize, dims: usize) -> String {
    let entries: Vec<String> = (0..n)
        .map(|i| {
            let values: Vec<String> = (0..dims)
                .map(|j| format!("{:.6}", (i * dims + j) as f32 * 0.001))
                .collect();
            format!(
                r#"{{"object": "embedding", "embedding": [{}], "index": {}}}"#,
                values.join(", "),
                i
            )
        })
        .collect();
    format!(
        r#"{{"object": "list", "data": [{}], "model": "voyage-4", "usage": {{"total_tokens": {}}}}}"#,
        entries.join(", "),
        n * 10
    )
}

fn contextualized_embed_json(n_docs: usize, n_chunks: usize, dims: usize) -> String {
    let docs: Vec<String> = (0..n_docs)
        .map(|d| {
            let chunks: Vec<String> = (0..n_chunks)
                .map(|c| {
                    let values: Vec<String> = (0..dims)
                        .map(|j| format!("{:.4}", ((d * n_chunks + c) * dims + j) as f32 * 0.0001))
                        .collect();
                    format!(
                        r#"{{"object": "embedding", "embedding": [{}], "index": {}}}"#,
                        values.join(", "),
                        c
                    )
                })
                .collect();
            format!(
                r#"{{"object": "list", "data": [{}], "index": {}}}"#,
                chunks.join(", "),
                d
            )
        })
        .collect();
    format!(
        r#"{{"object": "list", "data": [{}], "model": "voyage-context-3", "usage": {{"total_tokens": {}}}}}"#,
        docs.join(", "),
        n_docs * n_chunks * 10
    )
}

fn rerank_json(n: usize) -> String {
    let entries: Vec<String> = (0..n)
        .map(|i| {
            format!(
                r#"{{"index": {}, "document": "Document number {}", "relevance_score": {:.4}}}"#,
                i,
                i,
                1.0 - (i as f32 * 0.1)
            )
        })
        .collect();
    format!(
        r#"{{"object": "list", "data": [{}], "model": "rerank-2", "usage": {{"total_tokens": {}}}}}"#,
        entries.join(", "),
        n * 5
    )
}

fn sample_text_short() -> &'static str {
    "Rust is a systems programming language focused on safety and performance."
}

fn sample_text_long() -> String {
    let paragraphs = [
        "Rust is a systems programming language focused on safety, speed, and concurrency. \
         It achieves memory safety without a garbage collector through its ownership system. \
         The borrow checker enforces strict rules at compile time.",
        "The language was originally designed by Graydon Hoare at Mozilla Research. \
         It has since grown into a large open-source project with contributions from \
         hundreds of developers around the world.",
        "Rust's type system and ownership model guarantee memory safety and thread safety. \
         This enables developers to write concurrent code without fear of data races. \
         The compiler catches many classes of bugs at compile time.",
        "The ecosystem includes Cargo, a powerful package manager and build system. \
         Crates.io hosts over 100,000 packages covering everything from web frameworks \
         to embedded systems libraries.",
        "Major companies like Amazon, Google, Microsoft, and Meta use Rust in production. \
         It has been voted the most loved programming language in Stack Overflow surveys \
         for multiple consecutive years.",
    ];
    paragraphs.join("\n\n")
}

fn random_matrix(rows: usize, cols: usize) -> Array2<f32> {
    let data: Vec<f32> = (0..rows * cols)
        .map(|i| ((i as f32 * 0.7071) % 2.0) - 1.0)
        .collect();
    Array2::from_shape_vec((rows, cols), data).unwrap()
}

// ---------------------------------------------------------------------------
// Parsing benchmarks
// ---------------------------------------------------------------------------

fn bench_usage_parse(c: &mut Criterion) {
    let json = r#"{"total_tokens": 42}"#;
    c.bench_function("usage_parse", |b| {
        b.iter(|| {
            let _: Usage = serde_json::from_str(black_box(json)).unwrap();
        });
    });
}

fn bench_reranking_parse(c: &mut Criterion) {
    let json = r#"{"index": 0, "document": "Sample document text", "relevance_score": 0.875}"#;
    c.bench_function("reranking_parse", |b| {
        b.iter(|| {
            let _: Reranking = serde_json::from_str(black_box(json)).unwrap();
        });
    });
}

fn bench_embed_parse(c: &mut Criterion) {
    let mut group = c.benchmark_group("embed_parse");

    let json_1 = embed_json_single();
    group.bench_function("1x5d", |b| {
        b.iter(|| Embed::parse(black_box(&json_1)).unwrap());
    });

    for (n, dims) in [(10, 128), (10, 512), (10, 1024), (50, 512), (100, 512)] {
        let json = embed_json_multi(n, dims);
        group.bench_with_input(
            BenchmarkId::new("parse", format!("{n}x{dims}d")),
            &json,
            |b, json| {
                b.iter(|| Embed::parse(black_box(json)).unwrap());
            },
        );
    }

    group.finish();
}

fn bench_contextualized_embed_parse(c: &mut Criterion) {
    let mut group = c.benchmark_group("contextualized_embed_parse");

    for (n_docs, n_chunks, dims) in [(2, 4, 512), (5, 8, 512), (10, 4, 1024)] {
        let json = contextualized_embed_json(n_docs, n_chunks, dims);
        group.bench_with_input(
            BenchmarkId::new("parse", format!("{n_docs}docs_{n_chunks}chunks_{dims}d")),
            &json,
            |b, json| {
                b.iter(|| ContextualizedEmbed::parse(black_box(json)).unwrap());
            },
        );
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
        output_dtype: None,
    };
    group.bench_function("minimal", |b| {
        b.iter(|| serde_json::to_string(black_box(&minimal)).unwrap());
    });

    let full = client::EmbedInput {
        input: vec!["hello world".into()],
        model: model::VOYAGE_4_LARGE.into(),
        input_type: Some("document".into()),
        truncation: Some(true),
        output_dimension: Some(512),
        output_dtype: Some(OutputDtype::Int8),
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
        output_dtype: None,
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
// Client construction & accessor benchmarks
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

fn bench_embed_accessor(c: &mut Criterion) {
    let mut group = c.benchmark_group("embed_accessor");

    for (n, dims) in [(100, 128), (100, 512), (100, 1024)] {
        let json = embed_json_multi(n, dims);
        let embed = Embed::parse(&json).unwrap();

        group.bench_function(format!("{n}x{dims}d"), |b| {
            b.iter(|| {
                for i in 0..n {
                    black_box(embed.embedding(i));
                }
            });
        });
    }

    group.finish();
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
                    "model": "voyage-4",
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
            let result = client.embed("benchmark").send().await;
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
            let result = client.rerank("query", vec!["document"]).send().await;
            black_box(result.unwrap());
        });
    });
}

// ---------------------------------------------------------------------------
// Cosine similarity & KNN benchmarks
// ---------------------------------------------------------------------------

fn bench_cosine_similarity(c: &mut Criterion) {
    let mut group = c.benchmark_group("cosine_similarity");

    for (rows, cols) in [(10, 128), (10, 512), (10, 1024), (100, 512), (1000, 512)] {
        let x = random_matrix(rows, cols);
        let y = random_matrix(rows, cols);

        group.bench_function(format!("{rows}x{cols}d"), |b| {
            b.iter(|| cosine_similarity(black_box(x.view()), Some(black_box(y.view()))));
        });
    }

    // Self-similarity (y = None)
    let x = random_matrix(100, 512);
    group.bench_function("self_100x512d", |b| {
        b.iter(|| cosine_similarity(black_box(x.view()), None));
    });

    group.finish();
}

fn bench_knn(c: &mut Criterion) {
    let mut group = c.benchmark_group("k_nearest_neighbors");

    for (n_docs, dims, k) in [(100, 512, 5), (100, 512, 10), (1000, 512, 10), (1000, 1024, 5)] {
        let query = Array1::from_vec((0..dims).map(|i| ((i as f32 * 0.7071) % 2.0) - 1.0).collect());
        let docs = random_matrix(n_docs, dims);

        group.bench_function(format!("{n_docs}docs_{dims}d_top{k}"), |b| {
            b.iter(|| {
                k_nearest_neighbors(
                    black_box(query.view()),
                    black_box(docs.view()),
                    black_box(k),
                )
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Chunking benchmarks
// ---------------------------------------------------------------------------

fn bench_chunking(c: &mut Criterion) {
    let mut group = c.benchmark_group("chunking");

    let short = sample_text_short();
    let long = sample_text_long();

    let config_small = ChunkConfig {
        chunk_size: 200,
        chunk_overlap: 50,
    };
    let config_large = ChunkConfig {
        chunk_size: 500,
        chunk_overlap: 80,
    };

    // Fixed-size chunking
    group.bench_function("fixed_short_200", |b| {
        b.iter(|| chunk_fixed_size(black_box(short), black_box(&config_small)));
    });
    group.bench_function("fixed_long_200", |b| {
        b.iter(|| chunk_fixed_size(black_box(&long), black_box(&config_small)));
    });
    group.bench_function("fixed_long_500", |b| {
        b.iter(|| chunk_fixed_size(black_box(&long), black_box(&config_large)));
    });

    // Sentence-based chunking
    group.bench_function("sentence_short_200", |b| {
        b.iter(|| chunk_by_sentences(black_box(short), black_box(&config_small)));
    });
    group.bench_function("sentence_long_200", |b| {
        b.iter(|| chunk_by_sentences(black_box(&long), black_box(&config_small)));
    });
    group.bench_function("sentence_long_500", |b| {
        b.iter(|| chunk_by_sentences(black_box(&long), black_box(&config_large)));
    });

    // Recursive chunking
    group.bench_function("recursive_short_200", |b| {
        b.iter(|| chunk_recursive(black_box(short), black_box(&config_small)));
    });
    group.bench_function("recursive_long_200", |b| {
        b.iter(|| chunk_recursive(black_box(&long), black_box(&config_small)));
    });
    group.bench_function("recursive_long_500", |b| {
        b.iter(|| chunk_recursive(black_box(&long), black_box(&config_large)));
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Normalizer benchmarks
// ---------------------------------------------------------------------------

fn bench_normalize(c: &mut Criterion) {
    let mut group = c.benchmark_group("normalize");

    let dirty_text = "  Hello,\r\n\n\n  world!  \t This   has \r\n extra   whitespace.\n\n\
        And   multiple\r\n\r\nparagraphs   with  «special»  "quotes"  and  em—dashes.\n\
        Check https://example.com/path?q=test for more info.\n\
        Some hy-\nphenated words across line breaks.  ";

    let long_dirty: String = (0..20).map(|_| dirty_text).collect::<Vec<_>>().join("\n\n");

    let presets = [
        ("prose", NormalizerConfig::prose()),
        ("code_docs", NormalizerConfig::code_docs()),
        ("web_scraped", NormalizerConfig::web_scraped()),
        ("single_line", NormalizerConfig::single_line()),
    ];

    // Short text with all presets
    for (name, config) in &presets {
        group.bench_function(format!("short_{name}"), |b| {
            b.iter(|| normalize(black_box(dirty_text), black_box(config)));
        });
    }

    // Long text with prose preset
    group.bench_function("long_prose", |b| {
        b.iter(|| normalize(black_box(&long_dirty), black_box(&NormalizerConfig::prose())));
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Quantization serialization benchmarks
// ---------------------------------------------------------------------------

fn bench_output_dtype_serialize(c: &mut Criterion) {
    let mut group = c.benchmark_group("output_dtype_serialize");

    for dtype in [
        OutputDtype::Float,
        OutputDtype::Int8,
        OutputDtype::Uint8,
        OutputDtype::Binary,
        OutputDtype::Ubinary,
    ] {
        let name = format!("{:?}", dtype).to_lowercase();
        group.bench_function(&name, |b| {
            b.iter(|| serde_json::to_string(black_box(&dtype)).unwrap());
        });
    }

    group.finish();
}

fn bench_quantization_embed_builder(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantization_embed_builder");

    let config = Config {
        api_key: Some("bench-key".into()),
        ..Config::default()
    };
    let client = Client::new(&config).unwrap();

    group.bench_function("no_quantization", |b| {
        b.iter(|| {
            let builder = client
                .embed(black_box("test text"))
                .model(model::VOYAGE_4_LARGE)
                .output_dimension(512);
            black_box(builder);
        });
    });

    group.bench_function("with_int8", |b| {
        b.iter(|| {
            let builder = client
                .embed(black_box("test text"))
                .model(model::VOYAGE_4_LARGE)
                .output_dimension(512)
                .output_dtype(OutputDtype::Int8);
            black_box(builder);
        });
    });

    group.bench_function("with_binary", |b| {
        b.iter(|| {
            let builder = client
                .embed(black_box("test text"))
                .model(model::VOYAGE_4_LARGE)
                .output_dimension(512)
                .output_dtype(OutputDtype::Binary);
            black_box(builder);
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Criterion groups
// ---------------------------------------------------------------------------

criterion_group!(
    parsing,
    bench_usage_parse,
    bench_reranking_parse,
    bench_embed_parse,
    bench_contextualized_embed_parse,
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

criterion_group!(
    vectorops,
    bench_cosine_similarity,
    bench_knn,
);

criterion_group!(
    text_processing,
    bench_chunking,
    bench_normalize,
);

criterion_group!(
    quantization,
    bench_output_dtype_serialize,
    bench_quantization_embed_builder,
);

criterion_main!(parsing, serialization, client, vectorops, text_processing, quantization);
