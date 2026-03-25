# mongodb-voyageai

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE.txt)
[![Crates.io](https://img.shields.io/crates/v/mongodb-voyageai)](https://crates.io/crates/mongodb-voyageai)
[![docs.rs](https://img.shields.io/docsrs/mongodb-voyageai)](https://docs.rs/mongodb-voyageai)

An async Rust client for the [VoyageAI](https://www.voyageai.com) API — generate embeddings and rerank documents with ease.

> **Note:** Unofficial Voyage AI SDK for Rust. This is a Rust port of the original [voyageai Ruby gem](https://github.com/ksylvest/voyageai) by [Kevin Sylvestre](https://github.com/ksylvest).

## Installation

Add `mongodb-voyageai` to your `Cargo.toml`:

```toml
[dependencies]
mongodb-voyageai = "0.0.5"
tokio = { version = "1", features = ["full"] }
```

Requires **Rust edition 2024** (rustc 1.85+).

## Quick Start

```rust
use mongodb_voyageai::{Client, Config, model};

#[tokio::main]
async fn main() -> Result<(), mongodb_voyageai::Error> {
    // Reads VOYAGEAI_API_KEY from the environment by default
    let client = Client::new(&Config::new())?;

    // Generate an embedding
    let embed = client
        .embed(vec!["A quick brown fox jumps over the lazy dog.".into()])
        .model(model::VOYAGE_4_LITE)
        .send()
        .await?;

    println!("model: {}", embed.model);
    println!("tokens: {}", embed.usage.total_tokens);
    println!("embedding: {:?}", embed.embedding(0));

    Ok(())
}
```

### Multiple Embeddings

```rust
use mongodb_voyageai::{Client, Config};

#[tokio::main]
async fn main() -> Result<(), mongodb_voyageai::Error> {
    let client = Client::new(&Config::new())?;

    let input = vec![
        "John is a musician.".into(),
        "Paul is a plumber.".into(),
        "George is a teacher.".into(),
        "Ringo is a doctor.".into(),
    ];

    let embed = client
        .embed(input)
        .input_type("document")
        .send()
        .await?;

    println!("{}", embed.model);                // "voyage-4"
    println!("{}", embed.usage.total_tokens);    // total tokens used
    println!("{}", embed.embeddings.len());      // 4
    println!("{:?}", embed.embedding(0));        // Some([0.0, ...])

    Ok(())
}
```

### Embed Parameters

| Parameter          | Type            | Default              | Description                            |
|--------------------|-----------------|----------------------|----------------------------------------|
| `embed`            | `Vec<String>`   | *required*           | Texts to embed                         |
| `model`            | `Option<model>` | `VOYAGE_4`           | Model identifier                       |
| `input_type`       | `Option<&str>`  | `None`               | `"query"` or `"document"`              |
| `truncation`       | `Option<bool>`  | `None`               | Truncate inputs exceeding context       |
| `output_dimension` | `Option<u32>`   | `None`               | Reduce embedding dimensionality         |

### Reranking

```rust
use mongodb_voyageai::{Client, Config};

#[tokio::main]
async fn main() -> Result<(), mongodb_voyageai::Error> {
    let client = Client::new(&Config::new())?;

    let query = "Who is the best person to call for a toilet?";

    let documents = vec![
        "John is a musician.",
        "Paul is a plumber.",
        "George is a teacher.",
        "Ringo is a doctor.",
    ];

    let rerank = client
        .rerank(
            query,
            documents.iter().map(|s| s.to_string()).collect()
        )
        .top_k(3)
        .send()
        .await?;

    println!("Model:        {}", rerank.model);
    println!("Total tokens: {}", rerank.usage.total_tokens);
    println!();

    for result in &rerank.results {
        println!(
            "[{:.4}] {}",
            result.relevance_score, documents[result.index]
        );
    }

    Ok(())
}
```

#### Rerank Parameters

| Parameter       | Type            | Default        | Description                          |
|-----------------|-----------------|----------------|--------------------------------------|
| `query`         | `&str`          | *required*     | The search query                     |
| `documents`     | `Vec<String>`   | *required*     | Documents to rerank                  |
| `rerank_model`  | `Option<&str>`  | `"rerank-2"`   | Model identifier                     |
| `top_k`         | `Option<u32>`   | `None`         | Return only the top K results        |
| `truncation`    | `Option<bool>`  | `None`         | Truncate inputs exceeding context     |

## Configuration

The `Config` struct controls client behavior. All fields read from environment variables by default:

```rust
use std::time::Duration;
use mongodb_voyageai::{Client, Config};

let config = Config {
    api_key: Some("pa-...".into()),                   // or VOYAGEAI_API_KEY
    host: "https://api.voyageai.com".into(),           // or VOYAGEAI_HOST
    version: "v1".into(),                              // or VOYAGEAI_VERSION
    timeout: Some(Duration::from_secs(15)),
};

let client = Client::new(&config).unwrap();
```

| Field      | Env Variable        | Default                         |
|------------|---------------------|---------------------------------|
| `api_key`  | `VOYAGEAI_API_KEY`  | `None`                          |
| `host`     | `VOYAGEAI_HOST`     | `https://api.voyageai.com`      |
| `version`  | `VOYAGEAI_VERSION`  | `v1`                            |
| `timeout`  | —                   | `None`                          |

## Models

Pre-defined model constants are available in the `mongodb_voyageai::model` module:

### Embedding Models

| Constant                              | Value                     |
|---------------------------------------|---------------------------|
| `model::VOYAGE_4` (default)           | `voyage-4`                |
| `model::VOYAGE_4_LITE`                | `voyage-4-lite`           |
| `model::VOYAGE_4_LARGE`               | `voyage-4-large`          |
| `model::VOYAGE_3_5`                   | `voyage-3.5`              |
| `model::VOYAGE_3_5_LITE`              | `voyage-3.5-lite`         |
| `model::VOYAGE_3`                     | `voyage-3`                |
| `model::VOYAGE_3_LITE`                | `voyage-3-lite`           |
| `model::VOYAGE_3_LARGE`               | `voyage-3-large`          |
| `model::VOYAGE_CONTEXT_3`             | `voyage-context-3`        |
| `model::VOYAGE_CODE_3`                | `voyage-code-3`           |
| `model::VOYAGE_CODE_2`                | `voyage-code-2`           |
| `model::VOYAGE_FINANCE_2`             | `voyage-finance-2`        |
| `model::VOYAGE_MULTILINGUAL_2`        | `voyage-multilingual-2`   |
| `model::VOYAGE_LAW_2`                 | `voyage-law-2`            |
| `model::VOYAGE_MULTIMODAL_3`          | `voyage-multimodal-3`     |
| `model::VOYAGE_MULTIMODAL_3_5`        | `voyage-multimodal-3.5`   |
| `model::VOYAGE_LARGE_2`               | `voyage-large-2`          |
| `model::VOYAGE_LARGE_2_INSTRUCT`      | `voyage-large-2-instruct` |
| `model::VOYAGE_LITE_02_INSTRUCT`      | `voyage-lite-02-instruct` |
| `model::VOYAGE_LITE_01`               | `voyage-lite-01`          |
| `model::VOYAGE_LITE_01_INSTRUCT`      | `voyage-lite-01-instruct` |
| `model::VOYAGE_2`                     | `voyage-2`                |
| `model::VOYAGE_01`                    | `voyage-01`               |

### Reranking Models

| Constant                         | Value                   |
|----------------------------------|-------------------------|
| `model::RERANK_2_5` (default)    | `rerank-2.5`            |
| `model::RERANK_2_5_LITE`         | `rerank-2.5-lite`       |
| `model::RERANK_2`                | `rerank-2`              |
| `model::RERANK_2_LITE`           | `rerank-2-lite`         |
| `model::RERANK_1`                | `rerank-1`              |
| `model::RERANK_1_LITE`           | `rerank-lite-1`         |

## Error Handling

All fallible operations return `Result<T, mongodb_voyageai::Error>`:

```rust
use mongodb_voyageai::Error;

match client.embed(vec!["hello".into()]).send().await {
    Ok(embed) => println!("Got {} embeddings", embed.embeddings.len()),
    Err(Error::MissingApiKey) => eprintln!("Set VOYAGEAI_API_KEY"),
    Err(Error::RequestError { status, body }) => eprintln!("HTTP {status}: {body}"),
    Err(Error::Http(e)) => eprintln!("Network error: {e}"),
    Err(Error::Json(e)) => eprintln!("Parse error: {e}"),
}
```

| Variant        | Description                                      |
|----------------|--------------------------------------------------|
| `MissingApiKey`| No API key provided and `VOYAGEAI_API_KEY` unset |
| `RequestError` | Non-2xx HTTP response (includes status and body) |
| `Http`         | Network / connection level error (from `reqwest`) |
| `Json`         | JSON serialization / deserialization error        |

## Examples

Four runnable examples are included in the [examples/](examples/) directory. Each one requires a valid API key:

```bash
export VOYAGEAI_API_KEY="pa-..."
```

### Semantic Search

Embeds a set of documents, finds the 4 nearest neighbors by Euclidean distance, then reranks the top 2 with the rerank API.

```bash
cargo run --example search
```

### Single Embedding

Demonstrates generating an embedding for a single text and querying with a specific model (`voyage-3`).

```bash
cargo run --example embed-single
```
### Rerank Documents

Shows full ranking, top-k filtering, and model comparison (`rerank-2` vs `rerank-2-lite`).

```bash
cargo run --example rerank-documents
```
### Topic Classification

Zero-shot topic classification using cosine similarity between text embeddings and topic label embeddings.

```bash
cargo run --example classify-topics
```

### Running All Examples

```bash
export VOYAGEAI_API_KEY="pa-..."
cargo run --example search
cargo run --example embed-single
cargo run --example rerank-documents
cargo run --example classify-topics
```

## Project Structure

```
voyageai-rust/
  src/
    lib.rs          — Crate root, public re-exports, rustdoc
    client.rs       — Async HTTP client (embed + rerank) + unit tests
    config.rs       — Configuration (env vars, timeouts) + unit tests
    model.rs        — Model name constants + unit tests
    embed.rs        — Embedding response type + unit tests
    rerank.rs       — Rerank response type + unit tests
    reranking.rs    — Individual reranking result + unit tests
    usage.rs        — Token usage type + unit tests
  tests/
    client-embed.rs     — Integration tests: embed endpoint (mockito)
    client-rerank.rs    — Integration tests: rerank endpoint (mockito)
  benches/
    benchmarks.rs       — Criterion benchmarks (parsing, serialization, HTTP)
  examples/
    search.rs           — Semantic search (embed + rerank)
    embed-single.rs     — Single & query embedding
    rerank-documents.rs — Reranking with top-k and model comparison
    classify-topics.rs  — Zero-shot topic classification
```

## Dependencies

| Crate        | Version  | Purpose                               |
|--------------|----------|---------------------------------------|
| `reqwest`    | 0.13     | Async HTTP client (rustls TLS)        |
| `serde`      | 1        | Serialization / deserialization        |
| `serde_json` | 1        | JSON parsing                          |
| `thiserror`  | 2        | Ergonomic error types                 |
| `tokio`      | 1        | Async runtime                         |
| `tracing`    | 0.1      | Structured logging                    |
| `criterion`  | 0.5      | Benchmarking (dev only)               |
| `mockito`    | 1        | HTTP mocking (dev only)               |

## Testing

```bash
cargo test
```

The test suite has **105 tests** across three categories — no API key required:

| Category | Location | Tests | Description |
|----------|----------|-------|-------------|
| Unit tests | `src/*.rs` | 62 | Inline `#[cfg(test)]` modules testing parsing, serialization, display, clone, error messages |
| Integration tests | `tests/` | 13 | Async HTTP round-trips against a mock server (mockito) |
| Doc-tests | `src/*.rs` | 30 | Runnable code examples embedded in rustdoc comments |

## Benchmarks

```bash
cargo bench
```

Criterion benchmarks covering JSON parsing (1–100 embeddings/results), payload serialization, client construction, and full HTTP round-trips. Results are saved to `target/criterion/` with HTML reports.

## Documentation

Full API docs are generated with rustdoc. Every public item is documented with examples:

```bash
cargo doc --open
```

Or browse online at [docs.rs/mongodb-voyageai](https://docs.rs/mongodb-voyageai/latest/mongodb_voyageai/).

## License

Released under the [MIT License](LICENSE.txt).

## Acknowledgements

This crate is a Rust port of the [voyageai](https://github.com/ksylvest/voyageai) Ruby gem by [Kevin Sylvestre](https://github.com/ksylvest).
