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
mongodb-voyageai = "0.0.1"
tokio = { version = "1", features = ["full"] }
```

Requires **Rust edition 2024** (rustc 1.85+).

## Quick Start

```rust
use mongodb_voyageai::{Client, Config};

#[tokio::main]
async fn main() -> Result<(), mongodb_voyageai::Error> {
    // Reads VOYAGEAI_API_KEY from the environment by default
    let client = Client::new(&Config::new())?;

    // Generate an embedding
    let embed = client
        .embed(vec!["A quick brown fox jumps over the lazy dog.".into()], None, None, None, None)
        .await?;

    println!("model: {}", embed.model);
    println!("tokens: {}", embed.usage.total_tokens);
    println!("embedding: {:?}", embed.embedding(0));

    Ok(())
}
```

## Usage

### Embeddings

#### Single Embedding

```rust
use mongodb_voyageai::{Client, Config};

#[tokio::main]
async fn main() -> Result<(), mongodb_voyageai::Error> {
    let client = Client::with_api_key("pa-...")?;

    let embed = client
        .embed(vec!["A quick brown fox jumps over the lazy dog.".into()], None, None, None, None)
        .await?;

    println!("{}", embed.model);                  // "voyage-3.5"
    println!("{}", embed.usage.total_tokens);      // 11
    println!("{:?}", embed.embedding(0).unwrap()); // [0.0, ...]

    Ok(())
}
```

#### Multiple Embeddings

```rust
use mongodb_voyageai::{Client, Config};

#[tokio::main]
async fn main() -> Result<(), mongodb_voyageai::Error> {
    let client = Client::with_api_key("pa-...")?;

    let input = vec![
        "John is a musician.".into(),
        "Paul is a plumber.".into(),
        "George is a teacher.".into(),
        "Ringo is a doctor.".into(),
    ];

    let embed = client.embed(input, None, Some("document"), None, None).await?;

    println!("{}", embed.model);                // "voyage-3.5"
    println!("{}", embed.usage.total_tokens);    // total tokens used
    println!("{}", embed.embeddings.len());      // 4
    println!("{:?}", embed.embedding(0));        // Some([0.0, ...])

    Ok(())
}
```

#### Embed Parameters

| Parameter          | Type            | Default              | Description                            |
|--------------------|-----------------|----------------------|----------------------------------------|
| `input`            | `Vec<String>`   | *required*           | Texts to embed                         |
| `embed_model`      | `Option<&str>`  | `"voyage-3.5"`       | Model identifier                       |
| `input_type`       | `Option<&str>`  | `None`               | `"query"` or `"document"`              |
| `truncation`       | `Option<bool>`  | `None`               | Truncate inputs exceeding context       |
| `output_dimension` | `Option<u32>`   | `None`               | Reduce embedding dimensionality         |

### Reranking

```rust
use mongodb_voyageai::{Client, Config};

#[tokio::main]
async fn main() -> Result<(), mongodb_voyageai::Error> {
    let client = Client::with_api_key("pa-...")?;

    let query = "Who is the best person to call for a toilet?";

    let documents = vec![
        "John is a musician.".into(),
        "Paul is a plumber.".into(),
        "George is a teacher.".into(),
        "Ringo is a doctor.".into(),
    ];

    let rerank = client.rerank(query, documents, None, Some(3), None).await?;

    println!("{}", rerank.model);              // "rerank-2"
    println!("{}", rerank.usage.total_tokens); // total tokens used

    for result in &rerank.results {
        println!("index={} relevance_score={}", result.index, result.relevance_score);
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

| Constant                         | Value                   |
|----------------------------------|-------------------------|
| `model::VOYAGE` (default)        | `voyage-3.5`            |
| `model::VOYAGE_LITE`             | `voyage-3.5-lite`       |
| `model::VOYAGE_3`                | `voyage-3`              |
| `model::VOYAGE_3_LARGE`          | `voyage-3-large`        |
| `model::VOYAGE_3_LITE`           | `voyage-3-lite`         |
| `model::VOYAGE_FINANCE`          | `voyage-finance-2`      |
| `model::VOYAGE_MULTILINGUAL`     | `voyage-multilingual-2` |
| `model::VOYAGE_LAW`              | `voyage-law-2`          |
| `model::VOYAGE_CODE`             | `voyage-code-2`         |

### Reranking Models

| Constant                         | Value                   |
|----------------------------------|-------------------------|
| `model::RERANK` (default)        | `rerank-2`              |
| `model::RERANK_LITE`             | `rerank-2-lite`         |

## Error Handling

All fallible operations return `Result<T, mongodb_voyageai::Error>`:

```rust
use mongodb_voyageai::Error;

match client.embed(vec!["hello".into()], None, None, None, None).await {
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

```
query="What do George and Ringo do?"
document="Ringo is a doctor." relevance_score=0.67578125
document="George is a teacher." relevance_score=0.5859375

query="Who works in the medical field?"
document="Bill is a nurse." relevance_score=0.55078125
document="Ringo is a doctor." relevance_score=0.50390625
```

### Single Embedding

Demonstrates generating an embedding for a single text and querying with a specific model (`voyage-3`).

```bash
cargo run --example embed-single
```

```
Model: voyage-3.5
Tokens used: 13
Dimensions: 1024
First 5 values: [0.054095149, 0.034528818, -0.008440378, 0.026088441, 0.038365357]

Query embedding model: voyage-3
Query embedding dimensions: 1024
```

### Rerank Documents

Shows full ranking, top-k filtering, and model comparison (`rerank-2` vs `rerank-2-lite`).

```bash
cargo run --example rerank-documents
```

```
=== Full ranking ===
Query: "Who should I call if my pipes are leaking?"
Model: rerank-2
Tokens used: 98

  [1] score=0.5508  "Paul is a licensed plumber with 15 years of experience."
  [3] score=0.4863  "Ringo is a doctor specializing in cardiology."
  ...

=== Top 2 only ===
  [1] score=0.5508  "Paul is a licensed plumber with 15 years of experience."
  ...

=== Using rerank-2-lite ===
Model: rerank-2-lite
  [1] score=0.5352  "Paul is a licensed plumber with 15 years of experience."
  ...
```

### Topic Classification

Zero-shot topic classification using cosine similarity between text embeddings and topic label embeddings.

```bash
cargo run --example classify-topics
```

```
Tokens used: 97

[0.6481] Rust's borrow checker prevents data races at compile time. => Technology and programming
[0.6546] The sourdough bread needs to proof for at least 12 hours. => Cooking and food
[0.7451] She finished the marathon in under three hours.           => Sports and athletics
[0.7621] The new album features a blend of jazz and electronic music. => Music and entertainment
[0.7175] The researchers published their findings on CRISPR gene editing. => Science and research
[0.7398] Python is great for machine learning prototyping.         => Technology and programming
[0.6778] Add a pinch of saffron to the risotto for extra flavor.   => Cooking and food
[0.6927] The goalkeeper made an incredible save in the final minute. => Sports and athletics
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

Or browse online at [docs.rs/voyageai](https://docs.rs/voyageai).

## License

Released under the [MIT License](LICENSE.txt).

## Acknowledgements

This crate is a Rust port of the [voyageai](https://github.com/ksylvest/voyageai) Ruby gem by [Kevin Sylvestre](https://github.com/ksylvest).
