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
mongodb-voyageai = "0.1.3"
tokio = { version = "1", features = ["full"] }
```

Requires **Rust edition 2024** (rustc 1.85+).

## Quick Start

```rust
use mongodb_voyageai::{Client, model};

#[tokio::main]
async fn main() -> Result<(), mongodb_voyageai::Error> {
    // Reads VOYAGEAI_API_KEY from the environment by default
    let client = Client::from_env();

    // Generate an embedding
    let embed = client
        .embed("A quick brown fox jumps over the lazy dog.")
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
use mongodb_voyageai::Client;

#[tokio::main]
async fn main() -> Result<(), mongodb_voyageai::Error> {
    // Reads VOYAGEAI_API_KEY from the environment by default
    let client = Client::from_env();

    let input = vec![
        "John is a musician.",
        "Paul is a plumber.",
        "George is a teacher.",
        "Ringo is a doctor.",
    ];

    let embed = client
        .embed(&input)
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

| Parameter          | Type              | Default              | Description                            |
|--------------------|-------------------|----------------------|----------------------------------------|
| `embed`            | `Vec<String>`     | *required*           | Texts to embed                         |
| `model`            | `Option<model>`   | `VOYAGE_4`           | Model identifier                       |
| `input_type`       | `Option<&str>`    | `None`               | `"query"` or `"document"`              |
| `truncation`       | `Option<bool>`    | `None`               | Truncate inputs exceeding context       |
| `output_dimension` | `Option<u32>`     | `None`               | Reduce embedding dimensionality         |
| `output_dtype`     | `Option<OutputDtype>` | `None` (float)   | Quantization type (int8, uint8, binary, ubinary) |

### Contextualized Chunk Embeddings

Voyage AI provides contextualized chunk embeddings that maintain document context when embedding chunks. This is particularly useful for RAG applications where chunks need to preserve their relationship to the parent document.

```rust
use mongodb_voyageai::Client;

#[tokio::main]
async fn main() -> Result<(), mongodb_voyageai::Error> {
    let client = Client::from_env();

    // Document chunks - each inner list contains chunks from one document
    let documents = vec![
        vec![
            "text_1_1",
            "text_1_2",
        ],
        vec![
            "text_2_1",
            "text_2_2",
        ],
    ];

    let embed = client
        .contextualized_embed(&documents)
        .model("voyage-context-3")
        .input_type("document")
        .send()
        .await?;

    println!("Model: {}", embed.model);
    println!("Documents: {}", embed.results.len());
    
    for (i, result) in embed.results.iter().enumerate() {
        println!(
            "  Document {}: {} chunks",
            i,
            result.embeddings().len()
        );
    }

    Ok(())
}
```

#### Contextualized Embed Parameters

| Parameter          | Type              | Default              | Description                            |
|--------------------|-------------------|----------------------|----------------------------------------|
| `inputs`           | `Vec<Vec<String>>`| *required*           | List of lists of texts to embed        |
| `model`            | `Option<&str>`    | `"voyage-context-3"` | Model identifier                       |
| `input_type`       | `Option<&str>`    | `None`               | `"query"` or `"document"`              |
| `output_dimension` | `Option<u32>`     | `None`               | Reduce embedding dimensionality         |
| `output_dtype`     | `Option<OutputDtype>` | `None` (float)   | Quantization type (int8, uint8, binary, ubinary) |

**Note:** For queries, each inner list should contain a single query. For documents, each inner list typically contains chunks from a single document ordered by their position.

### Quantization (Storage Optimization)

Voyage AI models with Quantization-Aware Training support multiple output formats that dramatically reduce storage costs:

```rust
use mongodb_voyageai::{Client, model, OutputDtype};

#[tokio::main]
async fn main() -> Result<(), mongodb_voyageai::Error> {
    // Reads VOYAGEAI_API_KEY from the environment by default
    let client = Client::from_env();

    // 4× storage reduction with minimal quality loss
    let embed = client
        .embed(&["Efficient storage"])
        .model(model::VOYAGE_3_LARGE)
        .output_dimension(512)
        .output_dtype(OutputDtype::Int8)
        .send()
        .await?;

    // 32× compression for maximum efficiency
    let embed_binary = client
        .embed(&["Ultra compact"])
        .model(model::VOYAGE_4_LARGE)
        .output_dimension(512)
        .output_dtype(OutputDtype::Binary)
        .send()
        .await?;

    Ok(())
}
```

#### Storage Comparison (512 dimensions)

| Type     | Bytes | Compression | Use Case                          |
|----------|-------|-------------|-----------------------------------|
| `Float`  | 2048  | 1×          | Maximum precision required        |
| `Int8`   | 512   | 4×          | Production (minimal quality loss) |
| `Uint8`  | 512   | 4×          | Production (minimal quality loss) |
| `Binary` | 64    | 32×         | Large-scale (storage critical)    |
| `Ubinary`| 64    | 32×         | Large-scale (storage critical)    |

According to Voyage AI benchmarks, `voyage-3-large` with `int8` at 512 dimensions outperforms OpenAI-v3-large by 8.56% while using only 1/24 the storage.

For detailed performance benchmarks and optimization guidelines, see:
- [PERFORMANCE.md](PERFORMANCE.md) - Quick performance reference
- [BENCHMARKS.md](BENCHMARKS.md) - Detailed benchmark results

### Reranking

```rust
use mongodb_voyageai::Client;

#[tokio::main]
async fn main() -> Result<(), mongodb_voyageai::Error> {
    // Reads VOYAGEAI_API_KEY from the environment by default
    let client = Client::from_env();

    let query = "Who should I call if my pipes are leaking?";

    let documents = vec![
        "John is a musician.",
        "Paul is a plumber.",
        "George is a teacher.",
        "Ringo is a doctor.",
    ];

    let rerank = client
        .rerank(
            query,
            &documents
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
| `documents`     | `Vec<String>`   | *required*     | Documents to rerank (max 1,000)      |
| `model`         | `Option<&str>`  | `"rerank-2.5"` | Model identifier                     |
| `top_k`         | `Option<u32>`   | `None`         | Return only the top K results        |
| `truncation`    | `Option<bool>`  | `true` (API)   | Truncate inputs exceeding context    |

#### Rerank Model Limits

| Model           | Context Length | Query Tokens | Query+Doc Tokens | Total Tokens |
|-----------------|----------------|--------------|------------------|--------------|
| rerank-2.5      | 32,000         | 8,000        | 32,000           | 600K         |
| rerank-2.5-lite | 32,000         | 8,000        | 32,000           | 600K         |
| rerank-2        | 16,000         | 4,000        | 16,000           | 600K         |
| rerank-2-lite   | 8,000          | 2,000        | 8,000            | 600K         |
| rerank-1        | 8,000          | 2,000        | 8,000            | 300K         |
| rerank-lite-1   | 4,000          | 1,000        | 4,000            | 300K         |

**Note:** Total tokens = (query tokens × num documents) + sum of all document tokens

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

**Recommended:** Use `rerank-2.5` for best quality or `rerank-2.5-lite` for balanced latency/quality. Both support 32K context length and instruction-following.

## Note for Linux Installation

This crate uses `native-tls`, which depends on OpenSSL on Linux. Before building,
install the required system package for your distribution:

**Debian / Ubuntu**
```bash
sudo apt update && sudo apt install libssl-dev pkg-config
```

**Fedora / RHEL / CentOS**
```bash
sudo dnf install openssl-devel pkg-config
```

**Arch Linux**
```bash
sudo pacman -S openssl pkg-config
```

**Alpine Linux**
```bash
sudo apk add openssl-dev pkgconfig
```

> `pkg-config` is required so the Rust build system can locate the OpenSSL
> headers and libraries on your system.

No additional setup is needed on **Windows** (uses SChannel) or **macOS**
(uses Secure Transport), as both platforms provide a native TLS implementation
out of the box.

## Error Handling

All fallible operations return `Result<T, mongodb_voyageai::Error>`:

```rust
use mongodb_voyageai::Error;

match client.embed("hello").send().await {
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

Ten runnable examples are included in the [examples/](examples/) directory. Each one requires a valid API key:

```bash
export VOYAGEAI_API_KEY="pa-..."
```

### Contextualized Embeddings

Demonstrates how to use contextualized chunk embeddings to maintain document context when embedding chunks.

```bash
cargo run --example contextualized-embeddings
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

### Quantization

Demonstrates storage optimization using different quantization types (float, int8, uint8, binary, ubinary).

```bash
cargo run --example quantization
```

### Compare Quantization

Compares embedding quality across different quantization types to help choose the right trade-off.

```bash
cargo run --example compare-quantization
```

### RAG with Quantization

Production-ready RAG pipeline using int8 quantization for 4× storage reduction.

```bash
cargo run --example rag-with-quantization
```

### Asymmetric Retrieval

Demonstrates cost optimization by using different models for documents (voyage-4-large) vs queries (voyage-4-lite).

```bash
cargo run --example asymmetric-retrieval
```

## Documentation

Full API docs are generated with rustdoc. Every public item is documented with examples:

```bash
cargo doc --open
```

Or browse online at [docs.rs/mongodb-voyageai](https://docs.rs/mongodb-voyageai/latest/mongodb_voyageai/).

### Additional Resources

- [QUANTIZATION_GUIDE.md](QUANTIZATION_GUIDE.md) - Complete guide to using quantization
- [PERFORMANCE.md](PERFORMANCE.md) - Quick performance reference and optimization tips
- [BENCHMARKS.md](BENCHMARKS.md) - Detailed benchmark results and analysis

## License

Released under the [MIT License](LICENSE.txt).

## Acknowledgements

This crate is a Rust port of the [voyageai](https://github.com/ksylvest/voyageai) Ruby gem by [Kevin Sylvestre](https://github.com/ksylvest).
