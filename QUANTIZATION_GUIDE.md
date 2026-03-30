# Quantization Guide

Complete guide to using quantization with `mongodb-voyageai`.

## Table of Contents

1. [What is Quantization?](#what-is-quantization)
2. [Quick Start](#quick-start)
3. [Quantization Types](#quantization-types)
4. [Performance Impact](#performance-impact)
5. [Best Practices](#best-practices)
6. [Examples](#examples)
7. [Troubleshooting](#troubleshooting)

## What is Quantization?

Quantization reduces the precision of embedding vectors to save storage space while maintaining quality. Voyage AI models are trained with **Quantization-Aware Training**, meaning they're optimized to work well with reduced precision.

### How It Works

1. **Server-side**: Voyage AI generates embeddings with quantization built-in
2. **Client-side**: You specify the desired quantization type
3. **Storage**: Vectors are stored in compressed format
4. **Search**: Vector databases work with compressed vectors directly

### Benefits

- 💾 **4-32× storage reduction**
- ⚡ **Faster vector search** (smaller indexes)
- 💰 **Lower infrastructure costs**
- 🎯 **Minimal quality loss** (<2% with Int8)

## Quick Start

```rust
use mongodb_voyageai::{Client, Config, model, OutputDtype};

#[tokio::main]
async fn main() -> Result<(), mongodb_voyageai::Error> {
    let client = Client::new(&Config::new())?;

    // Embed with Int8 quantization (RECOMMENDED)
    let embed = client
        .embed(&["Your text here"])
        .model(model::VOYAGE_3_LARGE)
        .output_dimension(512)
        .output_dtype(OutputDtype::Int8)  // ← 4× storage reduction
        .send()
        .await?;

    println!("Dimensions: {}", embed.embedding(0).unwrap().len());
    println!("Storage: 512 bytes (vs 2048 for float)");

    Ok(())
}
```

## Quantization Types

### Float (Default)

```rust
// No output_dtype specified = Float
let embed = client
    .embed(&texts)
    .send()
    .await?;
```

- **Storage**: 4 bytes per dimension
- **Quality**: 100% (baseline)
- **Use when**: Maximum precision required

### Int8 (Recommended)

```rust
.output_dtype(OutputDtype::Int8)
```

- **Storage**: 1 byte per dimension (4× reduction)
- **Quality**: 98-99% (minimal loss)
- **Use when**: Production RAG systems
- **Best for**: General purpose applications

### Uint8

```rust
.output_dtype(OutputDtype::Uint8)
```

- **Storage**: 1 byte per dimension (4× reduction)
- **Quality**: 98-99% (minimal loss)
- **Use when**: Unsigned values preferred
- **Similar to**: Int8 in most cases

### Binary

```rust
.output_dtype(OutputDtype::Binary)
```

- **Storage**: 1 bit per dimension (32× reduction)
- **Quality**: 90-95% (noticeable loss)
- **Use when**: Storage cost is critical
- **Best for**: Large-scale systems (100M+ vectors)

### Ubinary

```rust
.output_dtype(OutputDtype::Ubinary)
```

- **Storage**: 1 bit per dimension (32× reduction)
- **Quality**: 90-95% (noticeable loss)
- **Use when**: Unsigned binary preferred
- **Similar to**: Binary in most cases

## Performance Impact

### CPU Overhead: ZERO ✅

Quantization has no measurable CPU overhead in the client:

```
Operation                    Time
────────────────────────────────────
OutputDtype serialization    2.5 ns
Builder with quantization    55 ns
Request serialization        300 ns
```

### Storage Savings

For 512-dimensional vectors:

| Type    | Bytes | Compression | 1M Vectors |
|---------|-------|-------------|------------|
| Float   | 2,048 | 1×          | 1.95 GB    |
| Int8    | 512   | 4×          | 488 MB     |
| Binary  | 64    | 32×         | 61 MB      |

### Quality Trade-offs

Based on Voyage AI benchmarks:

| Type   | NDCG@10 | Quality Loss |
|--------|---------|--------------|
| Float  | 100%    | 0%           |
| Int8   | 98-99%  | 1-2%         |
| Binary | 90-95%  | 5-10%        |

## Best Practices

### 1. Always Combine with Dimension Reduction

```rust
// ✅ BEST: 16× total compression
.output_dimension(512)      // 4× reduction (MRL)
.output_dtype(OutputDtype::Int8)  // 4× reduction (quantization)

// ❌ SUBOPTIMAL: Only 4× compression
.output_dimension(2048)
.output_dtype(OutputDtype::Int8)
```

### 2. Use Consistent Settings

```rust
// Documents and queries MUST use same settings
let doc_settings = (512, OutputDtype::Int8);
let query_settings = (512, OutputDtype::Int8);  // ← Must match!

// Embed documents
let doc_embed = client
    .embed(&documents)
    .output_dimension(doc_settings.0)
    .output_dtype(doc_settings.1)
    .send()
    .await?;

// Embed queries
let query_embed = client
    .embed(query)
    .output_dimension(query_settings.0)
    .output_dtype(query_settings.1)
    .send()
    .await?;
```

### 3. Test on Your Data

```rust
async fn test_quantization_quality(
    client: &Client,
    test_queries: &[&str],
    ground_truth: &[Vec<usize>],
) -> Result<(), Box<dyn std::error::Error>> {
    let configs = vec![
        (None, "Float"),
        (Some(OutputDtype::Int8), "Int8"),
        (Some(OutputDtype::Binary), "Binary"),
    ];

    for (dtype, name) in configs {
        let mut builder = client
            .embed(test_queries)
            .output_dimension(512);
        
        if let Some(dt) = dtype {
            builder = builder.output_dtype(dt);
        }
        
        let embed = builder.send().await?;
        let ndcg = calculate_ndcg(&embed, ground_truth);
        
        println!("{}: NDCG@10 = {:.4}", name, ndcg);
    }

    Ok(())
}
```

### 4. Monitor in Production

```rust
use std::time::Instant;

// Track key metrics
struct Metrics {
    storage_bytes: usize,
    query_latency_ms: f64,
    ndcg_at_10: f64,
}

fn calculate_metrics(
    num_vectors: usize,
    dimensions: usize,
    dtype: OutputDtype,
) -> Metrics {
    let bytes_per_dim = match dtype {
        OutputDtype::Float => 4,
        OutputDtype::Int8 | OutputDtype::Uint8 => 1,
        OutputDtype::Binary | OutputDtype::Ubinary => 1,
    };
    
    let storage_bytes = num_vectors * dimensions * bytes_per_dim;
    let storage_bytes = match dtype {
        OutputDtype::Binary | OutputDtype::Ubinary => storage_bytes / 8,
        _ => storage_bytes,
    };

    Metrics {
        storage_bytes,
        query_latency_ms: 0.0,  // Measure in production
        ndcg_at_10: 0.0,        // Measure in production
    }
}
```

## Examples

### Example 1: Basic Quantization

```rust
use mongodb_voyageai::{Client, Config, model, OutputDtype};

#[tokio::main]
async fn main() -> Result<(), mongodb_voyageai::Error> {
    let client = Client::new(&Config::new())?;

    let texts = vec![
        "Rust is a systems programming language.",
        "Python is great for data science.",
    ];

    // Embed with Int8 quantization
    let embed = client
        .embed(&texts)
        .model(model::VOYAGE_3_LARGE)
        .output_dimension(512)
        .output_dtype(OutputDtype::Int8)
        .send()
        .await?;

    println!("Embedded {} texts", embed.embeddings.len());
    println!("Dimensions: {}", embed.embedding(0).unwrap().len());
    println!("Storage per vector: 512 bytes");

    Ok(())
}
```

### Example 2: RAG Pipeline

```rust
use mongodb_voyageai::{Client, Config, model, OutputDtype};

async fn build_rag_index(
    client: &Client,
    documents: &[&str],
) -> Result<Vec<Vec<f64>>, Box<dyn std::error::Error>> {
    // Embed documents with Int8 quantization
    let embed = client
        .embed(documents)
        .model(model::VOYAGE_3_LARGE)
        .input_type("document")
        .output_dimension(512)
        .output_dtype(OutputDtype::Int8)
        .send()
        .await?;

    Ok(embed.embeddings)
}

async fn search(
    client: &Client,
    query: &str,
    index: &[Vec<f64>],
) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    // Embed query with same settings
    let query_embed = client
        .embed(query)
        .model(model::VOYAGE_3_LARGE)
        .input_type("query")
        .output_dimension(512)
        .output_dtype(OutputDtype::Int8)
        .send()
        .await?;

    let query_vec = query_embed.embedding(0).unwrap();

    // Find nearest neighbors
    let mut scored: Vec<(usize, f64)> = index
        .iter()
        .enumerate()
        .map(|(i, doc_vec)| {
            let similarity = cosine_similarity(query_vec, doc_vec);
            (i, similarity)
        })
        .collect();

    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    Ok(scored.into_iter().take(10).map(|(i, _)| i).collect())
}
```

### Example 3: Asymmetric Retrieval

```rust
use mongodb_voyageai::{Client, Config, model, OutputDtype};

async fn asymmetric_retrieval(
    client: &Client,
) -> Result<(), Box<dyn std::error::Error>> {
    let documents = vec!["doc1", "doc2", "doc3"];
    let query = "search query";

    // Use expensive model for documents (one-time cost)
    let doc_embed = client
        .embed(&documents)
        .model(model::VOYAGE_4_LARGE)  // Expensive, high quality
        .input_type("document")
        .output_dimension(512)
        .output_dtype(OutputDtype::Int8)
        .send()
        .await?;

    // Use cheap model for queries (per-request cost)
    let query_embed = client
        .embed(query)
        .model(model::VOYAGE_4_LITE)   // Cheap, fast
        .input_type("query")
        .output_dimension(512)
        .output_dtype(OutputDtype::Int8)
        .send()
        .await?;

    // Voyage 4 series has shared embedding spaces!
    // Vectors from different models are compatible

    Ok(())
}
```

## Troubleshooting

### Issue: Quality Loss Too High

**Problem**: Binary quantization causes >10% quality loss.

**Solution**: Use Int8 instead:

```rust
// ❌ Too much quality loss
.output_dtype(OutputDtype::Binary)

// ✅ Better quality
.output_dtype(OutputDtype::Int8)
```

### Issue: Inconsistent Results

**Problem**: Query results don't match document embeddings.

**Solution**: Ensure consistent settings:

```rust
// ❌ WRONG: Different settings
let docs = client.embed(&docs).output_dimension(512).send().await?;
let query = client.embed(q).output_dimension(1024).send().await?;

// ✅ CORRECT: Same settings
let settings = (512, OutputDtype::Int8);
let docs = client.embed(&docs)
    .output_dimension(settings.0)
    .output_dtype(settings.1)
    .send().await?;
let query = client.embed(q)
    .output_dimension(settings.0)
    .output_dtype(settings.1)
    .send().await?;
```

### Issue: Model Not Supported

**Problem**: Quantization not working with older models.

**Solution**: Use models with Quantization-Aware Training:

```rust
// ❌ Old model (no quantization support)
.model(model::VOYAGE_2)

// ✅ New model (quantization supported)
.model(model::VOYAGE_3_LARGE)
.model(model::VOYAGE_4)
.model(model::VOYAGE_4_LARGE)
```

### Issue: Storage Not Reduced

**Problem**: Vector database size didn't decrease.

**Solution**: Ensure your database supports quantized vectors:

- MongoDB Atlas Vector Search: ✅ Supports all types
- Pinecone: ✅ Supports int8, binary
- Weaviate: ✅ Supports all types
- Qdrant: ✅ Supports all types

Check your database documentation for quantization support.

## Further Reading

- [PERFORMANCE.md](PERFORMANCE.md) - Performance benchmarks
- [BENCHMARKS.md](BENCHMARKS.md) - Detailed benchmark results
- [Voyage AI Quantization Docs](https://docs.voyageai.com/docs/quantization)
- [Matryoshka Representation Learning Paper](https://arxiv.org/abs/2205.13147)

## Support

For issues or questions:

1. Check [examples/](examples/) for working code
2. Review [PERFORMANCE.md](PERFORMANCE.md) for optimization tips
3. Open an issue on [GitHub](https://github.com/esteban-pb-551/mongodb-voyageai)
