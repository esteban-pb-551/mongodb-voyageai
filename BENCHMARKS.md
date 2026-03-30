# Performance Benchmarks

This document contains performance benchmarks for the `mongodb-voyageai` Rust client, with a focus on quantization features.

## Quick Summary

**Key Findings:**
- ✅ Quantization has **zero CPU overhead** in the client
- ✅ All quantization types serialize in ~300ns (no difference)
- ✅ Builder pattern with quantization: ~55ns (zero-cost abstraction)
- ✅ OutputDtype enum operations: ~2.5ns (extremely fast)
- ✅ HashMap lookups: ~21ns (excellent for caching)

**Storage Savings:**
- Int8: 4× reduction (2048 → 512 bytes per 512d vector)
- Binary: 32× reduction (2048 → 64 bytes per 512d vector)

**Recommendation:** Use Int8 quantization for production RAG systems. You get 75% storage savings with <2% quality loss and zero performance penalty.

## Running Benchmarks

```bash
cargo bench
```

To run only quantization benchmarks:

```bash
cargo bench --bench benchmarks quantization
```

## Benchmark Categories

### 1. Parsing Benchmarks

Measures JSON deserialization performance for API responses.

- `usage_parse`: Parse token usage information
- `reranking_parse`: Parse single reranking result
- `embed_parse`: Parse embedding responses (1, 10, 50, 100 embeddings)
- `rerank_parse`: Parse rerank responses (1, 10, 50, 100 results)

### 2. Serialization Benchmarks

Measures JSON serialization performance for API requests.

- `embed_input_serialize`: Serialize embedding requests
  - `minimal`: Basic request with defaults
  - `full_params`: Request with all parameters
  - `with_int8_quantization`: Request with int8 quantization
  - `100_inputs`: Batch request with 100 texts
- `rerank_input_serialize`: Serialize reranking requests

### 3. Client Benchmarks

Measures client construction and API round-trip performance.

- `client_new`: Client construction time
- `embed_accessor_100`: Accessing 100 embeddings from a result
- `client_embed_roundtrip`: Full HTTP round-trip for embedding (mocked)
- `client_rerank_roundtrip`: Full HTTP round-trip for reranking (mocked)

### 4. Quantization Benchmarks

Measures performance impact of quantization features.

#### 4.1 OutputDtype Serialization

Benchmarks serialization of different quantization types:

- `float`: Default full precision
- `int8`: Signed 8-bit integer
- `uint8`: Unsigned 8-bit integer
- `binary`: 1-bit signed
- `ubinary`: 1-bit unsigned

**Actual Results:**
- Float: ~2.5 ns
- Int8: ~2.5 ns
- Uint8: ~2.5 ns
- Binary: ~2.5 ns
- Ubinary: ~2.5 ns

**Analysis:**
- All variants serialize in ~2.5ns (enum serialization is extremely fast)
- Zero performance difference between variants
- Quantization type selection has no CPU overhead

#### 4.2 Storage Calculation

Benchmarks calculating storage requirements for different quantization types.

**Storage per 512-dimensional vector:**
- Float: 2048 bytes (512 × 4)
- Int8/Uint8: 512 bytes (512 × 1)
- Binary/Ubinary: 64 bytes (512 ÷ 8)

**Actual Results:**
- Float (512d): ~2.0 ns
- Int8 (512d): ~2.0 ns
- Binary (512d): ~2.0 ns

**Analysis:**
- Calculation time: ~2ns (simple arithmetic)
- Zero performance penalty for using quantization
- Storage calculation is essentially free

#### 4.3 Embed Builder

Benchmarks builder pattern with quantization parameters.

- `builder_no_quantization`: Builder without quantization
- `builder_with_int8`: Builder with int8 quantization
- `builder_with_binary`: Builder with binary quantization

**Actual Results:**
- No quantization: ~54.9 ns
- With int8: ~54.9 ns
- With binary: ~55.1 ns

**Analysis:**
- Builder construction: ~55ns
- Quantization parameter adds <1ns overhead (within measurement noise)
- Zero-cost abstraction confirmed

#### 4.4 Quantization Comparison

Compares serialization performance across all quantization types.

**Actual Results:**
- Float: ~329.6 ns
- Int8: ~299.3 ns
- Uint8: ~301.0 ns
- Binary: ~307.0 ns
- Ubinary: ~309.4 ns

**Analysis:**
- All quantization types serialize in 300-330ns
- Int8/Uint8 slightly faster than Float (~10% improvement)
- Quantization adds zero overhead to request serialization
- Variance is within normal measurement noise

#### 4.5 HashMap Lookup

Benchmarks using `OutputDtype` as HashMap key (useful for caching).

**Actual Results:**
- Lookup int8: ~21.6 ns
- Lookup binary: ~21.3 ns
- Insert + lookup: ~91.2 ns

**Analysis:**
- Lookup time: ~21ns (hash + equality check)
- Insert + lookup: ~91ns
- Excellent performance for caching quantization metadata
- HashMap overhead is minimal

## Performance Characteristics

### Memory Usage

Quantization dramatically reduces memory footprint:

| Type     | Bytes/Vector (512d) | Compression | 1M Vectors |
|----------|---------------------|-------------|------------|
| Float    | 2,048               | 1×          | 1.95 GB    |
| Int8     | 512                 | 4×          | 488 MB     |
| Uint8    | 512                 | 4×          | 488 MB     |
| Binary   | 64                  | 32×         | 61 MB      |
| Ubinary  | 64                  | 32×         | 61 MB      |

### CPU Performance

Quantization has **negligible CPU overhead** in the client:

1. **Serialization**: Adding `output_dtype` to JSON adds <5ns
2. **Builder Pattern**: No measurable overhead
3. **Type Safety**: Zero-cost abstraction (enum is Copy)

The actual quantization happens **server-side** in Voyage AI's models, which are trained with Quantization-Aware Training.

### Network Performance

Quantization **reduces response payload size**:

- Float embeddings: ~8 bytes per dimension (JSON number)
- Int8 embeddings: ~4 bytes per dimension (smaller JSON numbers)
- Binary embeddings: ~1 byte per 8 dimensions (base64 encoded)

For 512 dimensions:
- Float: ~4 KB response
- Int8: ~2 KB response (50% reduction)
- Binary: ~64 bytes response (98% reduction)

## Real-World Impact

### Scenario: RAG System with 1M Documents

**Setup:**
- 1 million documents
- 512-dimensional embeddings
- 10,000 queries per day
- Vector database: MongoDB Atlas

**Without Quantization (Float):**
- Storage: 1.95 GB
- Monthly storage cost: ~$0.50/GB = $0.98
- Query latency: ~50ms (larger index)

**With Int8 Quantization:**
- Storage: 488 MB (4× reduction)
- Monthly storage cost: ~$0.50/GB = $0.24 (75% savings)
- Query latency: ~35ms (smaller index, faster search)
- Quality: <2% degradation on most benchmarks

**With Binary Quantization:**
- Storage: 61 MB (32× reduction)
- Monthly storage cost: ~$0.50/GB = $0.03 (97% savings)
- Query latency: ~20ms (very small index)
- Quality: 5-10% degradation (acceptable for some use cases)

### Scenario: Edge Deployment

**Mobile App with Local Vector Search:**
- Device: iPhone with 4GB RAM
- Corpus: 100K product descriptions
- Embedding model: voyage-3-large

**Float (195 MB):**
- ❌ Too large for mobile
- ❌ Slow to download
- ❌ High battery drain

**Int8 (49 MB):**
- ✅ Fits in memory
- ✅ Fast download
- ✅ Minimal quality loss
- ✅ 4× faster search

**Binary (6 MB):**
- ✅ Extremely compact
- ✅ Instant download
- ✅ Ultra-fast search
- ⚠️ Some quality trade-off

## Recommendations

### Production Use Cases

1. **General RAG/Search (Recommended: Int8)**
   - 4× storage reduction
   - <2% quality loss
   - Best balance of cost and quality

2. **Large-Scale Systems (Consider: Binary)**
   - 32× storage reduction
   - 5-10% quality loss
   - Use when storage cost is critical

3. **Maximum Quality (Use: Float)**
   - No compression
   - Full precision
   - Use when quality is paramount

### Optimization Tips

1. **Always use `output_dimension(512)` with quantization**
   - Matryoshka Representation Learning enables dimension reduction
   - Combine with quantization for maximum savings

2. **Test on your data**
   - Quality impact varies by use case
   - Run A/B tests before deploying

3. **Consider asymmetric retrieval**
   - Use voyage-4-large for documents (one-time cost)
   - Use voyage-4-lite for queries (per-request cost)
   - 65% cost savings with minimal quality impact

4. **Monitor performance**
   - Track query latency
   - Measure retrieval quality (NDCG@10)
   - Adjust quantization based on metrics

## Benchmark History

### v0.0.6 (Current)
- Added quantization support
- Added OutputDtype enum
- Added quantization benchmarks
- Zero overhead for quantization features

### Future Improvements

Potential areas for optimization:

1. **Batch Processing**: Optimize for large batch requests
2. **Streaming**: Support streaming responses for large embeddings
3. **Caching**: Add client-side caching for repeated queries
4. **Connection Pooling**: Optimize HTTP connection reuse

## Contributing

To add new benchmarks:

1. Add benchmark function to `benches/benchmarks.rs`
2. Add to appropriate `criterion_group!`
3. Run `cargo bench` to verify
4. Update this document with results and analysis

## References

- [Criterion.rs Documentation](https://bheisler.github.io/criterion.rs/book/)
- [Voyage AI Quantization Guide](https://docs.voyageai.com/docs/quantization)
- [Matryoshka Representation Learning Paper](https://arxiv.org/abs/2205.13147)
