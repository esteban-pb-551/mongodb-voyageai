# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.3] - 2026-04-04

### Changed

- **Embedding precision: `f64` → `f32`**: All embedding vectors now use `f32` instead of `f64`
  - `Embed.embeddings` type changed from `Vec<Vec<f64>>` to `Vec<Vec<f32>>`
  - `Embed.embedding()` returns `Option<&Vec<f32>>` instead of `Option<&Vec<f64>>`
  - `ContextualizedEmbeddingsResult.embeddings()` returns `Vec<Vec<f32>>`
  - `Reranking.relevance_score` changed from `f64` to `f32`
  - `pairwise::cosine_similarity` module operates on `f32` matrices
  - 50% memory reduction per embedding vector (4 bytes vs 8 bytes per dimension)
  - No precision loss — VoyageAI API returns `float32` values

### Fixed

- Fixed doc test in `normalizer::normalize` missing `use` import
- Fixed doc test in `normalizer::join_soft_line_breaks` referencing private function (marked `ignore`)
- Fixed doc test in `ContextualizedEmbedInput` using incorrect import path (`client::` → `context::`)
- Fixed doc test in `Rerank` struct running an example that requires API key (marked `no_run`)

### Documentation

- Updated `QUANTIZATION_GUIDE.md` code examples from `f64` to `f32`
- Fixed README.md example count ("Four" → "Ten" runnable examples)

[0.1.3]: https://github.com/esteban-pb-551/mongodb-voyageai/releases/tag/v0.1.3

## [0.1.2] - 2026-04-01

### Added

- **Contextualized Chunk Embeddings**: New support for Voyage AI's contextualized chunk embedding models
  - `Client::contextualized_embed()` method for generating context-aware embeddings
  - `ContextualizedEmbed` response type with `results` containing embeddings per document
  - `ContextualizedEmbeddingsResult` with `embeddings()` method to access embedding vectors
  - `ContextualizedEmbedBuilder` with fluent API for configuration:
    - `model()`: Select model (default: `voyage-context-3`)
    - `input_type()`: Specify `"query"` or `"document"`
    - `output_dimension()`: Reduce dimensionality (256, 512, 1024, 2048)
    - `output_dtype()`: Apply quantization (float, int8, uint8, binary, ubinary)
  - `IntoVecVecString` trait for flexible input types:
    - `Vec<Vec<&str>>`: Consumes the vector
    - `Vec<Vec<String>>`: Consumes the vector
    - `&Vec<Vec<&str>>`: Borrows without moving (original stays accessible)
    - `&Vec<Vec<String>>`: Borrows without moving (original stays accessible)
  - New module `context` with complete implementation
  - Example: `examples/contextualized-embeddings.rs` demonstrating document chunks, queries, and quantization
  - Comprehensive documentation: `CONTEXTUALIZED_EMBEDDINGS_GUIDE.md` with:
    - Detailed explanation of contextualized embeddings
    - When and how to use them
    - Complete RAG pipeline example
    - Comparison with standard embeddings
    - Best practices and optimization tips

### Changed

- Updated `README.md` with contextualized embeddings section and usage examples
- Enhanced `lib.rs` to export `context` module and `ContextualizedEmbed` type

### Technical Details

- Correctly handles nested API response structure with `data` arrays
- Maintains document-level context when embedding chunks
- Supports up to 1,000 inputs, 120K total tokens, and 16K total chunks
- Compatible with all quantization options for storage optimization
- Full test coverage with 5 new unit tests (176 total tests passing)

[0.1.2]: https://github.com/esteban-pb-551/mongodb-voyageai/releases/tag/v0.1.2

## [0.1.1] - 2026-03-30

### Added

- `Client::try_from_env()` method for creating a client from environment variables with error handling
- `Client::from_env()` method for creating a client from environment variables (panics if API key is not set)

### Changed

- Improved ergonomics for client creation, eliminating the need for `&Config::new()` pattern

[0.1.1]: https://github.com/esteban-pb-551/mongodb-voyageai/releases/tag/v0.1.1

## [0.1.0] - 2026-03-30

### Added

- **Quantization Support**: Added `OutputDtype` enum for embedding quantization
  - `Float`: Full precision (default)
  - `Int8`: Signed 8-bit integer (4× compression, <2% quality loss)
  - `Uint8`: Unsigned 8-bit integer (4× compression)
  - `Binary`: 1-bit signed (32× compression, 5-10% quality loss)
  - `Ubinary`: 1-bit unsigned (32× compression)
- `output_dtype()` method on `EmbedBuilder` for specifying quantization type
- `output_dtype` field in `EmbedInput` struct
- Comprehensive documentation:
  - `BENCHMARKS.md`: Detailed benchmark results and analysis
  - `PERFORMANCE.md`: Quick performance reference and optimization tips
  - `QUANTIZATION_GUIDE.md`: Complete guide to using quantization

### Changed

- `EmbedInput` struct now includes optional `output_dtype` field
- Enhanced API documentation with quantization usage examples

[0.1.0]: https://github.com/esteban-pb-551/mongodb-voyageai/releases/tag/v0.1.0

## [0.0.6] - 2026-03-26

### Changed
- `Client::embed` now accepts `&[S] where S: AsRef<str>` instead of `Vec<S>`,
  allowing callers to pass a borrowed slice without transferring ownership
- `Client::rerank` now accepts `&[S] where S: AsRef<str>` for `documents` by the same reason
- Both methods internally convert to `Vec<String>`, so the public API of
  `EmbedBuilder` and `RerankBuilder` is unchanged

[0.0.6]: https://github.com/esteban-pb-551/mongodb-voyageai/releases/tag/v0.0.6

## [0.0.5] - 2026-03-26

### Changed
- `Client::embed` and `Client::rerank` now accepts [Method Chaining](https://stackoverflow.com/questions/74965709/chaining-methods-in-rust)

[0.0.5]: https://github.com/esteban-pb-551/mongodb-voyageai/releases/tag/v0.0.5

## [0.0.4] - 2026-03-25

### Changed
- Updated default embedding model from `voyage-3.5` to `voyage-4`
- Updated default rerankking model from `rerank-2` to `rerank-2.5`

[0.0.4]: https://github.com/esteban-pb-551/mongodb-voyageai/releases/tag/v0.0.4

## [0.0.3] - 2026-03-24

### Fixed
- Refactor client setup in examples files and example outputs in README.md

[0.0.3]: https://github.com/esteban-pb-551/mongodb-voyageai/releases/tag/v0.0.3

## [0.0.2] - 2026-03-24

### Fixed
- Resolved incorrect package name in README.md file

[0.0.2]: https://github.com/esteban-pb-551/mongodb-voyageai/releases/tag/v0.0.2

## [0.0.1] - 2026-03-24

### Added

- Async client for the VoyageAI API (`Client`, `Config`)
- Embedding generation via the `/embeddings` endpoint
  - Single and batch embeddings
  - Configurable model, input type, truncation, and output dimension
- Document reranking via the `/rerank` endpoint
  - Configurable model, top-k, and truncation
- Pre-defined model constants in `mongodb_voyageai::model`
  - Embedding: `VOYAGE` (voyage-3.5), `VOYAGE_LITE`, `VOYAGE_3`, `VOYAGE_3_LARGE`, `VOYAGE_3_LITE`, `VOYAGE_FINANCE`, `VOYAGE_MULTILINGUAL`, `VOYAGE_LAW`, `VOYAGE_CODE`
  - Reranking: `RERANK` (rerank-2), `RERANK_LITE`
- Configuration from environment variables (`VOYAGEAI_API_KEY`, `VOYAGEAI_HOST`, `VOYAGEAI_VERSION`)
- Optional request timeout via `Config.timeout`
- API key masking in `Debug` output
- Typed error handling (`Error::MissingApiKey`, `Error::RequestError`, `Error::Http`, `Error::Json`)
- TLS via `rustls` (no OpenSSL dependency)
- Full rustdoc documentation with examples on every public item
- Four runnable examples: `search`, `embed-single`, `rerank-documents`, `classify-topics`
- Criterion benchmarks for parsing, serialization, client construction, and HTTP round-trips
- Comprehensive test suite (unit, integration, and doc-tests)

[0.0.1]: https://github.com/esteban-pb-551/mongodb-voyageai/releases/tag/v0.0.1
