# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/YOUR_USERNAME/voyageai-rust/compare/v0.0.1...HEAD
[0.0.1]: https://github.com/YOUR_USERNAME/voyageai-rust/releases/tag/v0.0.1
