# Contributing

Thank you for your interest in contributing to `voyageai`! This document explains how to get started.

## Getting Started

### Prerequisites

- **Rust 1.85+** (edition 2024)
- A VoyageAI API key (only needed for running examples, not for tests)

### Setup

```bash
git clone https://github.com/YOUR_USERNAME/voyageai-rust.git
cd voyageai-rust
cargo build
cargo test
```

## Development Workflow

### 1. Fork and Branch

```bash
git checkout -b your-feature-name
```

Use a descriptive branch name, e.g. `fix-timeout-handling` or `add-batch-rerank`.

### 2. Make Your Changes

- Write code in `src/`
- Add **unit tests** as `#[cfg(test)] mod tests` inside each source file
- Add **integration tests** in `tests/` for HTTP-level behavior (using mockito)
- Add **doc examples** (`///` comments with ` ```rust `) on all new public items

### 3. Verify

Run the full check suite before submitting:

```bash
# Format
cargo fmt --check

# Lint
cargo clippy --all-targets

# Tests (unit + integration + doc-tests)
cargo test

# Docs build without warnings
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps

# Benchmarks compile
cargo bench --no-run
```

### 4. Commit

- Write clear, concise commit messages
- Use the imperative mood: "Add batch embed support", not "Added batch embed support"
- Keep commits focused — one logical change per commit

### 5. Submit a Pull Request

- Open a PR against `main`
- Fill in the PR template (if provided)
- Describe **what** changed and **why**
- Link any related issues

## Code Style

### Formatting

This project uses `rustfmt` with default settings. Run `cargo fmt` before committing.

### Naming

| Item | Convention | Example |
|------|-----------|---------|
| Files (source) | `snake_case` | `embed.rs` |
| Files (examples, benches, tests) | `kebab-case` | `embed-single.rs` |
| Types, traits | `PascalCase` | `EmbedInput` |
| Functions, methods, variables | `snake_case` | `with_api_key` |
| Constants | `SCREAMING_SNAKE_CASE` | `VOYAGE_3_5` |

### Documentation

- Every `pub` item must have a `///` doc comment
- Include at least one `# Examples` section with a runnable code block
- Use `# Errors` sections on fallible functions
- Keep descriptions concise — lead with what the item does

### Testing

| Type | Location | Purpose |
|------|----------|---------|
| Unit tests | `src/<module>.rs` → `#[cfg(test)] mod tests` | Test parsing, serialization, display, edge cases |
| Integration tests | `tests/*.rs` | Test full HTTP round-trips with mockito |
| Doc-tests | `///` comments | Verify documentation examples compile and run |
| Benchmarks | `benches/benchmarks.rs` | Performance regression tracking |

### Error Handling

- Use `Result<T, Error>` — never panic in library code
- Surface errors with `thiserror` derive macros
- Do not use `.unwrap()` or `.expect()` outside of tests

## What to Contribute

### Good First Issues

- Improve test coverage for edge cases
- Add more doc examples
- Fix typos in documentation

### Feature Ideas

- Retry logic with exponential backoff
- Streaming / batched embedding for large inputs
- Builder pattern for `EmbedInput` / `RerankInput`
- `tracing` span instrumentation for requests

### Not in Scope

- Features that require `unsafe` code
- OpenSSL / non-rustls TLS backends
- Blocking (non-async) API — use `tokio::runtime::Runtime::block_on` if needed

## Reporting Bugs

1. Check existing [issues](https://github.com/YOUR_USERNAME/voyageai-rust/issues) first
2. Include: Rust version (`rustc --version`), OS, minimal reproduction code
3. Include the full error message and backtrace if applicable

## Security

If you find a security vulnerability, **do not open a public issue**. See [SECURITY.md](SECURITY.md) for responsible disclosure instructions.

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE.txt).
