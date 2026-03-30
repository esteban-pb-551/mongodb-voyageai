# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 0.0.x   | :white_check_mark: |

Only the latest release receives security patches. Users should always upgrade to the most recent version.

## Reporting a Vulnerability

**Please do NOT open a public GitHub issue for security vulnerabilities.**

If you discover a security vulnerability in this crate, please report it responsibly through one of the following channels:

1. **GitHub Security Advisories (preferred):** Use the [private vulnerability reporting](https://github.com/YOUR_USERNAME/voyageai-rust/security/advisories/new) feature on the repository.
2. **Email:** Send a detailed report to `security@YOUR_DOMAIN.com`.

### What to Include in Your Report

- A clear description of the vulnerability
- Steps to reproduce the issue
- The affected version(s)
- The potential impact (data leak, denial of service, remote code execution, etc.)
- A suggested fix or patch, if available

### Response Timeline

| Stage                  | Timeframe         |
|------------------------|-------------------|
| Acknowledgement        | Within 48 hours   |
| Initial assessment     | Within 5 days     |
| Patch and advisory     | Within 30 days    |
| Public disclosure      | Within 90 days    |

We follow [coordinated vulnerability disclosure](https://en.wikipedia.org/wiki/Coordinated_vulnerability_disclosure). We will work with you on timing and credit attribution before any public announcement.

## Security Model

### What This Crate Does

`voyageai` is an HTTP client library that communicates with the VoyageAI REST API over HTTPS. It:

- Sends text data to VoyageAI endpoints for embedding generation and document reranking
- Parses JSON responses into typed Rust structs
- Manages API key authentication via Bearer tokens

### Trust Boundaries

| Boundary               | Trust Level | Notes |
|------------------------|-------------|-------|
| VoyageAI API responses | External    | Parsed with `serde_json`; malformed JSON returns an error, never panics |
| User-provided input    | Caller      | Text inputs and API keys are passed through without sanitization (by design) |
| Environment variables  | Local       | `VOYAGEAI_API_KEY`, `VOYAGEAI_HOST`, `VOYAGEAI_VERSION` are read at config time |

### API Key Handling

- API keys are stored in memory as `String` values for the lifetime of the `Client`
- The `Debug` implementation masks the API key (shows only the first 5 characters followed by `***`)
- Keys are transmitted over HTTPS only (TLS enforced by `rustls`)
- Keys are never written to logs, files, or error messages by this crate

### TLS and Network Security

This crate uses **rustls** (not OpenSSL) for TLS, via the `reqwest` + `hyper-rustls` stack:

```
reqwest 0.13.2
  └── hyper-rustls 0.27.x
        └── rustls 0.23.x
              └── aws-lc-rs (cryptographic backend)
```

- **No OpenSSL dependency** — eliminates an entire class of C-library vulnerabilities
- **TLS 1.2+ enforced** by default via rustls
- **Certificate verification** uses platform-native roots via `rustls-platform-verifier`
- **No certificate pinning** — if you need pinning, use a custom `reqwest::Client` or a proxy

### Dependency Security Considerations

#### reqwest 0.13.2

`reqwest` is the HTTP client used for all API communication. Key security properties:

- **Redirect handling:** By default, reqwest follows up to 10 redirects. This crate communicates only with the configured `host` (default: `https://api.voyageai.com`). A malicious redirect could leak the `Authorization` header to a different host. Reqwest 0.13+ strips sensitive headers on cross-origin redirects by default.
- **Response size:** There is no explicit response body size limit. A malicious or compromised server could return an arbitrarily large response. For production use, configure an appropriate `timeout` in `Config`.
- **Timeout:** No default timeout is set. Always configure `Config.timeout` in production to prevent indefinite hangs.

#### serde_json

- Untrusted JSON is deserialized into strongly-typed structs
- Unknown fields are silently ignored (serde default), preventing injection of unexpected data
- Deserialization errors are surfaced as `Error::Json` — never as panics

#### Full Dependency Chain

Run `cargo tree` to inspect the complete dependency graph, or use `cargo audit` to check for known advisories:

```bash
cargo install cargo-audit
cargo audit
```

### What This Crate Does NOT Do

- Does **not** store or cache API keys on disk
- Does **not** log request/response bodies (unless the caller configures `tracing` subscribers)
- Does **not** execute any user-provided code or eval expressions
- Does **not** perform any filesystem operations
- Does **not** spawn subprocesses
- Does **not** use `unsafe` code

## Recommendations for Users

1. **Always set a timeout** to prevent resource exhaustion:
   ```rust
   use std::time::Duration;
   use mongodb_voyageai::Config;

   let config = Config {
       timeout: Some(Duration::from_secs(30)),
       ..Config::default()
   };
   ```

2. **Store API keys securely** — use environment variables or a secrets manager, never hardcode them in source.

3. **Pin your dependencies** — use `Cargo.lock` in applications (not libraries) to ensure reproducible builds.

4. **Run `cargo audit` regularly** to catch known vulnerabilities in the dependency tree.

5. **Keep dependencies updated** — run `cargo update` periodically and review changelogs.

## Auditing

This crate has not undergone a formal third-party security audit. If you require an audited HTTP client for regulated environments, please evaluate accordingly.

## Acknowledgements

**We appreciate responsible disclosure from security researchers. Contributors who report valid vulnerabilities will be credited in the advisory (unless they prefer to remain anonymous).**
