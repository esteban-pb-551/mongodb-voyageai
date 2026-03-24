# Publishing voyageai to crates.io

Step-by-step instructions for preparing, packaging, and publishing the `voyageai` crate to [crates.io](https://crates.io/).

---

## Prerequisites

### 1. Rust toolchain

You need `cargo` installed (ships with [rustup](https://rustup.rs/)):

```bash
rustup update stable
cargo --version   # 1.85+ required for edition 2024
```

### 2. crates.io account

1. Go to [https://crates.io/](https://crates.io/) and click **Log in with GitHub**.
2. Authorise the crates.io GitHub OAuth app.
3. You now have an account tied to your GitHub identity.

### 3. API token

1. Navigate to [https://crates.io/settings/tokens](https://crates.io/settings/tokens).
2. Click **New Token**.
3. Give it a name (e.g. `publish-voyageai`) and select the **publish-new** and **publish-update** scopes.
4. Copy the token and log in locally:

```bash
cargo login <YOUR_TOKEN>
```

This saves the token to `~/.cargo/credentials.toml`. Do **not** commit this file.

---

## Cargo.toml requirements

crates.io enforces several metadata fields. The current `Cargo.toml` is missing a few — here is the complete `[package]` section you need:

```toml
[package]
name = "mongodb-voyageai"
version = "1.0.1"
edition = "2024"
description = "A client for generating embeddings and reranking with Voyage AI"
license = "MIT"
repository = "https://github.com/<owner>/mongodb-voyageai"
homepage = "https://github.com/<owner>/mongodb-voyageai"
documentation = "https://docs.rs/mongodb-voyageai"
readme = "README.md"
keywords = ["voyageai", "embeddings", "rerank", "ai", "nlp", "mongodb"]
categories = ["api-bindings", "science"]
```

### Required fields

| Field | Why | Status |
|-------|-----|--------|
| `name` | Unique crate name on crates.io | Present |
| `version` | [SemVer](https://semver.org/) version | Present |
| `edition` | Rust edition | Present |
| `description` | Short summary (shown in search results) | Present |
| `license` **or** `license-file` | Must be a valid [SPDX expression](https://spdx.org/licenses/). `"MIT"` works since `LICENSE.txt` exists | Present |

### Strongly recommended fields

| Field | Why | Status |
|-------|-----|--------|
| `repository` | Link to source code (shown on crates.io page) | **Missing** |
| `homepage` | Project homepage | **Missing** |
| `documentation` | Link to docs (defaults to docs.rs) | **Missing** |
| `readme` | Path to README displayed on crates.io | **Missing** |
| `keywords` | Up to 5 keywords for search (lowercase, no spaces) | **Missing** |
| `categories` | Up to 5 [valid categories](https://crates.io/category_slugs) | **Missing** |

### Naming rules

- Crate names must be ASCII, start with a letter, and contain only letters, digits, `-`, or `_`.
- Maximum 64 characters.
- Names are first-come, first-served. Check availability: `cargo search voyageai`.
- The name `voyageai` may already be taken — if so, consider `voyageai-rs` or `voyage-ai`.

---

## Files included in the package

By default `cargo package` includes:

- `Cargo.toml` (rewritten with absolute dependency versions)
- `src/**/*.rs`
- `examples/**/*.rs`
- `README.md` (if declared via `readme`)
- `LICENSE.txt` (if declared via `license-file`, or auto-detected)

To see exactly what will be published:

```bash
cargo package --list
```

### .gitignore / .ignore

Make sure `target/`, `.env`, and any secrets are excluded. Create a `.gitignore` if one doesn't exist:

```gitignore
/target
.env
*.swp
```

### Size limit

crates.io enforces a **10 MB** maximum for the compressed package. Check with:

```bash
cargo package --allow-dirty
ls -lh target/package/voyageai-1.9.0.crate
```

---

## Pre-publish checklist

Run all of these and make sure they pass:

```bash
# 1. Format
cargo fmt --check

# 2. Lint
cargo clippy --all-targets

# 3. Unit tests + doc-tests
cargo test

# 4. Build docs (catch broken links)
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps

# 5. Check for missing docs on public items
cargo rustc --lib -- -W missing_docs

# 6. Dry-run package (catches metadata issues)
cargo package
```

All files must be committed to git before `cargo package` will succeed (or pass `--allow-dirty` for testing).

---

## Step-by-step publishing

### 1. Update Cargo.toml metadata

Add the missing fields listed above. Replace `<owner>` with your GitHub username or org:

```bash
# Edit Cargo.toml — add repository, homepage, documentation, readme, keywords, categories
```

### 2. Commit everything

```bash
git add -A
git commit -m "Prepare for crates.io release v1.9.0"
```

### 3. Create a git tag (recommended)

```bash
git tag v1.9.0
git push origin main --tags
```

### 4. Dry-run publish

```bash
cargo publish --dry-run
```

This runs `cargo package` and verifies everything without actually uploading. Fix any errors or warnings before proceeding.

### 5. Publish for real

```bash
cargo publish
```

This:
- Packages the crate into a `.crate` archive
- Uploads it to crates.io
- Makes it publicly available and **permanent** (you cannot delete a published version)

### 6. Verify

- Visit `https://crates.io/crates/voyageai`
- Docs will auto-build at `https://docs.rs/voyageai` within a few minutes

---

## Publishing updates

To publish a new version:

1. Bump the `version` in `Cargo.toml` (e.g. `1.9.0` -> `1.9.1`).
2. Run the pre-publish checklist.
3. Commit and tag.
4. `cargo publish`.

You **cannot** re-publish the same version. If you need to fix a mistake, bump the patch version.

---

## Yanking a version

If a published version has a critical bug, you can yank it (prevents new projects from depending on it, but existing lockfiles still work):

```bash
cargo yank --version 1.9.0
```

To undo:

```bash
cargo yank --version 1.9.0 --undo
```

---

## Common errors and fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `the remote server responded with an error: crate name is already taken` | Someone else owns the name | Choose a different name (e.g. `voyageai-rs`) |
| `manifest has no documentation, homepage or repository` | Warning — missing recommended fields | Add `repository`, `homepage`, `documentation` to `[package]` |
| `files in the working directory contain changes that were not yet committed` | Uncommitted changes | `git add -A && git commit` |
| `failed to verify package tarball` | Code doesn't compile from the packaged archive | Run `cargo package` and fix build errors |
| `the `license` field is not a valid SPDX expression` | Invalid license string | Use a valid SPDX identifier like `"MIT"`, `"Apache-2.0"`, or `"MIT OR Apache-2.0"` |
| `package exceeds the 10MB limit` | Too many or too large files included | Add unneeded files to `.gitignore` or use `exclude` in `Cargo.toml` |

---

## Summary of required changes before publishing

```diff
 [package]
 name = "voyageai"
 version = "1.9.0"
 edition = "2024"
 description = "A client for generating embeddings and reranking with https://voyageai.com"
 license = "MIT"
+repository = "https://github.com/<owner>/voyageai-rust"
+homepage = "https://github.com/<owner>/voyageai-rust"
+documentation = "https://docs.rs/voyageai"
+readme = "README.md"
+keywords = ["voyageai", "embeddings", "rerank", "ai", "nlp"]
+categories = ["api-bindings", "science"]
```

Once those fields are added, all pre-publish checks pass, and the code is committed and tagged — you're ready to `cargo publish`.
