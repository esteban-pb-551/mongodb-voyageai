//! Pre-defined model name constants for the VoyageAI API.
//!
//! Use these constants instead of raw strings to avoid typos and to make model
//! upgrades easier.
//!
//! # Examples
//!
//! ```rust
//! use mongodb_voyageai::model;
//!
//! // Default embedding model
//! assert_eq!(model::VOYAGE, "voyage-4");
//!
//! // Default reranking model
//! assert_eq!(model::RERANK, "rerank-2.5");
//!
//! // Use a specialised model
//! assert_eq!(model::VOYAGE_CODE_3, "voyage-code-3");
//! ```

// ── Embedding models ────────────────────────────────────────────────

/// Voyage 4 — balanced general-purpose embedding model.
pub const VOYAGE_4: &str = "voyage-4";
/// Voyage 4 Lite — fastest and cheapest variant.
pub const VOYAGE_4_LITE: &str = "voyage-4-lite";
/// Voyage 4 Large — highest quality (MoE architecture).
pub const VOYAGE_4_LARGE: &str = "voyage-4-large";

/// Voyage 3.5 — strong general-purpose model.
pub const VOYAGE_3_5: &str = "voyage-3.5";
/// Voyage 3.5 Lite — lighter, faster variant.
pub const VOYAGE_3_5_LITE: &str = "voyage-3.5-lite";

/// Voyage 3 — general-purpose embedding model.
pub const VOYAGE_3: &str = "voyage-3";
/// Voyage 3 Large — higher-capacity variant.
pub const VOYAGE_3_LARGE: &str = "voyage-3-large";
/// Voyage 3 Lite — cheaper, faster variant.
pub const VOYAGE_3_LITE: &str = "voyage-3-lite";

// --- Context-aware ---
/// Voyage Context 3 — contextual chunk embeddings (better RAG retrieval).
pub const VOYAGE_CONTEXT_3: &str = "voyage-context-3";

// --- Multimodal ---
/// Voyage Multimodal 3 — text + image embeddings.
pub const VOYAGE_MULTIMODAL_3: &str = "voyage-multimodal-3";
/// Voyage Multimodal 3.5 — adds video support.
pub const VOYAGE_MULTIMODAL_3_5: &str = "voyage-multimodal-3.5";

// --- Domain-specific models ---
/// Voyage Code 3 — optimized for code retrieval.
pub const VOYAGE_CODE_3: &str = "voyage-code-3";
/// Voyage Code 2 — previous generation code model.
pub const VOYAGE_CODE_2: &str = "voyage-code-2";

/// Voyage Finance 2 — optimized for financial text.
pub const VOYAGE_FINANCE_2: &str = "voyage-finance-2";
/// Voyage Multilingual 2 — multilingual embeddings.
pub const VOYAGE_MULTILINGUAL_2: &str = "voyage-multilingual-2";
/// Voyage Law 2 — legal domain embeddings.
pub const VOYAGE_LAW_2: &str = "voyage-law-2";

// --- Older / legacy general models ---
/// Voyage Large 2 — previous-gen large model.
pub const VOYAGE_LARGE_2: &str = "voyage-large-2";
/// Voyage Large 2 Instruct — instruction-tuned variant.
pub const VOYAGE_LARGE_2_INSTRUCT: &str = "voyage-large-2-instruct";

/// Voyage Lite 02 Instruct — lightweight instruct model.
pub const VOYAGE_LITE_02_INSTRUCT: &str = "voyage-lite-02-instruct";
/// Voyage Lite 01 — legacy lightweight model.
pub const VOYAGE_LITE_01: &str = "voyage-lite-01";
/// Voyage Lite 01 Instruct — instruction-tuned variant.
pub const VOYAGE_LITE_01_INSTRUCT: &str = "voyage-lite-01-instruct";

/// Voyage 2 — older general-purpose model.
pub const VOYAGE_2: &str = "voyage-2";
/// Voyage 01 — earliest legacy model.
pub const VOYAGE_01: &str = "voyage-01";

// ── Reranking models ────────────────────────────────────────────────

/// Rerank 2.5 — latest high-quality reranker.
pub const RERANK_2_5: &str = "rerank-2.5";
/// Rerank 2.5 Lite — faster, cheaper version.
pub const RERANK_2_5_LITE: &str = "rerank-2.5-lite";

/// Rerank 2 — standard reranking model.
pub const RERANK_2: &str = "rerank-2";
/// Rerank 2 Lite — lightweight variant.
pub const RERANK_2_LITE: &str = "rerank-2-lite";

/// Rerank 1 — legacy reranker.
pub const RERANK_1: &str = "rerank-1";
/// Rerank 1 Lite — lightweight legacy reranker.
pub const RERANK_1_LITE: &str = "rerank-lite-1";

// ── Convenience aliases (point to the latest version) ───────────────

/// Default embedding model (currently [`VOYAGE_4`]).
pub const VOYAGE: &str = VOYAGE_4;
/// Default reranking model (currently [`RERANK_2_5`]).
pub const RERANK: &str = RERANK_2_5;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedding_constants() {
        assert_eq!(VOYAGE, "voyage-4");
        assert_eq!(VOYAGE_4, "voyage-4");
        assert_eq!(VOYAGE_4_LITE, "voyage-4-lite");
        assert_eq!(VOYAGE_4_LARGE, "voyage-4-large");
        assert_eq!(VOYAGE_3_5, "voyage-3.5");
        assert_eq!(VOYAGE_3_5_LITE, "voyage-3.5-lite");
        assert_eq!(VOYAGE_3, "voyage-3");
        assert_eq!(VOYAGE_3_LARGE, "voyage-3-large");
        assert_eq!(VOYAGE_3_LITE, "voyage-3-lite");
        assert_eq!(VOYAGE_FINANCE_2, "voyage-finance-2");
        assert_eq!(VOYAGE_MULTILINGUAL_2, "voyage-multilingual-2");
        assert_eq!(VOYAGE_LAW_2, "voyage-law-2");
        assert_eq!(VOYAGE_CODE_3, "voyage-code-3");
        assert_eq!(VOYAGE_CODE_2, "voyage-code-2");
    }

    #[test]
    fn rerank_constants() {
        assert_eq!(RERANK, "rerank-2.5");
    }

    #[test]
    fn aliases_match_versioned() {
        assert_eq!(VOYAGE, VOYAGE_4);
        assert_eq!(RERANK, RERANK_2_5);
    }
}
