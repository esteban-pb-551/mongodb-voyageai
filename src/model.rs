//! Pre-defined model name constants for the VoyageAI API.
//!
//! Use these constants instead of raw strings to avoid typos and to make model
//! upgrades easier.
//!
//! # Examples
//!
//! ```rust
//! use voyageai::model;
//!
//! // Default embedding model
//! assert_eq!(model::VOYAGE, "voyage-3.5");
//!
//! // Default reranking model
//! assert_eq!(model::RERANK, "rerank-2");
//!
//! // Use a specialised model
//! assert_eq!(model::VOYAGE_CODE, "voyage-code-2");
//! ```

// ── Embedding models ────────────────────────────────────────────────

/// Voyage 3.5 — latest general-purpose embedding model.
pub const VOYAGE_3_5: &str = "voyage-3.5";
/// Voyage 3.5 Lite — lighter, faster variant of Voyage 3.5.
pub const VOYAGE_3_5_LITE: &str = "voyage-3.5-lite";
/// Voyage 3 — previous-generation general-purpose model.
pub const VOYAGE_3: &str = "voyage-3";
/// Voyage 3 Large — higher-capacity variant of Voyage 3.
pub const VOYAGE_3_LARGE: &str = "voyage-3-large";
/// Voyage 3 Lite — lighter variant of Voyage 3.
pub const VOYAGE_3_LITE: &str = "voyage-3-lite";
/// Voyage Finance 2 — optimised for financial text.
pub const VOYAGE_FINANCE_2: &str = "voyage-finance-2";
/// Voyage Multilingual 2 — optimised for multilingual text.
pub const VOYAGE_MULTILINGUAL_2: &str = "voyage-multilingual-2";
/// Voyage Law 2 — optimised for legal text.
pub const VOYAGE_LAW_2: &str = "voyage-law-2";
/// Voyage Code 2 — optimised for source code.
pub const VOYAGE_CODE_2: &str = "voyage-code-2";

// ── Reranking models ────────────────────────────────────────────────

/// Rerank 2 — default reranking model.
pub const RERANK_2: &str = "rerank-2";
/// Rerank 2 Lite — lighter, faster reranking model.
pub const RERANK_2_LITE: &str = "rerank-2-lite";

// ── Convenience aliases (point to the latest version) ───────────────

/// Default embedding model (currently [`VOYAGE_3_5`]).
pub const VOYAGE: &str = VOYAGE_3_5;
/// Default lite embedding model (currently [`VOYAGE_3_5_LITE`]).
pub const VOYAGE_LITE: &str = VOYAGE_3_5_LITE;
/// Finance embedding model (currently [`VOYAGE_FINANCE_2`]).
pub const VOYAGE_FINANCE: &str = VOYAGE_FINANCE_2;
/// Multilingual embedding model (currently [`VOYAGE_MULTILINGUAL_2`]).
pub const VOYAGE_MULTILINGUAL: &str = VOYAGE_MULTILINGUAL_2;
/// Law embedding model (currently [`VOYAGE_LAW_2`]).
pub const VOYAGE_LAW: &str = VOYAGE_LAW_2;
/// Code embedding model (currently [`VOYAGE_CODE_2`]).
pub const VOYAGE_CODE: &str = VOYAGE_CODE_2;

/// Default reranking model (currently [`RERANK_2`]).
pub const RERANK: &str = RERANK_2;
/// Lite reranking model (currently [`RERANK_2_LITE`]).
pub const RERANK_LITE: &str = RERANK_2_LITE;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedding_constants() {
        assert_eq!(VOYAGE, "voyage-3.5");
        assert_eq!(VOYAGE_LITE, "voyage-3.5-lite");
        assert_eq!(VOYAGE_3, "voyage-3");
        assert_eq!(VOYAGE_3_LARGE, "voyage-3-large");
        assert_eq!(VOYAGE_3_LITE, "voyage-3-lite");
        assert_eq!(VOYAGE_FINANCE, "voyage-finance-2");
        assert_eq!(VOYAGE_MULTILINGUAL, "voyage-multilingual-2");
        assert_eq!(VOYAGE_LAW, "voyage-law-2");
        assert_eq!(VOYAGE_CODE, "voyage-code-2");
    }

    #[test]
    fn rerank_constants() {
        assert_eq!(RERANK, "rerank-2");
        assert_eq!(RERANK_LITE, "rerank-2-lite");
    }

    #[test]
    fn aliases_match_versioned() {
        assert_eq!(VOYAGE, VOYAGE_3_5);
        assert_eq!(VOYAGE_LITE, VOYAGE_3_5_LITE);
        assert_eq!(VOYAGE_FINANCE, VOYAGE_FINANCE_2);
        assert_eq!(VOYAGE_MULTILINGUAL, VOYAGE_MULTILINGUAL_2);
        assert_eq!(VOYAGE_LAW, VOYAGE_LAW_2);
        assert_eq!(VOYAGE_CODE, VOYAGE_CODE_2);
        assert_eq!(RERANK, RERANK_2);
        assert_eq!(RERANK_LITE, RERANK_2_LITE);
    }
}
