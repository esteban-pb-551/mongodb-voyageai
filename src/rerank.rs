//! Reranking response types.

use serde::Deserialize;
use std::fmt;

use crate::reranking::Reranking;
use crate::usage::Usage;

#[derive(Debug, Clone, Deserialize)]
struct RerankResponse {
    model: String,
    data: Vec<Reranking>,
    usage: Usage,
}

/// The response from a reranking request.
///
/// Contains the model used, token usage, and the ranked results.
///
/// # Example
///
/// ```rust,no_run
#[doc = include_str!("../examples/rerank-documents.rs")]
/// ```
#[derive(Debug, Clone)]
pub struct Rerank {
    /// The model that performed the reranking.
    pub model: String,
    /// Token usage for this request.
    pub usage: Usage,
    /// Reranked results, in the order returned by the API.
    pub results: Vec<Reranking>,
}

impl Rerank {
    /// Parses a JSON response body into a [`Rerank`].
    ///
    /// # Errors
    ///
    /// Returns a [`serde_json::Error`] if the JSON is malformed or missing
    /// required fields (`model`, `data`, `usage`).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use mongodb_voyageai::Rerank;
    ///
    /// let json = r#"{
    ///     "object": "list",
    ///     "data": [{"index": 0, "relevance_score": 0.75}],
    ///     "model": "rerank-2",
    ///     "usage": {"total_tokens": 10}
    /// }"#;
    /// let rerank = Rerank::parse(json).unwrap();
    /// assert_eq!(rerank.results[0].relevance_score, 0.75);
    /// ```
    pub fn parse(data: &str) -> Result<Self, serde_json::Error> {
        let response: RerankResponse = serde_json::from_str(data)?;
        Ok(Self {
            model: response.model,
            usage: response.usage,
            results: response.data,
        })
    }
}

impl fmt::Display for Rerank {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Rerank {{ model: {:?}, usage: {} }}",
            self.model, self.usage
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_single_result() {
        let json = r#"{
            "object": "list",
            "data": [{"index": 0, "document": "Sample", "relevance_score": 0.75}],
            "model": "rerank-2", "usage": {"total_tokens": 5}
        }"#;
        let rerank = Rerank::parse(json).unwrap();
        assert_eq!(rerank.model, "rerank-2");
        assert_eq!(rerank.usage.total_tokens, 5);
        assert_eq!(rerank.results.len(), 1);
        assert_eq!(rerank.results[0].relevance_score, 0.75);
    }

    #[test]
    fn parse_multiple_results() {
        let json = r#"{
            "object": "list",
            "data": [
                {"index": 1, "relevance_score": 0.9},
                {"index": 0, "relevance_score": 0.6},
                {"index": 2, "relevance_score": 0.3}
            ],
            "model": "rerank-2", "usage": {"total_tokens": 20}
        }"#;
        let rerank = Rerank::parse(json).unwrap();
        assert_eq!(rerank.results.len(), 3);
        assert_eq!(rerank.results[0].index, 1);
    }

    #[test]
    fn parse_empty_results() {
        let json = r#"{
            "object": "list", "data": [],
            "model": "rerank-2", "usage": {"total_tokens": 0}
        }"#;
        assert!(Rerank::parse(json).unwrap().results.is_empty());
    }

    #[test]
    fn parse_invalid_json() {
        assert!(Rerank::parse("{{garbage").is_err());
    }

    #[test]
    fn parse_missing_model() {
        let json =
            r#"{"data": [{"index": 0, "relevance_score": 0.5}], "usage": {"total_tokens": 0}}"#;
        assert!(Rerank::parse(json).is_err());
    }

    #[test]
    fn display() {
        let json = r#"{
            "object": "list",
            "data": [{"index": 0, "relevance_score": 0.5}],
            "model": "rerank-2", "usage": {"total_tokens": 10}
        }"#;
        let display = format!("{}", Rerank::parse(json).unwrap());
        assert!(display.contains("rerank-2"));
        assert!(display.contains("10"));
    }

    #[test]
    fn debug_output() {
        let json = r#"{
            "object": "list", "data": [],
            "model": "rerank-2", "usage": {"total_tokens": 0}
        }"#;
        let debug = format!("{:?}", Rerank::parse(json).unwrap());
        assert!(debug.contains("Rerank"));
    }

    #[test]
    fn clone_rerank() {
        let json = r#"{
            "object": "list",
            "data": [{"index": 0, "relevance_score": 0.5}],
            "model": "rerank-2", "usage": {"total_tokens": 3}
        }"#;
        let rerank = Rerank::parse(json).unwrap();
        let cloned = rerank.clone();
        assert_eq!(cloned.model, rerank.model);
        assert_eq!(cloned.results.len(), rerank.results.len());
    }
}
