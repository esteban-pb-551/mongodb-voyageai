//! Token usage information returned by the VoyageAI API.

use serde::Deserialize;
use std::fmt;

/// Token usage for a single API request.
///
/// Every embedding and reranking response includes a `usage` object that
/// reports how many tokens were consumed.
///
/// # Examples
///
/// Deserialise from JSON:
///
/// ```rust
/// use voyageai::Usage;
///
/// let json = r#"{"total_tokens": 42}"#;
/// let usage: Usage = serde_json::from_str(json).unwrap();
/// assert_eq!(usage.total_tokens, 42);
/// ```
///
/// Display:
///
/// ```rust
/// use voyageai::Usage;
///
/// let usage = Usage { total_tokens: 100 };
/// assert_eq!(format!("{usage}"), "Usage { total_tokens: 100 }");
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct Usage {
    /// The total number of tokens consumed by the request.
    pub total_tokens: u64,
}

impl fmt::Display for Usage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Usage {{ total_tokens: {} }}", self.total_tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse() {
        let usage: Usage = serde_json::from_str(r#"{"total_tokens": 42}"#).unwrap();
        assert_eq!(usage.total_tokens, 42);
    }

    #[test]
    fn parse_zero() {
        let usage: Usage = serde_json::from_str(r#"{"total_tokens": 0}"#).unwrap();
        assert_eq!(usage.total_tokens, 0);
    }

    #[test]
    fn parse_large_value() {
        let usage: Usage = serde_json::from_str(r#"{"total_tokens": 999999999}"#).unwrap();
        assert_eq!(usage.total_tokens, 999_999_999);
    }

    #[test]
    fn parse_ignores_extra_fields() {
        let usage: Usage = serde_json::from_str(
            r#"{"total_tokens": 10, "prompt_tokens": 5, "completion_tokens": 5}"#,
        )
        .unwrap();
        assert_eq!(usage.total_tokens, 10);
    }

    #[test]
    fn parse_missing_field() {
        let result: Result<Usage, _> = serde_json::from_str(r#"{}"#);
        assert!(result.is_err());
    }

    #[test]
    fn display() {
        assert_eq!(
            format!("{}", Usage { total_tokens: 0 }),
            "Usage { total_tokens: 0 }"
        );
        assert_eq!(
            format!("{}", Usage { total_tokens: 1234 }),
            "Usage { total_tokens: 1234 }"
        );
    }

    #[test]
    fn debug_output() {
        let debug = format!("{:?}", Usage { total_tokens: 7 });
        assert!(debug.contains("7"));
        assert!(debug.contains("Usage"));
    }

    #[test]
    fn clone_usage() {
        let usage = Usage { total_tokens: 42 };
        assert_eq!(usage.clone().total_tokens, 42);
    }
}
