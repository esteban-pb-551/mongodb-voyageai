//! Individual reranking result.

use serde::Deserialize;
use std::fmt;

/// A single reranking result for one document.
///
/// Each [`Reranking`] maps a document (by index) to its relevance score
/// relative to the query.
///
/// # Examples
///
/// Deserialise from JSON:
///
/// ```rust
/// use mongodb_voyageai::Reranking;
///
/// let json = r#"{"index": 2, "document": "Rust is fast.", "relevance_score": 0.95}"#;
/// let r: Reranking = serde_json::from_str(json).unwrap();
///
/// assert_eq!(r.index, 2);
/// assert_eq!(r.document, Some("Rust is fast.".into()));
/// assert_eq!(r.relevance_score, 0.95);
/// ```
///
/// The `document` field is optional — the API may omit it:
///
/// ```rust
/// use mongodb_voyageai::Reranking;
///
/// let json = r#"{"index": 0, "relevance_score": 0.5}"#;
/// let r: Reranking = serde_json::from_str(json).unwrap();
///
/// assert_eq!(r.document, None);
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct Reranking {
    /// The zero-based index of the document in the original input list.
    pub index: usize,
    /// The document text, if returned by the API.
    pub document: Option<String>,
    /// Relevance score between 0.0 and 1.0 (higher is more relevant).
    pub relevance_score: f64,
}

impl fmt::Display for Reranking {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Reranking {{ index: {}, relevance_score: {} }}",
            self.index, self.relevance_score
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_with_document() {
        let r: Reranking =
            serde_json::from_str(r#"{"index": 0, "document": "Sample", "relevance_score": 0.5}"#)
                .unwrap();
        assert_eq!(r.index, 0);
        assert_eq!(r.document, Some("Sample".into()));
        assert_eq!(r.relevance_score, 0.5);
    }

    #[test]
    fn parse_without_document() {
        let r: Reranking = serde_json::from_str(r#"{"index": 2, "relevance_score": 0.9}"#).unwrap();
        assert_eq!(r.index, 2);
        assert_eq!(r.document, None);
    }

    #[test]
    fn parse_null_document() {
        let r: Reranking =
            serde_json::from_str(r#"{"index": 1, "document": null, "relevance_score": 0.3}"#)
                .unwrap();
        assert_eq!(r.document, None);
    }

    #[test]
    fn parse_boundary_scores() {
        let zero: Reranking =
            serde_json::from_str(r#"{"index": 0, "relevance_score": 0.0}"#).unwrap();
        assert_eq!(zero.relevance_score, 0.0);

        let one: Reranking =
            serde_json::from_str(r#"{"index": 0, "relevance_score": 1.0}"#).unwrap();
        assert_eq!(one.relevance_score, 1.0);
    }

    #[test]
    fn parse_missing_index() {
        assert!(serde_json::from_str::<Reranking>(r#"{"relevance_score": 0.5}"#).is_err());
    }

    #[test]
    fn parse_missing_score() {
        assert!(serde_json::from_str::<Reranking>(r#"{"index": 0}"#).is_err());
    }

    #[test]
    fn display() {
        let r = Reranking {
            index: 0,
            document: Some("Sample".into()),
            relevance_score: 0.0,
        };
        assert_eq!(format!("{r}"), "Reranking { index: 0, relevance_score: 0 }");
    }

    #[test]
    fn display_high_precision() {
        let r = Reranking {
            index: 3,
            document: None,
            relevance_score: 0.87654321,
        };
        let s = format!("{r}");
        assert!(s.contains("index: 3"));
        assert!(s.contains("0.87654321"));
    }

    #[test]
    fn clone_reranking() {
        let r = Reranking {
            index: 1,
            document: Some("Doc".into()),
            relevance_score: 0.75,
        };
        let c = r.clone();
        assert_eq!(c.index, 1);
        assert_eq!(c.document, Some("Doc".into()));
        assert_eq!(c.relevance_score, 0.75);
    }
}
