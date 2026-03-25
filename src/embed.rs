//! Embedding response types.

use serde::Deserialize;
use std::fmt;

use crate::usage::Usage;

#[derive(Debug, Clone, Deserialize)]
struct EmbeddingData {
    embedding: Vec<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct EmbedResponse {
    model: String,
    data: Vec<EmbeddingData>,
    usage: Usage,
}

/// The response from an embeddings request.
///
/// Contains the model used, token usage, and the resulting embedding vectors.
///
/// # Examples
///
/// Parse a raw JSON response:
///
/// ```rust
/// use mongodb_voyageai::Embed;
///
/// let json = r#"{
///     "object": "list",
///     "data": [
///         {"object": "embedding", "embedding": [0.1, 0.2, 0.3], "index": 0}
///     ],
///     "model": "voyage-4",
///     "usage": {"total_tokens": 5}
/// }"#;
///
/// let embed = Embed::parse(json).unwrap();
/// assert_eq!(embed.model, "voyage-4");
/// assert_eq!(embed.usage.total_tokens, 5);
/// assert_eq!(embed.embeddings.len(), 1);
/// assert_eq!(embed.embedding(0).unwrap(), &vec![0.1, 0.2, 0.3]);
/// ```
///
/// Access individual embeddings:
///
/// ```rust
/// # use mongodb_voyageai::Embed;
/// # let json = r#"{
/// #     "object": "list",
/// #     "data": [
/// #         {"object": "embedding", "embedding": [0.1], "index": 0},
/// #         {"object": "embedding", "embedding": [0.2], "index": 1}
/// #     ],
/// #     "model": "voyage-3",
/// #     "usage": {"total_tokens": 2}
/// # }"#;
/// # let embed = Embed::parse(json).unwrap();
/// // First embedding
/// assert_eq!(embed.embedding(0), Some(&vec![0.1]));
///
/// // Out-of-bounds returns None
/// assert_eq!(embed.embedding(99), None);
/// ```
#[derive(Debug, Clone)]
pub struct Embed {
    /// The model that produced the embeddings.
    pub model: String,
    /// Token usage for this request.
    pub usage: Usage,
    /// The embedding vectors, one per input text.
    pub embeddings: Vec<Vec<f64>>,
}

impl Embed {
    /// Parses a JSON response body into an [`Embed`].
    ///
    /// # Errors
    ///
    /// Returns a [`serde_json::Error`] if the JSON is malformed or missing
    /// required fields (`model`, `data`, `usage`).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use mongodb_voyageai::Embed;
    ///
    /// let json = r#"{
    ///     "object": "list",
    ///     "data": [{"object": "embedding", "embedding": [1.0], "index": 0}],
    ///     "model": "voyage-3",
    ///     "usage": {"total_tokens": 1}
    /// }"#;
    /// let embed = Embed::parse(json).unwrap();
    /// assert_eq!(embed.embeddings[0], vec![1.0]);
    /// ```
    pub fn parse(data: &str) -> Result<Self, serde_json::Error> {
        let response: EmbedResponse = serde_json::from_str(data)?;
        let embeddings = response.data.into_iter().map(|d| d.embedding).collect();
        Ok(Self {
            model: response.model,
            usage: response.usage,
            embeddings,
        })
    }

    /// Returns the embedding at the given `index`, or `None` if out of bounds.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use mongodb_voyageai::Embed;
    /// # let json = r#"{
    /// #     "object": "list",
    /// #     "data": [{"object": "embedding", "embedding": [0.5, 0.6], "index": 0}],
    /// #     "model": "voyage-3",
    /// #     "usage": {"total_tokens": 1}
    /// # }"#;
    /// # let embed = Embed::parse(json).unwrap();
    /// let first = embed.embedding(0).unwrap();
    /// assert_eq!(first, &vec![0.5, 0.6]);
    ///
    /// assert!(embed.embedding(1).is_none());
    /// ```
    pub fn embedding(&self, index: usize) -> Option<&Vec<f64>> {
        self.embeddings.get(index)
    }
}

impl fmt::Display for Embed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Embed {{ model: {:?}, embeddings: {:?}, usage: {} }}",
            self.model, self.embeddings, self.usage
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_single() {
        let json = r#"{
            "object": "list",
            "data": [{"object": "embedding", "embedding": [0.0, 1.0, 2.0], "index": 0}],
            "model": "voyage-3",
            "usage": {"total_tokens": 10}
        }"#;
        let embed = Embed::parse(json).unwrap();
        assert_eq!(embed.model, "voyage-3");
        assert_eq!(embed.usage.total_tokens, 10);
        assert_eq!(embed.embeddings, vec![vec![0.0, 1.0, 2.0]]);
    }

    #[test]
    fn parse_multiple() {
        let json = r#"{
            "object": "list",
            "data": [
                {"object": "embedding", "embedding": [0.1, 0.2], "index": 0},
                {"object": "embedding", "embedding": [0.3, 0.4], "index": 1},
                {"object": "embedding", "embedding": [0.5, 0.6], "index": 2}
            ],
            "model": "voyage-3-large",
            "usage": {"total_tokens": 30}
        }"#;
        let embed = Embed::parse(json).unwrap();
        assert_eq!(embed.embeddings.len(), 3);
        assert_eq!(embed.embeddings[0], vec![0.1, 0.2]);
        assert_eq!(embed.embeddings[2], vec![0.5, 0.6]);
    }

    #[test]
    fn parse_empty_data() {
        let json = r#"{
            "object": "list", "data": [],
            "model": "voyage-3", "usage": {"total_tokens": 0}
        }"#;
        let embed = Embed::parse(json).unwrap();
        assert!(embed.embeddings.is_empty());
        assert_eq!(embed.embedding(0), None);
    }

    #[test]
    fn parse_high_dimensional() {
        let values: Vec<f64> = (0..1024).map(|i| i as f64 * 0.001).collect();
        let values_json = serde_json::to_string(&values).unwrap();
        let json = format!(
            r#"{{"object":"list","data":[{{"object":"embedding","embedding":{},"index":0}}],"model":"voyage-3","usage":{{"total_tokens":1}}}}"#,
            values_json
        );
        let embed = Embed::parse(&json).unwrap();
        assert_eq!(embed.embeddings[0].len(), 1024);
        assert!((embed.embeddings[0][500] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn parse_negative_values() {
        let json = r#"{
            "object": "list",
            "data": [{"object": "embedding", "embedding": [-1.5, 0.0, 1.5], "index": 0}],
            "model": "voyage-3", "usage": {"total_tokens": 1}
        }"#;
        assert_eq!(
            Embed::parse(json).unwrap().embeddings[0],
            vec![-1.5, 0.0, 1.5]
        );
    }

    #[test]
    fn parse_ignores_extra_fields() {
        let json = r#"{
            "object": "list",
            "data": [{"object": "embedding", "embedding": [1.0], "index": 0}],
            "model": "voyage-3", "usage": {"total_tokens": 1}, "extra_field": "ignored"
        }"#;
        assert_eq!(Embed::parse(json).unwrap().embeddings[0], vec![1.0]);
    }

    #[test]
    fn parse_invalid_json() {
        assert!(Embed::parse("not json").is_err());
    }

    #[test]
    fn parse_missing_model() {
        let json = r#"{
            "object": "list",
            "data": [{"object": "embedding", "embedding": [1.0], "index": 0}],
            "usage": {"total_tokens": 0}
        }"#;
        assert!(Embed::parse(json).is_err());
    }

    #[test]
    fn parse_missing_data() {
        assert!(Embed::parse(r#"{"model": "voyage-3", "usage": {"total_tokens": 0}}"#).is_err());
    }

    #[test]
    fn parse_missing_usage() {
        let json = r#"{
            "object": "list",
            "data": [{"object": "embedding", "embedding": [1.0], "index": 0}],
            "model": "voyage-3"
        }"#;
        assert!(Embed::parse(json).is_err());
    }

    #[test]
    fn embedding_accessor() {
        let json = r#"{
            "object": "list",
            "data": [
                {"object": "embedding", "embedding": [0.0], "index": 0},
                {"object": "embedding", "embedding": [1.0], "index": 1}
            ],
            "model": "voyage-large-2", "usage": {"total_tokens": 0}
        }"#;
        let embed = Embed::parse(json).unwrap();
        assert_eq!(embed.embedding(0), Some(&vec![0.0]));
        assert_eq!(embed.embedding(1), Some(&vec![1.0]));
        assert_eq!(embed.embedding(2), None);
        assert_eq!(embed.embedding(999), None);
    }

    #[test]
    fn display() {
        let json = r#"{
            "object": "list",
            "data": [{"object": "embedding", "embedding": [0.5], "index": 0}],
            "model": "voyage-3", "usage": {"total_tokens": 3}
        }"#;
        let display = format!("{}", Embed::parse(json).unwrap());
        assert!(display.contains("voyage-3"));
        assert!(display.contains("0.5"));
    }

    #[test]
    fn debug_output() {
        let json = r#"{
            "object": "list", "data": [],
            "model": "voyage-3", "usage": {"total_tokens": 0}
        }"#;
        let debug = format!("{:?}", Embed::parse(json).unwrap());
        assert!(debug.contains("Embed"));
        assert!(debug.contains("voyage-3"));
    }

    #[test]
    fn clone_embed() {
        let json = r#"{
            "object": "list",
            "data": [{"object": "embedding", "embedding": [1.0, 2.0], "index": 0}],
            "model": "voyage-3", "usage": {"total_tokens": 5}
        }"#;
        let embed = Embed::parse(json).unwrap();
        let cloned = embed.clone();
        assert_eq!(cloned.model, embed.model);
        assert_eq!(cloned.embeddings, embed.embeddings);
        assert_eq!(cloned.usage.total_tokens, embed.usage.total_tokens);
    }
}
