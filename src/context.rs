//! Contextualized chunk embedding types.

use serde::{Deserialize, Serialize};
use std::fmt;

use crate::usage::Usage;

/// A single embedding within a document result.
#[derive(Debug, Clone, Deserialize)]
struct EmbeddingData {
    embedding: Vec<f64>,
    #[allow(dead_code)]
    index: usize,
}

/// A single result from a contextualized embeddings request.
///
/// Contains the embeddings for a query, document, or document chunks,
/// along with optional chunk texts if a chunking function was used.
#[derive(Debug, Clone, Deserialize)]
pub struct ContextualizedEmbeddingsResult {
    /// The embeddings data for the input texts (query, document, or chunks).
    data: Vec<EmbeddingData>,
    /// The text of the document chunks (only present when chunk_fn is provided).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chunk_texts: Option<Vec<String>>,
    /// The index of the query or document in the input list.
    pub index: usize,
}

impl ContextualizedEmbeddingsResult {
    /// Returns the embeddings as a vector of vectors.
    pub fn embeddings(&self) -> Vec<Vec<f64>> {
        self.data.iter().map(|d| d.embedding.clone()).collect()
    }
}

/// The response from a contextualized embeddings request.
///
/// Contains the model used, token usage, and the resulting contextualized
/// embedding vectors for each input.
///
/// # Examples
///
/// Parse a raw JSON response:
///
/// ```rust
/// use mongodb_voyageai::ContextualizedEmbed;
///
/// let json = r#"{
///     "object": "list",
///     "data": [
///         {
///             "object": "list",
///             "data": [
///                 {"object": "embedding", "embedding": [0.1, 0.2], "index": 0}
///             ],
///             "index": 0
///         }
///     ],
///     "model": "voyage-context-3",
///     "usage": {"total_tokens": 10}
/// }"#;
///
/// let embed = ContextualizedEmbed::parse(json).unwrap();
/// assert_eq!(embed.model, "voyage-context-3");
/// assert_eq!(embed.usage.total_tokens, 10);
/// assert_eq!(embed.results.len(), 1);
/// ```
#[derive(Debug, Clone)]
pub struct ContextualizedEmbed {
    /// The model that produced the embeddings.
    pub model: String,
    /// Token usage for this request.
    pub usage: Usage,
    /// The results, one per input (query or document).
    pub results: Vec<ContextualizedEmbeddingsResult>,
}

#[derive(Debug, Clone, Deserialize)]
struct ContextualizedEmbedResponse {
    model: String,
    data: Vec<ContextualizedEmbeddingsResult>,
    usage: Usage,
}

impl ContextualizedEmbed {
    /// Parses a JSON response body into a [`ContextualizedEmbed`].
    ///
    /// # Errors
    ///
    /// Returns a [`serde_json::Error`] if the JSON is malformed or missing
    /// required fields (`model`, `data`, `usage`).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use mongodb_voyageai::ContextualizedEmbed;
    ///
    /// let json = r#"{
    ///     "object": "list",
    ///     "data": [
    ///         {
    ///             "object": "list",
    ///             "data": [{"object": "embedding", "embedding": [1.0], "index": 0}],
    ///             "index": 0
    ///         }
    ///     ],
    ///     "model": "voyage-context-3",
    ///     "usage": {"total_tokens": 5}
    /// }"#;
    /// let embed = ContextualizedEmbed::parse(json).unwrap();
    /// assert_eq!(embed.results[0].embeddings()[0], vec![1.0]);
    /// ```
    pub fn parse(data: &str) -> Result<Self, serde_json::Error> {
        let response: ContextualizedEmbedResponse = serde_json::from_str(data)?;
        Ok(Self {
            model: response.model,
            usage: response.usage,
            results: response.data,
        })
    }

    /// Returns the result at the given `index`, or `None` if out of bounds.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use mongodb_voyageai::ContextualizedEmbed;
    /// # let json = r#"{
    /// #     "object": "list",
    /// #     "data": [
    /// #         {
    /// #             "object": "list",
    /// #             "data": [{"object": "embedding", "embedding": [0.5, 0.6], "index": 0}],
    /// #             "index": 0
    /// #         }
    /// #     ],
    /// #     "model": "voyage-context-3",
    /// #     "usage": {"total_tokens": 1}
    /// # }"#;
    /// # let embed = ContextualizedEmbed::parse(json).unwrap();
    /// let first = embed.result(0).unwrap();
    /// assert_eq!(first.embeddings()[0], vec![0.5, 0.6]);
    ///
    /// assert!(embed.result(1).is_none());
    /// ```
    pub fn result(&self, index: usize) -> Option<&ContextualizedEmbeddingsResult> {
        self.results.get(index)
    }
}

impl fmt::Display for ContextualizedEmbed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ContextualizedEmbed {{ model: {:?}, results: {} result(s), usage: {} }}",
            self.model,
            self.results.len(),
            self.usage
        )
    }
}

/// The JSON payload sent to the `/contextualizedembeddings` endpoint.
///
/// # Examples
///
/// ```rust
/// use mongodb_voyageai::client::ContextualizedEmbedInput;
/// use mongodb_voyageai::OutputDtype;
///
/// let input = ContextualizedEmbedInput {
///     inputs: vec![
///         vec!["chunk 1".into(), "chunk 2".into()],
///         vec!["chunk 3".into()],
///     ],
///     model: "voyage-context-3".into(),
///     input_type: Some("document".into()),
///     output_dimension: None,
///     output_dtype: None,
/// };
///
/// let json = serde_json::to_value(&input).unwrap();
/// assert_eq!(json["model"], "voyage-context-3");
/// assert_eq!(json["input_type"], "document");
/// ```
#[derive(Debug, Clone, Serialize)]
pub struct ContextualizedEmbedInput {
    /// A list of lists, where each inner list contains texts to be vectorized together.
    pub inputs: Vec<Vec<String>>,
    /// The embedding model to use.
    pub model: String,
    /// Optional input type (`"query"` or `"document"`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_type: Option<String>,
    /// Reduce the embedding to this many dimensions.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_dimension: Option<u32>,
    /// Output data type for quantization (float, int8, uint8, binary, ubinary).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_dtype: Option<crate::output_dtype::OutputDtype>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_single_result() {
        let json = r#"{
            "object": "list",
            "data": [
                {
                    "object": "list",
                    "data": [
                        {"object": "embedding", "embedding": [0.1, 0.2], "index": 0},
                        {"object": "embedding", "embedding": [0.3, 0.4], "index": 1}
                    ],
                    "index": 0
                }
            ],
            "model": "voyage-context-3",
            "usage": {"total_tokens": 10}
        }"#;
        let embed = ContextualizedEmbed::parse(json).unwrap();
        assert_eq!(embed.model, "voyage-context-3");
        assert_eq!(embed.usage.total_tokens, 10);
        assert_eq!(embed.results.len(), 1);
        assert_eq!(embed.results[0].embeddings().len(), 2);
        assert_eq!(embed.results[0].index, 0);
    }

    #[test]
    fn parse_multiple_results() {
        let json = r#"{
            "object": "list",
            "data": [
                {
                    "object": "list",
                    "data": [
                        {"object": "embedding", "embedding": [0.1, 0.2], "index": 0}
                    ],
                    "index": 0
                },
                {
                    "object": "list",
                    "data": [
                        {"object": "embedding", "embedding": [0.3, 0.4], "index": 0},
                        {"object": "embedding", "embedding": [0.5, 0.6], "index": 1}
                    ],
                    "index": 1
                }
            ],
            "model": "voyage-context-3",
            "usage": {"total_tokens": 20}
        }"#;
        let embed = ContextualizedEmbed::parse(json).unwrap();
        assert_eq!(embed.results.len(), 2);
        assert_eq!(embed.results[0].embeddings().len(), 1);
        assert_eq!(embed.results[1].embeddings().len(), 2);
    }

    #[test]
    fn parse_with_chunk_texts() {
        let json = r#"{
            "object": "list",
            "data": [
                {
                    "object": "list",
                    "data": [
                        {"object": "embedding", "embedding": [0.1, 0.2], "index": 0}
                    ],
                    "chunk_texts": ["chunk 1", "chunk 2"],
                    "index": 0
                }
            ],
            "model": "voyage-context-3",
            "usage": {"total_tokens": 5}
        }"#;
        let embed = ContextualizedEmbed::parse(json).unwrap();
        assert!(embed.results[0].chunk_texts.is_some());
        assert_eq!(
            embed.results[0].chunk_texts.as_ref().unwrap(),
            &vec!["chunk 1", "chunk 2"]
        );
    }

    #[test]
    fn result_accessor() {
        let json = r#"{
            "object": "list",
            "data": [
                {
                    "object": "list",
                    "data": [{"object": "embedding", "embedding": [0.1], "index": 0}],
                    "index": 0
                },
                {
                    "object": "list",
                    "data": [{"object": "embedding", "embedding": [0.2], "index": 0}],
                    "index": 1
                }
            ],
            "model": "voyage-context-3",
            "usage": {"total_tokens": 2}
        }"#;
        let embed = ContextualizedEmbed::parse(json).unwrap();
        assert!(embed.result(0).is_some());
        assert!(embed.result(1).is_some());
        assert!(embed.result(2).is_none());
    }

    #[test]
    fn display() {
        let json = r#"{
            "object": "list",
            "data": [
                {
                    "object": "list",
                    "data": [{"object": "embedding", "embedding": [0.5], "index": 0}],
                    "index": 0
                }
            ],
            "model": "voyage-context-3",
            "usage": {"total_tokens": 3}
        }"#;
        let display = format!("{}", ContextualizedEmbed::parse(json).unwrap());
        assert!(display.contains("voyage-context-3"));
        assert!(display.contains("1 result"));
    }
}
