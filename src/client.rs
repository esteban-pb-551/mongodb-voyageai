//! HTTP client for the VoyageAI API.

use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue};
use serde::Serialize;

use crate::config::Config;
use crate::embed::Embed;
use crate::model;
use crate::rerank::Rerank;

/// Errors that can occur when using the VoyageAI [`Client`].
///
/// # Examples
///
/// ```rust
/// use mongodb_voyageai::Error;
///
/// let err = Error::RequestError {
///     status: 401,
///     body: "Unauthorized".into(),
/// };
/// assert!(format!("{err}").contains("401"));
/// ```
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// No API key was provided and `VOYAGEAI_API_KEY` is not set.
    #[error("api_key is required or ENV['VOYAGEAI_API_KEY'] must be present")]
    MissingApiKey,
    /// The API returned a non-2xx HTTP status.
    #[error("request error: status={status} body={body}")]
    RequestError {
        /// HTTP status code.
        status: u16,
        /// Response body.
        body: String,
    },
    /// A network or connection-level error from `reqwest`.
    #[error(transparent)]
    Http(#[from] reqwest::Error),
    /// A JSON serialisation or deserialisation error.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

/// The JSON payload sent to the `/embeddings` endpoint.
///
/// # Examples
///
/// ```rust
/// use mongodb_voyageai::client::EmbedInput;
///
/// let input = EmbedInput {
///     input: vec!["hello".into()],
///     model: "voyage-3.5".into(),
///     input_type: Some("query".into()),
///     truncation: None,
///     output_dimension: None,
/// };
///
/// let json = serde_json::to_value(&input).unwrap();
/// assert_eq!(json["model"], "voyage-3.5");
/// assert_eq!(json["input_type"], "query");
/// // None fields are omitted
/// assert!(json.get("truncation").is_none());
/// ```
#[derive(Debug, Clone, Serialize)]
pub struct EmbedInput {
    /// The texts to embed.
    pub input: Vec<String>,
    /// The embedding model to use.
    pub model: String,
    /// Optional input type (`"query"` or `"document"`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_type: Option<String>,
    /// Whether to truncate inputs that exceed the model's context length.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncation: Option<bool>,
    /// Reduce the embedding to this many dimensions.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_dimension: Option<u32>,
}

/// The JSON payload sent to the `/rerank` endpoint.
///
/// # Examples
///
/// ```rust
/// use mongodb_voyageai::client::RerankInput;
///
/// let input = RerankInput {
///     query: "search query".into(),
///     documents: vec!["doc A".into(), "doc B".into()],
///     model: "rerank-2".into(),
///     top_k: Some(1),
///     truncation: None,
/// };
///
/// let json = serde_json::to_value(&input).unwrap();
/// assert_eq!(json["top_k"], 1);
/// assert!(json.get("truncation").is_none());
/// ```
#[derive(Debug, Clone, Serialize)]
pub struct RerankInput {
    /// The search query.
    pub query: String,
    /// The documents to rerank.
    pub documents: Vec<String>,
    /// The reranking model to use.
    pub model: String,
    /// Return only the top K results.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    /// Whether to truncate inputs that exceed the model's context length.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncation: Option<bool>,
}

/// An async client for the VoyageAI API.
///
/// The client holds a connection pool and can be cloned cheaply to share
/// across tasks. The API key is masked in [`Debug`] output.
///
/// # Examples
///
/// Create from a [`Config`]:
///
/// ```rust
/// use mongodb_voyageai::{Client, Config};
///
/// let config = Config {
///     api_key: Some("pa-test-key".into()),
///     ..Config::default()
/// };
/// let client = Client::new(&config).unwrap();
/// ```
///
/// Shorthand with just an API key:
///
/// ```rust
/// use mongodb_voyageai::Client;
///
/// let client = Client::with_api_key("pa-test-key").unwrap();
/// ```
///
/// Debug output masks the key:
///
/// ```rust
/// use mongodb_voyageai::Client;
///
/// let client = Client::with_api_key("pa-secret-key-123").unwrap();
/// let debug = format!("{client:?}");
/// assert!(debug.contains("pa-se***"));
/// assert!(!debug.contains("pa-secret-key-123"));
/// ```
#[derive(Clone)]
pub struct Client {
    api_key: String,
    host: String,
    version: String,
    http: reqwest::Client,
}

impl std::fmt::Debug for Client {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let masked = if self.api_key.len() > 5 {
            format!("{}***", &self.api_key[..5])
        } else {
            "***".to_string()
        };
        f.debug_struct("Client")
            .field("api_key", &masked)
            .field("host", &self.host)
            .field("version", &self.version)
            .finish()
    }
}

impl Client {
    /// Creates a new client from a [`Config`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::MissingApiKey`] if `config.api_key` is `None`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use mongodb_voyageai::{Client, Config};
    ///
    /// let config = Config {
    ///     api_key: Some("pa-key".into()),
    ///     ..Config::default()
    /// };
    /// let client = Client::new(&config).unwrap();
    /// ```
    ///
    /// Missing key returns an error:
    ///
    /// ```rust
    /// use mongodb_voyageai::{Client, Config, Error};
    ///
    /// let config = Config {
    ///     api_key: None,
    ///     ..Config::default()
    /// };
    /// assert!(matches!(Client::new(&config), Err(Error::MissingApiKey)));
    /// ```
    pub fn new(config: &Config) -> Result<Self, Error> {
        let api_key = config.api_key.clone().ok_or(Error::MissingApiKey)?;

        let mut headers = HeaderMap::new();
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {}", api_key)).expect("invalid api key"),
        );
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        let mut builder = reqwest::Client::builder().default_headers(headers);
        if let Some(timeout) = config.timeout {
            builder = builder.timeout(timeout);
        }

        Ok(Self {
            api_key,
            host: config.host.clone(),
            version: config.version.clone(),
            http: builder.build()?,
        })
    }

    /// Creates a new client using only an API key, with all other settings
    /// at their defaults.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Http`] if the underlying HTTP client fails to build.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use mongodb_voyageai::Client;
    ///
    /// let client = Client::with_api_key("pa-my-key").unwrap();
    /// ```
    pub fn with_api_key(api_key: &str) -> Result<Self, Error> {
        let config = Config {
            api_key: Some(api_key.to_string()),
            ..Config::default()
        };
        Self::new(&config)
    }

    /// Generates embeddings for the given input texts.
    ///
    /// # Arguments
    ///
    /// * `input` — One or more texts to embed.
    /// * `embed_model` — Model name (defaults to [`model::VOYAGE`]).
    /// * `input_type` — `"query"` or `"document"` (optional).
    /// * `truncation` — Truncate long inputs (optional).
    /// * `output_dimension` — Reduce dimensionality (optional).
    ///
    /// # Errors
    ///
    /// Returns [`Error::RequestError`] on non-2xx responses, [`Error::Http`]
    /// on network failures, or [`Error::Json`] on parse failures.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use mongodb_voyageai::Client;
    /// # use mongodb_voyageai::model;
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), mongodb_voyageai::Error> {
    /// let client = Client::with_api_key("pa-...")?;
    ///
    /// // Single text, default model
    /// let embed = client
    ///     .embed(vec!["Hello world".into()], None, None, None, None)
    ///     .await?;
    /// println!("dimensions: {}", embed.embedding(0).unwrap().len());
    ///
    /// // Multiple texts with a specific model and input type
    /// let embed = client
    ///     .embed(
    ///         vec!["doc one".into(), "doc two".into()],
    ///         Some(model::VOYAGE_3_LARGE),
    ///         Some("document"),
    ///         Some(true),
    ///         None,
    ///     )
    ///     .await?;
    /// assert_eq!(embed.embeddings.len(), 2);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn embed(
        &self,
        input: Vec<String>,
        embed_model: Option<&str>,
        input_type: Option<&str>,
        truncation: Option<bool>,
        output_dimension: Option<u32>,
    ) -> Result<Embed, Error> {
        let payload = EmbedInput {
            input,
            model: embed_model.unwrap_or(model::VOYAGE).to_string(),
            input_type: input_type.map(|s| s.to_string()),
            truncation,
            output_dimension,
        };

        let url = format!("{}/{}/embeddings", self.host, self.version);
        let response = self.http.post(&url).json(&payload).send().await?;

        let status = response.status();
        let body = response.text().await?;

        if !status.is_success() {
            return Err(Error::RequestError {
                status: status.as_u16(),
                body,
            });
        }

        Ok(Embed::parse(&body)?)
    }

    /// Reranks documents by relevance to a query.
    ///
    /// # Arguments
    ///
    /// * `query` — The search query.
    /// * `documents` — Documents to rerank.
    /// * `rerank_model` — Model name (defaults to [`model::RERANK`]).
    /// * `top_k` — Return only the top K results (optional).
    /// * `truncation` — Truncate long inputs (optional).
    ///
    /// # Errors
    ///
    /// Returns [`Error::RequestError`] on non-2xx responses, [`Error::Http`]
    /// on network failures, or [`Error::Json`] on parse failures.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use mongodb_voyageai::Client;
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), mongodb_voyageai::Error> {
    /// let client = Client::with_api_key("pa-...")?;
    ///
    /// let rerank = client
    ///     .rerank(
    ///         "Who fixes pipes?",
    ///         vec![
    ///             "Paul is a plumber.".into(),
    ///             "John is a musician.".into(),
    ///         ],
    ///         None,
    ///         Some(1), // top 1 result only
    ///         None,
    ///     )
    ///     .await?;
    ///
    /// println!("best: index={} score={}", rerank.results[0].index, rerank.results[0].relevance_score);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn rerank(
        &self,
        query: &str,
        documents: Vec<String>,
        rerank_model: Option<&str>,
        top_k: Option<u32>,
        truncation: Option<bool>,
    ) -> Result<Rerank, Error> {
        let payload = RerankInput {
            query: query.to_string(),
            documents,
            model: rerank_model.unwrap_or(model::RERANK).to_string(),
            top_k,
            truncation,
        };

        let url = format!("{}/{}/rerank", self.host, self.version);
        let response = self.http.post(&url).json(&payload).send().await?;

        let status = response.status();
        let body = response.text().await?;

        if !status.is_success() {
            return Err(Error::RequestError {
                status: status.as_u16(),
                body,
            });
        }

        Ok(Rerank::parse(&body)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn missing_api_key() {
        let config = Config {
            api_key: None,
            ..Config::default()
        };
        assert!(matches!(Client::new(&config), Err(Error::MissingApiKey)));
    }

    #[test]
    fn with_api_key() {
        assert!(Client::with_api_key("test_key_12345").is_ok());
    }

    #[test]
    fn with_config_and_timeout() {
        let config = Config {
            api_key: Some("key".into()),
            host: "https://custom.api.com".into(),
            version: "v2".into(),
            timeout: Some(Duration::from_secs(60)),
        };
        assert!(Client::new(&config).is_ok());
    }

    #[test]
    fn debug_masks_api_key() {
        let debug = format!("{:?}", Client::with_api_key("fake_api_key").unwrap());
        assert!(debug.contains("fake_***"));
        assert!(!debug.contains("fake_api_key"));
    }

    #[test]
    fn debug_masks_short_key() {
        let debug = format!("{:?}", Client::with_api_key("abc").unwrap());
        assert!(debug.contains("***"));
        assert!(!debug.contains("abc\""));
    }

    #[test]
    fn debug_shows_host_and_version() {
        let config = Config {
            api_key: Some("key123456".into()),
            host: "https://my-proxy.example.com".into(),
            version: "v2".into(),
            timeout: None,
        };
        let debug = format!("{:?}", Client::new(&config).unwrap());
        assert!(debug.contains("https://my-proxy.example.com"));
        assert!(debug.contains("v2"));
    }

    #[test]
    fn clone_client() {
        let client = Client::with_api_key("clone_test_key").unwrap();
        let cloned = client.clone();
        assert_eq!(format!("{:?}", client), format!("{:?}", cloned));
    }

    #[test]
    fn error_missing_api_key_display() {
        let msg = format!("{}", Error::MissingApiKey);
        assert!(msg.contains("api_key is required"));
        assert!(msg.contains("VOYAGEAI_API_KEY"));
    }

    #[test]
    fn error_request_error_display() {
        let err = Error::RequestError {
            status: 401,
            body: "Unauthorized".into(),
        };
        let msg = format!("{err}");
        assert!(msg.contains("401"));
        assert!(msg.contains("Unauthorized"));
    }

    #[test]
    fn error_request_error_display_500() {
        let err = Error::RequestError {
            status: 500,
            body: r#"{"error":"internal server error"}"#.into(),
        };
        let msg = format!("{err}");
        assert!(msg.contains("500"));
        assert!(msg.contains("internal server error"));
    }

    #[test]
    fn error_json_display() {
        let json_err = serde_json::from_str::<crate::Usage>("bad").unwrap_err();
        let msg = format!("{}", Error::Json(json_err));
        assert!(!msg.is_empty());
    }

    #[test]
    fn embed_input_serialization_minimal() {
        let input = EmbedInput {
            input: vec!["hello".into()],
            model: "voyage-3.5".into(),
            input_type: None,
            truncation: None,
            output_dimension: None,
        };
        let json = serde_json::to_value(&input).unwrap();
        assert_eq!(json["model"], "voyage-3.5");
        assert!(json.get("input_type").is_none());
        assert!(json.get("truncation").is_none());
        assert!(json.get("output_dimension").is_none());
    }

    #[test]
    fn embed_input_serialization_full() {
        let input = EmbedInput {
            input: vec!["a".into(), "b".into()],
            model: "voyage-3-large".into(),
            input_type: Some("document".into()),
            truncation: Some(true),
            output_dimension: Some(256),
        };
        let json = serde_json::to_value(&input).unwrap();
        assert_eq!(json["input_type"], "document");
        assert_eq!(json["truncation"], true);
        assert_eq!(json["output_dimension"], 256);
    }

    #[test]
    fn rerank_input_serialization_minimal() {
        let input = RerankInput {
            query: "search".into(),
            documents: vec!["doc1".into()],
            model: "rerank-2".into(),
            top_k: None,
            truncation: None,
        };
        let json = serde_json::to_value(&input).unwrap();
        assert_eq!(json["query"], "search");
        assert!(json.get("top_k").is_none());
    }

    #[test]
    fn rerank_input_serialization_full() {
        let input = RerankInput {
            query: "q".into(),
            documents: vec!["a".into(), "b".into()],
            model: "rerank-2-lite".into(),
            top_k: Some(5),
            truncation: Some(false),
        };
        let json = serde_json::to_value(&input).unwrap();
        assert_eq!(json["top_k"], 5);
        assert_eq!(json["truncation"], false);
    }
}
