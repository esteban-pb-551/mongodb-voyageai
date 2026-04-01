//! HTTP client for the VoyageAI API.

use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue};
use serde::Serialize;

use crate::config::Config;
use crate::context::ContextualizedEmbed;
use crate::embed::Embed;
use crate::model;
use crate::output_dtype::OutputDtype;
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

// ─── IntoStringVec ───────────────────────────────────────────────────────────

/// Conversion trait that allows [`Client::embed`] and [`Client::rerank`] to
/// accept inputs in multiple forms without forcing the caller to allocate or
/// move data unnecessarily.
///
/// Implemented for:
///
/// | Type              | Behaviour                                     |
/// |-------------------|-----------------------------------------------|
/// | `&str`            | Wraps the single string in a one-element vec. |
/// | `String`          | Wraps the single string in a one-element vec. |
/// | `&[S]`            | Borrows a slice — original binding stays live.|
/// | `Vec<S>`          | Consumes the vec — no extra allocation.       |
///
/// Where `S: AsRef<str>` covers both `&str` and `String` for the slice/vec
/// variants.
///
/// # Examples
///
/// ```rust
/// use mongodb_voyageai::client::IntoStringVec;
///
/// // Single &str
/// assert_eq!("hello".into_string_vec(), vec!["hello"]);
///
/// // Single String
/// assert_eq!("world".to_string().into_string_vec(), vec!["world"]);
///
/// // Slice of &str (original stays borrowed, not moved)
/// let docs = vec!["a", "b"];
/// assert_eq!(docs.as_slice().into_string_vec(), vec!["a", "b"]);
///
/// // Vec<&str> (consumed, no extra allocation needed by the caller)
/// assert_eq!(vec!["x", "y"].into_string_vec(), vec!["x", "y"]);
/// ```
pub trait IntoStringVec {
    /// Converts `self` into a `Vec<String>`.
    fn into_string_vec(self) -> Vec<String>;
}

/// Wraps a single `&str` in a one-element `Vec<String>`.
impl IntoStringVec for &str {
    fn into_string_vec(self) -> Vec<String> {
        vec![self.to_owned()]
    }
}

/// Wraps a single `String` in a one-element `Vec<String>`.
impl IntoStringVec for String {
    fn into_string_vec(self) -> Vec<String> {
        vec![self]
    }
}

/// Borrows a slice and copies each element into a new `Vec<String>`.
///
/// Because this implementation takes a shared reference, the original
/// binding is **not** moved and can be used after the call.
///
/// ```rust
/// use mongodb_voyageai::client::IntoStringVec;
///
/// let docs = vec!["first", "second"];
/// let _ = docs.as_slice().into_string_vec();
/// // `docs` is still accessible here
/// println!("{}", docs[0]);
/// ```
impl<S: AsRef<str>> IntoStringVec for &[S] {
    fn into_string_vec(self) -> Vec<String> {
        self.iter().map(|s| s.as_ref().to_owned()).collect()
    }
}

/// Borrows a `Vec<S>` without moving it.
///
/// Delegates to the `&[S]` implementation so that `&vec` and `&slice` behave
/// identically. The original `Vec` remains accessible after the call.
///
/// ```rust
/// use mongodb_voyageai::client::IntoStringVec;
///
/// let docs = vec!["a", "b"];
/// let _ = (&docs).into_string_vec();
/// // `docs` is still accessible here
/// println!("{}", docs[0]);
/// ```
impl<S: AsRef<str>> IntoStringVec for &Vec<S> {
    fn into_string_vec(self) -> Vec<String> {
        self.as_slice().into_string_vec()
    }
}

/// Consumes a `Vec<S>` and converts each element into a `String`.
///
/// Accepts both `Vec<&str>` and `Vec<String>`. When the element type is
/// already `String`, the conversion is a no-op move.
///
/// ```rust
/// use mongodb_voyageai::client::IntoStringVec;
///
/// let v: Vec<String> = vec!["a".to_string(), "b".to_string()];
/// assert_eq!(v.into_string_vec(), vec!["a", "b"]);
/// ```
impl<S: AsRef<str>> IntoStringVec for Vec<S> {
    fn into_string_vec(self) -> Vec<String> {
        self.into_iter().map(|s| s.as_ref().to_owned()).collect()
    }
}

// ─── IntoVecVecString ────────────────────────────────────────────────────────

/// Conversion trait for contextualized embeddings that accepts nested vectors
/// in multiple forms.
///
/// Implemented for:
///
/// | Type                  | Behaviour                                     |
/// |-----------------------|-----------------------------------------------|
/// | `Vec<Vec<&str>>`      | Converts each inner vec to `Vec<String>`      |
/// | `Vec<Vec<String>>`    | Consumes the vec — no extra allocation        |
/// | `&Vec<Vec<&str>>`     | Borrows and converts to `Vec<Vec<String>>`    |
/// | `&Vec<Vec<String>>`   | Borrows and converts to `Vec<Vec<String>>`    |
///
/// # Examples
///
/// ```rust
/// use mongodb_voyageai::client::IntoVecVecString;
///
/// // Vec<Vec<&str>>
/// let inputs = vec![vec!["a", "b"], vec!["c"]];
/// assert_eq!(
///     inputs.into_vec_vec_string(),
///     vec![vec!["a".to_string(), "b".to_string()], vec!["c".to_string()]]
/// );
///
/// // Vec<Vec<String>>
/// let inputs = vec![vec!["x".to_string()], vec!["y".to_string()]];
/// let result = inputs.into_vec_vec_string();
/// assert_eq!(result, vec![vec!["x"], vec!["y"]]);
///
/// // &Vec<Vec<&str>> - original stays accessible
/// let inputs = vec![vec!["a", "b"], vec!["c"]];
/// let result = (&inputs).into_vec_vec_string();
/// assert_eq!(result, vec![vec!["a".to_string(), "b".to_string()], vec!["c".to_string()]]);
/// // inputs is still accessible here
/// assert_eq!(inputs[0][0], "a");
/// ```
pub trait IntoVecVecString {
    /// Converts `self` into a `Vec<Vec<String>>`.
    fn into_vec_vec_string(self) -> Vec<Vec<String>>;
}

/// Converts `Vec<Vec<&str>>` into `Vec<Vec<String>>`.
impl IntoVecVecString for Vec<Vec<&str>> {
    fn into_vec_vec_string(self) -> Vec<Vec<String>> {
        self.into_iter()
            .map(|inner| inner.into_iter().map(|s| s.to_owned()).collect())
            .collect()
    }
}

/// Converts `Vec<Vec<String>>` into `Vec<Vec<String>>` (no-op move).
impl IntoVecVecString for Vec<Vec<String>> {
    fn into_vec_vec_string(self) -> Vec<Vec<String>> {
        self
    }
}

/// Borrows `&Vec<Vec<&str>>` and converts to `Vec<Vec<String>>`.
///
/// The original binding is **not** moved and can be used after the call.
///
/// ```rust
/// use mongodb_voyageai::client::IntoVecVecString;
///
/// let inputs = vec![vec!["a", "b"], vec!["c"]];
/// let _ = (&inputs).into_vec_vec_string();
/// // `inputs` is still accessible here
/// println!("{}", inputs[0][0]);
/// ```
impl IntoVecVecString for &Vec<Vec<&str>> {
    fn into_vec_vec_string(self) -> Vec<Vec<String>> {
        self.iter()
            .map(|inner| inner.iter().map(|s| s.to_string()).collect())
            .collect()
    }
}

/// Borrows `&Vec<Vec<String>>` and converts to `Vec<Vec<String>>`.
///
/// The original binding is **not** moved and can be used after the call.
///
/// ```rust
/// use mongodb_voyageai::client::IntoVecVecString;
///
/// let inputs = vec![vec!["a".to_string()], vec!["b".to_string()]];
/// let _ = (&inputs).into_vec_vec_string();
/// // `inputs` is still accessible here
/// println!("{}", inputs[0][0]);
/// ```
impl IntoVecVecString for &Vec<Vec<String>> {
    fn into_vec_vec_string(self) -> Vec<Vec<String>> {
        self.iter()
            .map(|inner| inner.iter().map(|s| s.to_string()).collect())
            .collect()
    }
}

// ─── Request payloads ────────────────────────────────────────────────────────

/// The JSON payload sent to the `/embeddings` endpoint.
///
/// # Examples
///
/// ```rust
/// use mongodb_voyageai::client::EmbedInput;
/// use mongodb_voyageai::OutputDtype;
///
/// let input = EmbedInput {
///     input: vec!["hello".into()],
///     model: "voyage-4".into(),
///     input_type: Some("query".into()),
///     truncation: None,
///     output_dimension: None,
///     output_dtype: None,
/// };
///
/// let json = serde_json::to_value(&input).unwrap();
/// assert_eq!(json["model"], "voyage-4");
/// assert_eq!(json["input_type"], "query");
/// // None fields are omitted
/// assert!(json.get("truncation").is_none());
///
/// // With quantization
/// let input_quantized = EmbedInput {
///     input: vec!["hello".into()],
///     model: "voyage-3-large".into(),
///     input_type: None,
///     truncation: None,
///     output_dimension: Some(512),
///     output_dtype: Some(OutputDtype::Int8),
/// };
/// let json = serde_json::to_value(&input_quantized).unwrap();
/// assert_eq!(json["output_dtype"], "int8");
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
    /// Output data type for quantization (float, int8, uint8, binary, ubinary).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_dtype: Option<OutputDtype>,
}

/// The JSON payload sent to the `/rerank` endpoint.
///
/// # Limits
///
/// - Maximum documents: 1,000
/// - Query token limits:
///   - rerank-2.5/2.5-lite: 8,000 tokens
///   - rerank-2: 4,000 tokens
///   - rerank-2-lite/rerank-1: 2,000 tokens
///   - rerank-lite-1: 1,000 tokens
/// - Query + Document token limits:
///   - rerank-2.5/2.5-lite: 32,000 tokens
///   - rerank-2: 16,000 tokens
///   - rerank-2-lite/rerank-1: 8,000 tokens
///   - rerank-lite-1: 4,000 tokens
/// - Total tokens (query tokens × num documents + sum of all document tokens):
///   - rerank-2.5/2.5-lite/rerank-2/rerank-2-lite: 600K
///   - rerank-1/rerank-lite-1: 300K
///
/// # Examples
///
/// ```rust
/// use mongodb_voyageai::client::RerankInput;
///
/// let input = RerankInput {
///     query: "search query".into(),
///     documents: vec!["doc A".into(), "doc B".into()],
///     model: "rerank-2.5".into(),
///     top_k: Some(1),
///     truncation: None,  // Defaults to true on API side
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
    /// Defaults to true on the API side when not specified.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncation: Option<bool>,
}

// ─── Client ──────────────────────────────────────────────────────────────────

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

    /// Creates a new client from environment variables.
    ///
    /// This is a convenience method that returns a [`Result`], allowing you to
    /// handle the error if the API key is not set.
    ///
    /// # Errors
    ///
    /// Returns [`Error::MissingApiKey`] if `VOYAGEAI_API_KEY` is not set.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use mongodb_voyageai::Client;
    ///
    /// let client = Client::try_from_env()?;
    /// # Ok::<(), mongodb_voyageai::Error>(())
    /// ```
    pub fn try_from_env() -> Result<Self, Error> {
        Self::new(&Config::new())
    }

    /// Creates a new client from environment variables.
    ///
    /// This is a convenience method for when you know the API key is configured.
    /// If you need error handling, use [`Client::try_from_env`] instead.
    ///
    /// # Panics
    ///
    /// Panics if `VOYAGEAI_API_KEY` environment variable is not set.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use mongodb_voyageai::Client;
    ///
    /// let client = Client::from_env();
    /// ```
    pub fn from_env() -> Self {
        Self::try_from_env().expect("Failed to create client: ensure VOYAGEAI_API_KEY is set")
    }

    /// Creates an [`EmbedBuilder`] for the given input texts.
    ///
    /// The `input` argument accepts any type that implements [`IntoStringVec`],
    /// which covers the following without extra allocation on the caller side:
    ///
    /// | Argument type  | Example                                    |
    /// |----------------|--------------------------------------------|
    /// | `&str`         | `client.embed("single text")`              |
    /// | `String`       | `client.embed(owned_string)`               |
    /// | `&[S]`         | `client.embed(&docs)` — `docs` stays live  |
    /// | `Vec<S>`       | `client.embed(vec!["a", "b"])`             |
    ///
    /// Use the builder's setter methods to configure the request, then call
    /// [`send`](EmbedBuilder::send) to execute it.
    ///
    /// # Arguments
    ///
    /// * `input` — One or more texts to embed. See the table above for
    ///   accepted types.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use mongodb_voyageai::{Client, model};
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), mongodb_voyageai::Error> {
    /// let client = Client::with_api_key("pa-...")?;
    ///
    /// // Single &str
    /// let embed = client.embed("Hello world").send().await?;
    ///
    /// // Slice — original vec stays available after the call
    /// let docs = vec!["first doc", "second doc"];
    /// let embed = client
    ///     .embed(&docs)
    ///     .model(model::VOYAGE_3_LARGE)
    ///     .input_type("document")
    ///     .send()
    ///     .await?;
    /// println!("original still accessible: {}", docs[0]);
    ///
    /// // Vec<&str> — consumed but no .into() boilerplate needed
    /// let embed = client
    ///     .embed(vec!["a", "b"])
    ///     .input_type("query")
    ///     .send()
    ///     .await?;
    ///
    /// assert_eq!(embed.embeddings.len(), 2);
    /// # Ok(())
    /// # }
    /// ```
    pub fn embed<I: IntoStringVec>(&self, input: I) -> EmbedBuilder<'_> {
        EmbedBuilder::new(self, input.into_string_vec())
    }

    /// Creates a [`RerankBuilder`] for the given query and documents.
    ///
    /// Both `query` and `documents` follow the same flexible input rules.
    /// `query` accepts any `&str` or `String`. `documents` accepts any type
    /// that implements [`IntoStringVec`]:
    ///
    /// | Argument type  | Example                                        |
    /// |----------------|------------------------------------------------|
    /// | `&str`         | `client.rerank("q", "single doc")`             |
    /// | `String`       | `client.rerank("q", owned_string)`             |
    /// | `&[S]`         | `client.rerank("q", &docs)` — `docs` stays live|
    /// | `Vec<S>`       | `client.rerank("q", vec!["a", "b"])`           |
    ///
    /// Use the builder's setter methods to configure the request, then call
    /// [`send`](RerankBuilder::send) to execute it.
    ///
    /// # Arguments
    ///
    /// * `query`     — The search query.
    /// * `documents` — The documents to rerank. See the table above for
    ///   accepted types.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use mongodb_voyageai::Client;
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), mongodb_voyageai::Error> {
    /// let client = Client::with_api_key("pa-...")?;
    ///
    /// // Slice — original vec stays available after the call
    /// let docs = vec!["Paul is a plumber.", "John is a musician."];
    /// let rerank = client
    ///     .rerank("Who fixes pipes?", &docs)
    ///     .top_k(1)
    ///     .send()
    ///     .await?;
    /// println!("original still accessible: {}", docs[0]);
    ///
    /// // Vec<&str> — no .into() boilerplate
    /// let rerank = client
    ///     .rerank(
    ///         "Who fixes pipes?",
    ///         vec!["Paul is a plumber.", "John is a musician."],
    ///     )
    ///     .send()
    ///     .await?;
    ///
    /// println!(
    ///     "best: index={} score={}",
    ///     rerank.results[0].index,
    ///     rerank.results[0].relevance_score,
    /// );
    /// # Ok(())
    /// # }
    /// ```
    pub fn rerank<'a, I: IntoStringVec>(
        &'a self,
        query: &'a str,
        documents: I,
    ) -> RerankBuilder<'a> {
        RerankBuilder::new(self, query, documents.into_string_vec())
    }

    /// Creates a [`ContextualizedEmbedBuilder`] for the given inputs.
    ///
    /// The `inputs` argument is a list of lists, where each inner list contains
    /// texts to be vectorized together. Most commonly, each inner list contains
    /// chunks from a single document, ordered by their position in the document.
    ///
    /// The `inputs` argument accepts any type that implements [`IntoVecVecString`],
    /// which covers the following without extra allocation on the caller side:
    ///
    /// | Argument type         | Example                                    |
    /// |-----------------------|--------------------------------------------|
    /// | `Vec<Vec<&str>>`      | `client.contextualized_embed(chunks)`      |
    /// | `Vec<Vec<String>>`    | `client.contextualized_embed(owned)`       |
    /// | `&Vec<Vec<&str>>`     | `client.contextualized_embed(&chunks)`     |
    /// | `&Vec<Vec<String>>`   | `client.contextualized_embed(&owned)`      |
    ///
    /// # Arguments
    ///
    /// * `inputs` — A list of lists of texts. Each inner list represents a set
    ///   of text elements that will be embedded together with context.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use mongodb_voyageai::{Client, model};
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), mongodb_voyageai::Error> {
    /// let client = Client::with_api_key("pa-...")?;
    ///
    /// // Document chunks - original vec stays available
    /// let chunks = vec![
    ///     vec!["doc 1 chunk 1", "doc 1 chunk 2"],
    ///     vec!["doc 2 chunk 1", "doc 2 chunk 2"],
    /// ];
    /// let embed = client
    ///     .contextualized_embed(&chunks)
    ///     .model("voyage-context-3")
    ///     .input_type("document")
    ///     .send()
    ///     .await?;
    /// println!("original still accessible: {}", chunks[0][0]);
    ///
    /// // Single query (each inner list should contain one query)
    /// let queries = vec![vec!["What is Rust?"]];
    /// let query_embed = client
    ///     .contextualized_embed(queries)
    ///     .input_type("query")
    ///     .send()
    ///     .await?;
    ///
    /// assert_eq!(embed.results.len(), 2);
    /// # Ok(())
    /// # }
    /// ```
    pub fn contextualized_embed<I: IntoVecVecString>(
        &self,
        inputs: I,
    ) -> ContextualizedEmbedBuilder<'_> {
        ContextualizedEmbedBuilder::new(self, inputs.into_vec_vec_string())
    }
}

// ─── EmbedBuilder ────────────────────────────────────────────────────────────

/// A builder for constructing and sending embedding requests.
///
/// Created via [`Client::embed`]. Chain setter methods to configure optional
/// parameters, then call [`send`](EmbedBuilder::send) to execute the request.
///
/// # Examples
///
/// ```rust,no_run
/// # use mongodb_voyageai::{Client, model, OutputDtype};
/// # #[tokio::main]
/// # async fn main() -> Result<(), mongodb_voyageai::Error> {
/// let client = Client::with_api_key("pa-...")?;
///
/// // Minimal — single &str, all defaults
/// let embed = client.embed("Hello world").send().await?;
///
/// // Slice — docs stays live after the call
/// let docs = vec!["doc one", "doc two"];
/// let embed = client
///     .embed(&docs)
///     .model(model::VOYAGE_3_LARGE)
///     .input_type("document")
///     .truncation(true)
///     .output_dimension(512)
///     .output_dtype(OutputDtype::Int8)
///     .send()
///     .await?;
/// // docs is still accessible here
/// # Ok(())
/// # }
/// ```
pub struct EmbedBuilder<'a> {
    client: &'a Client,
    input: Vec<String>,
    model: &'a str,
    input_type: Option<&'a str>,
    truncation: Option<bool>,
    output_dimension: Option<u32>,
    output_dtype: Option<OutputDtype>,
}

impl<'a> EmbedBuilder<'a> {
    fn new(client: &'a Client, input: Vec<String>) -> Self {
        Self {
            client,
            input,
            model: model::VOYAGE,
            input_type: None,
            truncation: None,
            output_dimension: None,
            output_dtype: None,
        }
    }

    /// Overrides the embedding model.
    ///
    /// Defaults to [`model::VOYAGE`] when not set.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use mongodb_voyageai::{Client, model};
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), mongodb_voyageai::Error> {
    /// # let client = Client::with_api_key("pa-...")?;
    /// let embed = client
    ///     .embed("text")
    ///     .model(model::VOYAGE_3_LARGE)
    ///     .send()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn model(mut self, model: &'a str) -> Self {
        self.model = model;
        self
    }

    /// Sets the input type hint for the model.
    ///
    /// Accepted values are `"query"` and `"document"`. Providing the correct
    /// type improves retrieval quality for asymmetric search tasks.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use mongodb_voyageai::Client;
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), mongodb_voyageai::Error> {
    /// # let client = Client::with_api_key("pa-...")?;
    /// // Embed a search query
    /// let query_embed = client
    ///     .embed("what is rust ownership?")
    ///     .input_type("query")
    ///     .send()
    ///     .await?;
    ///
    /// // Embed documents to store
    /// let doc_embed = client
    ///     .embed("Ownership is Rust's memory model...")
    ///     .input_type("document")
    ///     .send()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn input_type(mut self, input_type: &'a str) -> Self {
        self.input_type = Some(input_type);
        self
    }

    /// Enables or disables truncation of inputs that exceed the model's token limit.
    ///
    /// When `true`, long inputs are silently truncated instead of returning an
    /// error. Defaults to the API's own default when not set.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use mongodb_voyageai::Client;
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), mongodb_voyageai::Error> {
    /// # let client = Client::with_api_key("pa-...")?;
    /// let embed = client
    ///     .embed("A very long document...")
    ///     .truncation(true)
    ///     .send()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn truncation(mut self, truncation: bool) -> Self {
        self.truncation = Some(truncation);
        self
    }

    /// Reduces the output embedding to the given number of dimensions.
    ///
    /// Smaller dimensions lower storage and compute costs at a potential
    /// trade-off in retrieval accuracy. Only supported by models that
    /// advertise Matryoshka Representation Learning (MRL).
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use mongodb_voyageai::{Client, model};
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), mongodb_voyageai::Error> {
    /// # let client = Client::with_api_key("pa-...")?;
    /// let embed = client
    ///     .embed("compact representation")
    ///     .model(model::VOYAGE_3_LARGE)
    ///     .output_dimension(512)
    ///     .send()
    ///     .await?;
    ///
    /// assert_eq!(embed.embedding(0).unwrap().len(), 512);
    /// # Ok(())
    /// # }
    /// ```
    pub fn output_dimension(mut self, dim: u32) -> Self {
        self.output_dimension = Some(dim);
        self
    }

    /// Sets the output data type for quantization.
    ///
    /// Quantization dramatically reduces storage costs while maintaining
    /// quality. Only supported by models with Quantization-Aware Training
    /// (voyage-3-large, voyage-4 series).
    ///
    /// # Storage Savings
    ///
    /// For a 512-dimensional embedding:
    /// - `Float` (default): 2048 bytes
    /// - `Int8` / `Uint8`: 512 bytes (4× compression)
    /// - `Binary` / `Ubinary`: 64 bytes (32× compression)
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use mongodb_voyageai::{Client, model, OutputDtype};
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), mongodb_voyageai::Error> {
    /// # let client = Client::with_api_key("pa-...")?;
    /// // 4× storage reduction with minimal quality loss
    /// let embed = client
    ///     .embed("efficient storage")
    ///     .model(model::VOYAGE_3_LARGE)
    ///     .output_dimension(512)
    ///     .output_dtype(OutputDtype::Int8)
    ///     .send()
    ///     .await?;
    ///
    /// // 32× compression for maximum efficiency
    /// let embed_binary = client
    ///     .embed("ultra compact")
    ///     .model(model::VOYAGE_4_LARGE)
    ///     .output_dimension(512)
    ///     .output_dtype(OutputDtype::Binary)
    ///     .send()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn output_dtype(mut self, dtype: OutputDtype) -> Self {
        self.output_dtype = Some(dtype);
        self
    }

    /// Sends the embedding request and returns the result.
    ///
    /// # Errors
    ///
    /// - [`Error::RequestError`] — API returned a non-2xx status.
    /// - [`Error::Http`] — Network or transport failure.
    /// - [`Error::Json`] — Response body could not be parsed.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use mongodb_voyageai::Client;
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), mongodb_voyageai::Error> {
    /// # let client = Client::with_api_key("pa-...")?;
    /// let embed = client.embed("Hello world").send().await?;
    /// println!("dimensions: {}", embed.embedding(0).unwrap().len());
    /// # Ok(())
    /// # }
    /// ```
    pub async fn send(self) -> Result<Embed, Error> {
        let payload = EmbedInput {
            input: self.input,
            model: self.model.to_string(),
            input_type: self.input_type.map(|s| s.to_string()),
            truncation: self.truncation,
            output_dimension: self.output_dimension,
            output_dtype: self.output_dtype,
        };

        let url = format!("{}/{}/embeddings", self.client.host, self.client.version);
        let response = self.client.http.post(&url).json(&payload).send().await?;

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
}

// ─── RerankBuilder ───────────────────────────────────────────────────────────

/// A builder for constructing and sending rerank requests.
///
/// Created via [`Client::rerank`]. Chain setter methods to configure optional
/// parameters, then call [`send`](RerankBuilder::send) to execute the request.
///
/// # Examples
///
/// ```rust,no_run
/// # use mongodb_voyageai::{Client, model};
/// # #[tokio::main]
/// # async fn main() -> Result<(), mongodb_voyageai::Error> {
/// let client = Client::with_api_key("pa-...")?;
///
/// // Slice — docs stays live after the call
/// let docs = vec!["Paul is a plumber.", "John is a musician."];
/// let rerank = client
///     .rerank("Who fixes pipes?", &docs)
///     .top_k(1)
///     .send()
///     .await?;
/// println!("original still accessible: {}", docs[0]);
///
/// // Fully configured with Vec<&str>
/// let rerank = client
///     .rerank(
///         "Who fixes pipes?",
///         vec!["Paul is a plumber.", "John is a musician."],
///     )
///     .model(model::RERANK)
///     .top_k(1)
///     .truncation(true)
///     .send()
///     .await?;
/// # Ok(())
/// # }
/// ```
pub struct RerankBuilder<'a> {
    client: &'a Client,
    query: &'a str,
    documents: Vec<String>,
    model: &'a str,
    top_k: Option<u32>,
    truncation: Option<bool>,
}

impl<'a> RerankBuilder<'a> {
    fn new(client: &'a Client, query: &'a str, documents: Vec<String>) -> Self {
        Self {
            client,
            query,
            documents,
            model: model::RERANK,
            top_k: None,
            truncation: None,
        }
    }

    /// Overrides the reranking model.
    ///
    /// Defaults to [`model::RERANK`] when not set.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use mongodb_voyageai::{Client, model};
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), mongodb_voyageai::Error> {
    /// # let client = Client::with_api_key("pa-...")?;
    /// let rerank = client
    ///     .rerank("query", "doc")
    ///     .model(model::RERANK)
    ///     .send()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn model(mut self, model: &'a str) -> Self {
        self.model = model;
        self
    }

    /// Returns only the top K highest-scoring documents.
    ///
    /// When not set, all documents are returned in ranked order.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use mongodb_voyageai::Client;
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), mongodb_voyageai::Error> {
    /// # let client = Client::with_api_key("pa-...")?;
    /// let rerank = client
    ///     .rerank(
    ///         "Who fixes pipes?",
    ///         vec!["Paul is a plumber.", "John is a musician."],
    ///     )
    ///     .top_k(1)
    ///     .send()
    ///     .await?;
    ///
    /// assert_eq!(rerank.results.len(), 1);
    /// # Ok(())
    /// # }
    /// ```
    pub fn top_k(mut self, top_k: u32) -> Self {
        self.top_k = Some(top_k);
        self
    }

    /// Enables or disables truncation of inputs that exceed the model's token limit.
    ///
    /// When `true`, long inputs are silently truncated instead of returning an
    /// error. When `false`, an error will be raised if inputs exceed limits.
    ///
    /// **Defaults to `true`** when not specified (API default).
    ///
    /// # Token Limits
    ///
    /// - Query: 8,000 tokens (rerank-2.5/2.5-lite), 4,000 (rerank-2),
    ///   2,000 (rerank-2-lite/rerank-1), 1,000 (rerank-lite-1)
    /// - Query + Document: 32,000 tokens (rerank-2.5/2.5-lite),
    ///   16,000 (rerank-2), 8,000 (rerank-2-lite/rerank-1), 4,000 (rerank-lite-1)
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use mongodb_voyageai::Client;
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), mongodb_voyageai::Error> {
    /// # let client = Client::with_api_key("pa-...")?;
    /// // Explicitly disable truncation (will error on long inputs)
    /// let rerank = client
    ///     .rerank("query", "a very long document...")
    ///     .truncation(false)
    ///     .send()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn truncation(mut self, truncation: bool) -> Self {
        self.truncation = Some(truncation);
        self
    }

    /// Sends the rerank request and returns the result.
    ///
    /// # Errors
    ///
    /// - [`Error::RequestError`] — API returned a non-2xx status.
    /// - [`Error::Http`] — Network or transport failure.
    /// - [`Error::Json`] — Response body could not be parsed.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use mongodb_voyageai::Client;
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), mongodb_voyageai::Error> {
    /// # let client = Client::with_api_key("pa-...")?;
    /// let rerank = client
    ///     .rerank(
    ///         "Who fixes pipes?",
    ///         vec!["Paul is a plumber.", "John is a musician."],
    ///     )
    ///     .send()
    ///     .await?;
    ///
    /// println!(
    ///     "best: index={} score={}",
    ///     rerank.results[0].index,
    ///     rerank.results[0].relevance_score,
    /// );
    /// # Ok(())
    /// # }
    /// ```
    pub async fn send(self) -> Result<Rerank, Error> {
        let payload = RerankInput {
            query: self.query.to_string(),
            documents: self.documents,
            model: self.model.to_string(),
            top_k: self.top_k,
            truncation: self.truncation,
        };

        let url = format!("{}/{}/rerank", self.client.host, self.client.version);
        let response = self.client.http.post(&url).json(&payload).send().await?;

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

// ─── ContextualizedEmbedBuilder ──────────────────────────────────────────────

/// A builder for constructing and sending contextualized embedding requests.
///
/// Created via [`Client::contextualized_embed`]. Chain setter methods to configure
/// optional parameters, then call [`send`](ContextualizedEmbedBuilder::send) to
/// execute the request.
///
/// # Examples
///
/// ```rust,no_run
/// # use mongodb_voyageai::{Client, OutputDtype};
/// # #[tokio::main]
/// # async fn main() -> Result<(), mongodb_voyageai::Error> {
/// let client = Client::with_api_key("pa-...")?;
///
/// // Document chunks with context
/// let chunks = vec![
///     vec!["chunk 1 from doc 1", "chunk 2 from doc 1"],
///     vec!["chunk 1 from doc 2"],
/// ];
/// let embed = client
///     .contextualized_embed(chunks)
///     .model("voyage-context-3")
///     .input_type("document")
///     .output_dimension(512)
///     .output_dtype(OutputDtype::Int8)
///     .send()
///     .await?;
///
/// assert_eq!(embed.results.len(), 2);
/// # Ok(())
/// # }
/// ```
pub struct ContextualizedEmbedBuilder<'a> {
    client: &'a Client,
    inputs: Vec<Vec<String>>,
    model: &'a str,
    input_type: Option<&'a str>,
    output_dimension: Option<u32>,
    output_dtype: Option<OutputDtype>,
}

impl<'a> ContextualizedEmbedBuilder<'a> {
    fn new(client: &'a Client, inputs: Vec<Vec<String>>) -> Self {
        Self {
            client,
            inputs,
            model: "voyage-context-3",
            input_type: None,
            output_dimension: None,
            output_dtype: None,
        }
    }

    /// Overrides the contextualized embedding model.
    ///
    /// Defaults to `"voyage-context-3"` when not set.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use mongodb_voyageai::Client;
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), mongodb_voyageai::Error> {
    /// # let client = Client::with_api_key("pa-...")?;
    /// let embed = client
    ///     .contextualized_embed(vec![vec!["text"]])
    ///     .model("voyage-context-3")
    ///     .send()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn model(mut self, model: &'a str) -> Self {
        self.model = model;
        self
    }

    /// Sets the input type hint for the model.
    ///
    /// Accepted values are `"query"` and `"document"`. For queries, each inner
    /// list should contain a single query. For documents, each inner list
    /// typically contains chunks from a single document.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use mongodb_voyageai::Client;
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), mongodb_voyageai::Error> {
    /// # let client = Client::with_api_key("pa-...")?;
    /// // Embed document chunks
    /// let doc_embed = client
    ///     .contextualized_embed(vec![vec!["chunk 1", "chunk 2"]])
    ///     .input_type("document")
    ///     .send()
    ///     .await?;
    ///
    /// // Embed queries
    /// let query_embed = client
    ///     .contextualized_embed(vec![vec!["search query"]])
    ///     .input_type("query")
    ///     .send()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn input_type(mut self, input_type: &'a str) -> Self {
        self.input_type = Some(input_type);
        self
    }

    /// Reduces the output embedding to the given number of dimensions.
    ///
    /// voyage-context-3 supports: 2048, 1024 (default), 512, and 256.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use mongodb_voyageai::Client;
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), mongodb_voyageai::Error> {
    /// # let client = Client::with_api_key("pa-...")?;
    /// let embed = client
    ///     .contextualized_embed(vec![vec!["text"]])
    ///     .output_dimension(512)
    ///     .send()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn output_dimension(mut self, dim: u32) -> Self {
        self.output_dimension = Some(dim);
        self
    }

    /// Sets the output data type for quantization.
    ///
    /// Options: Float (default), Int8, Uint8, Binary, Ubinary.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use mongodb_voyageai::{Client, OutputDtype};
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), mongodb_voyageai::Error> {
    /// # let client = Client::with_api_key("pa-...")?;
    /// let embed = client
    ///     .contextualized_embed(vec![vec!["text"]])
    ///     .output_dimension(512)
    ///     .output_dtype(OutputDtype::Int8)
    ///     .send()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn output_dtype(mut self, dtype: OutputDtype) -> Self {
        self.output_dtype = Some(dtype);
        self
    }

    /// Sends the contextualized embedding request and returns the result.
    ///
    /// # Errors
    ///
    /// - [`Error::RequestError`] — API returned a non-2xx status.
    /// - [`Error::Http`] — Network or transport failure.
    /// - [`Error::Json`] — Response body could not be parsed.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use mongodb_voyageai::Client;
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), mongodb_voyageai::Error> {
    /// # let client = Client::with_api_key("pa-...")?;
    /// let embed = client
    ///     .contextualized_embed(vec![vec!["chunk 1", "chunk 2"]])
    ///     .send()
    ///     .await?;
    /// println!("results: {}", embed.results.len());
    /// # Ok(())
    /// # }
    /// ```
    pub async fn send(self) -> Result<ContextualizedEmbed, Error> {
        let payload = crate::context::ContextualizedEmbedInput {
            inputs: self.inputs,
            model: self.model.to_string(),
            input_type: self.input_type.map(|s| s.to_string()),
            output_dimension: self.output_dimension,
            output_dtype: self.output_dtype,
        };

        let url = format!(
            "{}/{}/contextualizedembeddings",
            self.client.host, self.client.version
        );
        let response = self.client.http.post(&url).json(&payload).send().await?;

        let status = response.status();
        let body = response.text().await?;

        if !status.is_success() {
            return Err(Error::RequestError {
                status: status.as_u16(),
                body,
            });
        }

        Ok(ContextualizedEmbed::parse(&body)?)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    // ── Client construction ──────────────────────────────────────────────────

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

    // ── Error display ────────────────────────────────────────────────────────

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

    // ── IntoStringVec ────────────────────────────────────────────────────────

    #[test]
    fn into_string_vec_from_str() {
        assert_eq!("hello".into_string_vec(), vec!["hello"]);
    }

    #[test]
    fn into_string_vec_from_string() {
        assert_eq!("world".to_string().into_string_vec(), vec!["world"]);
    }

    #[test]
    fn into_string_vec_from_slice_str() {
        let docs = vec!["a", "b", "c"];
        assert_eq!(docs.as_slice().into_string_vec(), vec!["a", "b", "c"]);
        // original still accessible
        assert_eq!(docs[0], "a");
    }

    #[test]
    fn into_string_vec_from_slice_string() {
        let docs = vec!["x".to_string(), "y".to_string()];
        assert_eq!(docs.as_slice().into_string_vec(), vec!["x", "y"]);
        assert_eq!(docs[0], "x");
    }

    #[test]
    fn into_string_vec_from_vec_str() {
        assert_eq!(vec!["p", "q"].into_string_vec(), vec!["p", "q"]);
    }

    #[test]
    fn into_string_vec_from_vec_string() {
        let v = vec!["one".to_string(), "two".to_string()];
        assert_eq!(v.into_string_vec(), vec!["one", "two"]);
    }

    // ── Payload serialization ────────────────────────────────────────────────

    #[test]
    fn embed_input_serialization_minimal() {
        let input = EmbedInput {
            input: vec!["hello".into()],
            model: "voyage-4".into(),
            input_type: None,
            truncation: None,
            output_dimension: None,
            output_dtype: None,
        };
        let json = serde_json::to_value(&input).unwrap();
        assert_eq!(json["model"], "voyage-4");
        assert!(json.get("input_type").is_none());
        assert!(json.get("truncation").is_none());
        assert!(json.get("output_dimension").is_none());
        assert!(json.get("output_dtype").is_none());
    }

    #[test]
    fn embed_input_serialization_full() {
        let input = EmbedInput {
            input: vec!["a".into(), "b".into()],
            model: "voyage-3-large".into(),
            input_type: Some("document".into()),
            truncation: Some(true),
            output_dimension: Some(256),
            output_dtype: Some(OutputDtype::Int8),
        };
        let json = serde_json::to_value(&input).unwrap();
        assert_eq!(json["input_type"], "document");
        assert_eq!(json["truncation"], true);
        assert_eq!(json["output_dimension"], 256);
        assert_eq!(json["output_dtype"], "int8");
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

    // ── EmbedBuilder ─────────────────────────────────────────────────────────

    #[test]
    fn embed_builder_defaults() {
        let client = Client::with_api_key("key12345").unwrap();
        let builder = client.embed("text");
        assert_eq!(builder.model, model::VOYAGE);
        assert!(builder.input_type.is_none());
        assert!(builder.truncation.is_none());
        assert!(builder.output_dimension.is_none());
        assert!(builder.output_dtype.is_none());
    }

    #[test]
    fn embed_builder_accepts_str() {
        let client = Client::with_api_key("key12345").unwrap();
        let builder = client.embed("single");
        assert_eq!(builder.input, vec!["single"]);
    }

    #[test]
    fn embed_builder_accepts_vec_str() {
        let client = Client::with_api_key("key12345").unwrap();
        let builder = client.embed(vec!["a", "b"]);
        assert_eq!(builder.input, vec!["a", "b"]);
    }

    #[test]
    fn embed_builder_accepts_slice_without_move() {
        let client = Client::with_api_key("key12345").unwrap();
        let docs = vec!["x", "y"];
        let builder = client.embed(&docs);
        assert_eq!(builder.input, vec!["x", "y"]);
        // docs is still accessible
        assert_eq!(docs[0], "x");
    }

    #[test]
    fn embed_builder_setters() {
        let client = Client::with_api_key("key12345").unwrap();
        let builder = client
            .embed(vec!["text"])
            .model(model::VOYAGE_3_LARGE)
            .input_type("document")
            .truncation(true)
            .output_dimension(256)
            .output_dtype(OutputDtype::Int8);
        assert_eq!(builder.model, model::VOYAGE_3_LARGE);
        assert_eq!(builder.input_type, Some("document"));
        assert_eq!(builder.truncation, Some(true));
        assert_eq!(builder.output_dimension, Some(256));
        assert_eq!(builder.output_dtype, Some(OutputDtype::Int8));
    }

    // ── RerankBuilder ────────────────────────────────────────────────────────

    #[test]
    fn rerank_builder_defaults() {
        let client = Client::with_api_key("key12345").unwrap();
        let builder = client.rerank("query", vec!["doc"]);
        assert_eq!(builder.model, model::RERANK);
        assert!(builder.top_k.is_none());
        assert!(builder.truncation.is_none());
    }

    #[test]
    fn rerank_builder_accepts_str() {
        let client = Client::with_api_key("key12345").unwrap();
        let builder = client.rerank("q", "single doc");
        assert_eq!(builder.documents, vec!["single doc"]);
    }

    #[test]
    fn rerank_builder_accepts_vec_str() {
        let client = Client::with_api_key("key12345").unwrap();
        let builder = client.rerank("q", vec!["a", "b"]);
        assert_eq!(builder.documents, vec!["a", "b"]);
    }

    #[test]
    fn rerank_builder_accepts_slice_without_move() {
        let client = Client::with_api_key("key12345").unwrap();
        let docs = vec!["p", "q"];
        let builder = client.rerank("query", &docs);
        assert_eq!(builder.documents, vec!["p", "q"]);
        // docs is still accessible
        assert_eq!(docs[0], "p");
    }

    #[test]
    fn rerank_builder_setters() {
        let client = Client::with_api_key("key12345").unwrap();
        let builder = client
            .rerank("query", vec!["doc"])
            .model(model::RERANK)
            .top_k(3)
            .truncation(false);
        assert_eq!(builder.model, model::RERANK);
        assert_eq!(builder.top_k, Some(3));
        assert_eq!(builder.truncation, Some(false));
    }
}
