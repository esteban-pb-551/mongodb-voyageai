//! # voyageai
//!
//! An async Rust client for the [VoyageAI](https://www.voyageai.com) API —
//! generate embeddings and rerank documents.
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use mongodb_voyageai::{Client, model};
//!
//! # #[tokio::main]
//! # async fn main() -> Result<(), mongodb_voyageai::Error> {
//! // Reads VOYAGEAI_API_KEY from the environment
//! let client = Client::from_env();
//!
//! let embed = client
//!     .embed(vec!["Hello, world!"])
//!     .model(model::VOYAGE_4_LITE)
//!     .input_type("document")
//!     .send()
//!     .await?;
//!
//! println!("model: {}", embed.model);
//! println!("dimensions: {}", embed.embedding(0).unwrap().len());
//! # Ok(())
//! # }
//! ```
//!
//! ## Embeddings
//!
//! ```rust,no_run
//! # use mongodb_voyageai::{Client, model, OutputDtype};
//! # #[tokio::main]
//! # async fn main() -> Result<(), mongodb_voyageai::Error> {
//! let client = Client::from_env();
//!
//! // Single embedding
//! let embed = client
//!     .embed(vec!["A quick brown fox."])
//!     .send()
//!     .await?;
//! let vector = embed.embedding(0).unwrap();
//!
//! // Multiple embeddings with a specific model
//! let embed = client
//!     .embed(vec!["doc one", "doc two"])
//!     .model(model::VOYAGE_3)
//!     .input_type("document")
//!     .send()
//!     .await?;
//!
//! // With quantization for 4× storage reduction
//! let embed = client
//!     .embed(vec!["efficient storage"])
//!     .model(model::VOYAGE_3_LARGE)
//!     .output_dimension(512)
//!     .output_dtype(OutputDtype::Int8)
//!     .send()
//!     .await?;
//!
//! assert_eq!(embed.embeddings.len(), 1);
//! # Ok(())
//! # }
//! ```
//!
//! ## Reranking
//!
//! ```rust,no_run
//! # use mongodb_voyageai::{Client, Config};
//! # #[tokio::main]
//! # async fn main() -> Result<(), mongodb_voyageai::Error> {
//! let client = Client::with_api_key("pa-...")?;
//!
//! let rerank = client
//!     .rerank(
//!         "Who fixes pipes?",
//!         vec!["Paul is a plumber.", "John is a musician."]
//!     )
//!     .send()
//!     .await?;
//!
//! println!("best match: index={}", rerank.results[0].index);
//! # Ok(())
//! # }
//! ```
//!
//! ## Configuration
//!
//! ```rust
//! use std::time::Duration;
//! use mongodb_voyageai::Config;
//!
//! let config = Config {
//!     api_key: Some("pa-...".into()),
//!     host: "https://api.voyageai.com".into(),
//!     version: "v1".into(),
//!     timeout: Some(Duration::from_secs(30)),
//! };
//! ```
//!
//! ## Error Handling
//!
//! ```rust,no_run
//! # use mongodb_voyageai::{Client, Error};
//! # #[tokio::main]
//! # async fn main() {
//! # let client = Client::with_api_key("pa-...").unwrap();
//! match client.embed(vec!["hello"]).send().await {
//!     Ok(embed) => println!("{} embeddings", embed.embeddings.len()),
//!     Err(Error::MissingApiKey) => eprintln!("set VOYAGEAI_API_KEY"),
//!     Err(Error::RequestError { status, body }) => eprintln!("HTTP {status}: {body}"),
//!     Err(Error::Http(e)) => eprintln!("network: {e}"),
//!     Err(Error::Json(e)) => eprintln!("parse: {e}"),
//! }
//! # }
//! ```

pub mod client;
pub mod config;
pub mod embed;
pub mod model;
pub mod output_dtype;
pub mod rerank;
pub mod reranking;
pub mod usage;
pub mod chunk;
pub mod pairwise;

pub use client::{Client, Error};
pub use config::Config;
pub use embed::Embed;
pub use output_dtype::OutputDtype;
pub use rerank::Rerank;
pub use reranking::Reranking;
pub use usage::Usage;
