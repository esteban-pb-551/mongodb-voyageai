//! Client configuration for the VoyageAI API.

use std::env;
use std::time::Duration;

const DEFAULT_HOST: &str = "https://api.voyageai.com";
const DEFAULT_VERSION: &str = "v1";

/// Configuration for the VoyageAI [`Client`](crate::Client).
///
/// All fields default from environment variables when using [`Config::new`] or
/// [`Config::default`].
///
/// # Environment Variables
///
/// | Variable           | Field     | Fallback                    |
/// |--------------------|-----------|-----------------------------|
/// | `VOYAGEAI_API_KEY` | `api_key` | `None`                      |
/// | `VOYAGEAI_HOST`    | `host`    | `https://api.voyageai.com`  |
/// | `VOYAGEAI_VERSION` | `version` | `v1`                        |
///
/// # Examples
///
/// Using environment defaults:
///
/// ```rust
/// use voyageai::Config;
///
/// let config = Config::new();
/// assert_eq!(config.host, "https://api.voyageai.com");
/// assert_eq!(config.version, "v1");
/// ```
///
/// Custom configuration:
///
/// ```rust
/// use std::time::Duration;
/// use voyageai::Config;
///
/// let config = Config {
///     api_key: Some("pa-...".into()),
///     host: "https://custom-proxy.example.com".into(),
///     version: "v1".into(),
///     timeout: Some(Duration::from_secs(30)),
/// };
/// ```
#[derive(Debug, Clone)]
pub struct Config {
    /// API key for authentication. Defaults to `VOYAGEAI_API_KEY` env var.
    pub api_key: Option<String>,
    /// Base URL of the API. Defaults to `https://api.voyageai.com`.
    pub host: String,
    /// API version path segment. Defaults to `v1`.
    pub version: String,
    /// Optional request timeout.
    pub timeout: Option<Duration>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            api_key: env::var("VOYAGEAI_API_KEY").ok(),
            host: env::var("VOYAGEAI_HOST").unwrap_or_else(|_| DEFAULT_HOST.to_string()),
            version: env::var("VOYAGEAI_VERSION").unwrap_or_else(|_| DEFAULT_VERSION.to_string()),
            timeout: None,
        }
    }
}

impl Config {
    /// Creates a new [`Config`] populated from environment variables.
    ///
    /// This is equivalent to [`Config::default()`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use voyageai::Config;
    ///
    /// let config = Config::new();
    /// assert_eq!(config.version, "v1");
    /// ```
    pub fn new() -> Self {
        Self::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults() {
        let config = Config::new();
        assert_eq!(config.host, "https://api.voyageai.com");
        assert_eq!(config.version, "v1");
        assert!(config.timeout.is_none());
    }

    #[test]
    fn default_trait() {
        let config = Config::default();
        assert_eq!(config.host, "https://api.voyageai.com");
        assert_eq!(config.version, "v1");
    }

    #[test]
    fn custom_values() {
        let config = Config {
            api_key: Some("test-key".into()),
            host: "https://custom.host".into(),
            version: "v2".into(),
            timeout: Some(Duration::from_secs(30)),
        };
        assert_eq!(config.api_key, Some("test-key".into()));
        assert_eq!(config.host, "https://custom.host");
        assert_eq!(config.version, "v2");
        assert_eq!(config.timeout, Some(Duration::from_secs(30)));
    }

    #[test]
    fn clone() {
        let config = Config {
            api_key: Some("key".into()),
            host: "https://h".into(),
            version: "v1".into(),
            timeout: Some(Duration::from_secs(5)),
        };
        let cloned = config.clone();
        assert_eq!(cloned.api_key, config.api_key);
        assert_eq!(cloned.host, config.host);
        assert_eq!(cloned.timeout, config.timeout);
    }

    #[test]
    fn debug_output() {
        let debug = format!("{:?}", Config::new());
        assert!(debug.contains("Config"));
        assert!(debug.contains("host"));
    }
}
