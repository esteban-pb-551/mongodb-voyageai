//! Output data type for embedding quantization.

use serde::Serialize;

/// Output data type for embedding vectors.
///
/// Voyage AI models with Quantization-Aware Training support multiple output
/// formats that dramatically reduce storage costs while maintaining quality.
///
/// # Storage Comparison
///
/// For a 512-dimensional embedding:
///
/// | Type     | Bytes per vector | Compression vs float |
/// |----------|------------------|----------------------|
/// | `Float`  | 2048 (512 × 4)   | 1×                   |
/// | `Int8`   | 512 (512 × 1)    | 4×                   |
/// | `Uint8`  | 512 (512 × 1)    | 4×                   |
/// | `Binary` | 64 (512 ÷ 8)     | 32×                  |
/// | `Ubinary`| 64 (512 ÷ 8)     | 32×                  |
///
/// # Examples
///
/// ```rust
/// use mongodb_voyageai::OutputDtype;
///
/// // Default: full precision
/// let dtype = OutputDtype::Float;
///
/// // 4× compression with minimal quality loss
/// let dtype = OutputDtype::Int8;
///
/// // 32× compression for maximum efficiency
/// let dtype = OutputDtype::Binary;
/// ```
///
/// # Model Support
///
/// Only models with Quantization-Aware Training support non-float types:
/// - `voyage-3-large`
/// - `voyage-4` series
/// - Check the [Voyage AI docs](https://docs.voyageai.com) for the latest list
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum OutputDtype {
    /// Full precision floating-point (default).
    ///
    /// 32-bit floats, no compression. Use when maximum precision is required.
    Float,

    /// Signed 8-bit integer quantization.
    ///
    /// Values in range [-128, 127]. Provides 4× storage reduction with
    /// minimal quality loss. Recommended for most production use cases.
    Int8,

    /// Unsigned 8-bit integer quantization.
    ///
    /// Values in range [0, 255]. Provides 4× storage reduction.
    Uint8,

    /// Binary quantization (signed).
    ///
    /// Each dimension becomes a single bit (-1 or +1). Provides 32× storage
    /// reduction. Best for very large-scale deployments where storage cost
    /// is critical.
    Binary,

    /// Binary quantization (unsigned).
    ///
    /// Each dimension becomes a single bit (0 or 1). Provides 32× storage
    /// reduction.
    Ubinary,
}

impl Default for OutputDtype {
    fn default() -> Self {
        Self::Float
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serialize_float() {
        let json = serde_json::to_string(&OutputDtype::Float).unwrap();
        assert_eq!(json, r#""float""#);
    }

    #[test]
    fn serialize_int8() {
        let json = serde_json::to_string(&OutputDtype::Int8).unwrap();
        assert_eq!(json, r#""int8""#);
    }

    #[test]
    fn serialize_uint8() {
        let json = serde_json::to_string(&OutputDtype::Uint8).unwrap();
        assert_eq!(json, r#""uint8""#);
    }

    #[test]
    fn serialize_binary() {
        let json = serde_json::to_string(&OutputDtype::Binary).unwrap();
        assert_eq!(json, r#""binary""#);
    }

    #[test]
    fn serialize_ubinary() {
        let json = serde_json::to_string(&OutputDtype::Ubinary).unwrap();
        assert_eq!(json, r#""ubinary""#);
    }

    #[test]
    fn default_is_float() {
        assert_eq!(OutputDtype::default(), OutputDtype::Float);
    }

    #[test]
    fn equality() {
        assert_eq!(OutputDtype::Int8, OutputDtype::Int8);
        assert_ne!(OutputDtype::Int8, OutputDtype::Uint8);
    }

    #[test]
    fn clone() {
        let dtype = OutputDtype::Binary;
        assert_eq!(dtype.clone(), OutputDtype::Binary);
    }

    #[test]
    fn debug() {
        let debug = format!("{:?}", OutputDtype::Int8);
        assert!(debug.contains("Int8"));
    }
}
