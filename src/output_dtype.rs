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

    // ── Serialization tests ──────────────────────────────────────────────────

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
    fn serialize_all_variants() {
        let variants = vec![
            (OutputDtype::Float, r#""float""#),
            (OutputDtype::Int8, r#""int8""#),
            (OutputDtype::Uint8, r#""uint8""#),
            (OutputDtype::Binary, r#""binary""#),
            (OutputDtype::Ubinary, r#""ubinary""#),
        ];

        for (dtype, expected) in variants {
            let json = serde_json::to_string(&dtype).unwrap();
            assert_eq!(json, expected, "Failed for {:?}", dtype);
        }
    }

    #[test]
    fn serialize_in_struct() {
        #[derive(serde::Serialize)]
        struct TestStruct {
            dtype: OutputDtype,
        }

        let test = TestStruct {
            dtype: OutputDtype::Int8,
        };
        let json = serde_json::to_string(&test).unwrap();
        assert!(json.contains(r#""dtype":"int8""#));
    }

    #[test]
    fn serialize_in_option_some() {
        let dtype: Option<OutputDtype> = Some(OutputDtype::Binary);
        let json = serde_json::to_string(&dtype).unwrap();
        assert_eq!(json, r#""binary""#);
    }

    #[test]
    fn serialize_in_option_none() {
        let dtype: Option<OutputDtype> = None;
        let json = serde_json::to_string(&dtype).unwrap();
        assert_eq!(json, "null");
    }

    #[test]
    fn serialize_in_vec() {
        let dtypes = vec![OutputDtype::Float, OutputDtype::Int8, OutputDtype::Binary];
        let json = serde_json::to_string(&dtypes).unwrap();
        assert_eq!(json, r#"["float","int8","binary"]"#);
    }

    // ── Default trait tests ──────────────────────────────────────────────────

    #[test]
    fn default_is_float() {
        assert_eq!(OutputDtype::default(), OutputDtype::Float);
    }

    #[test]
    fn default_in_struct() {
        #[derive(Default)]
        struct Config {
            #[allow(dead_code)]
            dtype: OutputDtype,
        }

        let config = Config::default();
        assert_eq!(config.dtype, OutputDtype::Float);
    }

    // ── Equality and comparison tests ────────────────────────────────────────

    #[test]
    fn equality_same_variant() {
        assert_eq!(OutputDtype::Float, OutputDtype::Float);
        assert_eq!(OutputDtype::Int8, OutputDtype::Int8);
        assert_eq!(OutputDtype::Uint8, OutputDtype::Uint8);
        assert_eq!(OutputDtype::Binary, OutputDtype::Binary);
        assert_eq!(OutputDtype::Ubinary, OutputDtype::Ubinary);
    }

    #[test]
    fn inequality_different_variants() {
        assert_ne!(OutputDtype::Float, OutputDtype::Int8);
        assert_ne!(OutputDtype::Int8, OutputDtype::Uint8);
        assert_ne!(OutputDtype::Binary, OutputDtype::Ubinary);
        assert_ne!(OutputDtype::Float, OutputDtype::Binary);
    }

    #[test]
    fn equality_reflexive() {
        let dtype = OutputDtype::Int8;
        assert_eq!(dtype, dtype);
    }

    #[test]
    fn equality_symmetric() {
        let a = OutputDtype::Binary;
        let b = OutputDtype::Binary;
        assert_eq!(a, b);
        assert_eq!(b, a);
    }

    #[test]
    fn equality_transitive() {
        let a = OutputDtype::Uint8;
        let b = OutputDtype::Uint8;
        let c = OutputDtype::Uint8;
        assert_eq!(a, b);
        assert_eq!(b, c);
        assert_eq!(a, c);
    }

    // ── Clone trait tests ────────────────────────────────────────────────────

    #[test]
    fn clone_float() {
        let original = OutputDtype::Float;
        let cloned = original.clone();
        assert_eq!(original, cloned);
    }

    #[test]
    fn clone_int8() {
        let original = OutputDtype::Int8;
        let cloned = original.clone();
        assert_eq!(original, cloned);
    }

    #[test]
    fn clone_binary() {
        let dtype = OutputDtype::Binary;
        assert_eq!(dtype.clone(), OutputDtype::Binary);
    }

    #[test]
    fn clone_all_variants() {
        let variants = vec![
            OutputDtype::Float,
            OutputDtype::Int8,
            OutputDtype::Uint8,
            OutputDtype::Binary,
            OutputDtype::Ubinary,
        ];

        for dtype in variants {
            let cloned = dtype.clone();
            assert_eq!(dtype, cloned);
        }
    }

    // ── Copy trait tests ─────────────────────────────────────────────────────

    #[test]
    fn copy_semantics() {
        let original = OutputDtype::Int8;
        let copied = original; // Copy, not move
        assert_eq!(original, copied);
        assert_eq!(original, OutputDtype::Int8); // original still accessible
    }

    #[test]
    fn copy_in_function() {
        fn takes_dtype(_dtype: OutputDtype) {}

        let dtype = OutputDtype::Binary;
        takes_dtype(dtype);
        // dtype is still accessible because it was copied
        assert_eq!(dtype, OutputDtype::Binary);
    }

    // ── Debug trait tests ────────────────────────────────────────────────────

    #[test]
    fn debug_float() {
        let debug = format!("{:?}", OutputDtype::Float);
        assert_eq!(debug, "Float");
    }

    #[test]
    fn debug_int8() {
        let debug = format!("{:?}", OutputDtype::Int8);
        assert_eq!(debug, "Int8");
    }

    #[test]
    fn debug_uint8() {
        let debug = format!("{:?}", OutputDtype::Uint8);
        assert_eq!(debug, "Uint8");
    }

    #[test]
    fn debug_binary() {
        let debug = format!("{:?}", OutputDtype::Binary);
        assert_eq!(debug, "Binary");
    }

    #[test]
    fn debug_ubinary() {
        let debug = format!("{:?}", OutputDtype::Ubinary);
        assert_eq!(debug, "Ubinary");
    }

    #[test]
    fn debug_all_variants() {
        let variants = vec![
            (OutputDtype::Float, "Float"),
            (OutputDtype::Int8, "Int8"),
            (OutputDtype::Uint8, "Uint8"),
            (OutputDtype::Binary, "Binary"),
            (OutputDtype::Ubinary, "Ubinary"),
        ];

        for (dtype, expected) in variants {
            let debug = format!("{:?}", dtype);
            assert_eq!(debug, expected);
        }
    }

    #[test]
    fn debug_in_vec() {
        let dtypes = vec![OutputDtype::Float, OutputDtype::Int8];
        let debug = format!("{:?}", dtypes);
        assert!(debug.contains("Float"));
        assert!(debug.contains("Int8"));
    }

    // ── Pattern matching tests ───────────────────────────────────────────────

    #[test]
    fn match_all_variants() {
        let variants = vec![
            OutputDtype::Float,
            OutputDtype::Int8,
            OutputDtype::Uint8,
            OutputDtype::Binary,
            OutputDtype::Ubinary,
        ];

        for dtype in variants {
            let name = match dtype {
                OutputDtype::Float => "float",
                OutputDtype::Int8 => "int8",
                OutputDtype::Uint8 => "uint8",
                OutputDtype::Binary => "binary",
                OutputDtype::Ubinary => "ubinary",
            };
            assert!(!name.is_empty());
        }
    }

    #[test]
    fn match_compression_level() {
        fn compression_factor(dtype: OutputDtype) -> u32 {
            match dtype {
                OutputDtype::Float => 1,
                OutputDtype::Int8 | OutputDtype::Uint8 => 4,
                OutputDtype::Binary | OutputDtype::Ubinary => 32,
            }
        }

        assert_eq!(compression_factor(OutputDtype::Float), 1);
        assert_eq!(compression_factor(OutputDtype::Int8), 4);
        assert_eq!(compression_factor(OutputDtype::Uint8), 4);
        assert_eq!(compression_factor(OutputDtype::Binary), 32);
        assert_eq!(compression_factor(OutputDtype::Ubinary), 32);
    }

    // ── Integration tests ────────────────────────────────────────────────────

    #[test]
    fn use_in_option() {
        let some_dtype: Option<OutputDtype> = Some(OutputDtype::Int8);
        let none_dtype: Option<OutputDtype> = None;

        assert!(some_dtype.is_some());
        assert!(none_dtype.is_none());
        assert_eq!(some_dtype.unwrap(), OutputDtype::Int8);
    }

    #[test]
    fn use_in_result() {
        let ok_dtype: Result<OutputDtype, &str> = Ok(OutputDtype::Binary);
        let err_dtype: Result<OutputDtype, &str> = Err("invalid");

        assert!(ok_dtype.is_ok());
        assert!(err_dtype.is_err());
        assert_eq!(ok_dtype.unwrap(), OutputDtype::Binary);
    }

    #[test]
    fn store_in_vec() {
        let mut dtypes = Vec::new();
        dtypes.push(OutputDtype::Float);
        dtypes.push(OutputDtype::Int8);
        dtypes.push(OutputDtype::Binary);

        assert_eq!(dtypes.len(), 3);
        assert_eq!(dtypes[0], OutputDtype::Float);
        assert_eq!(dtypes[1], OutputDtype::Int8);
        assert_eq!(dtypes[2], OutputDtype::Binary);
    }

    #[test]
    fn use_as_hashmap_key() {
        use std::collections::HashMap;

        let mut map = HashMap::new();
        map.insert(OutputDtype::Float, "full precision");
        map.insert(OutputDtype::Int8, "4x compression");
        map.insert(OutputDtype::Binary, "32x compression");

        assert_eq!(map.get(&OutputDtype::Float), Some(&"full precision"));
        assert_eq!(map.get(&OutputDtype::Int8), Some(&"4x compression"));
        assert_eq!(map.get(&OutputDtype::Binary), Some(&"32x compression"));
    }

    // ── Practical usage tests ────────────────────────────────────────────────

    #[test]
    fn calculate_storage_bytes() {
        fn storage_per_dimension(dtype: OutputDtype) -> f64 {
            match dtype {
                OutputDtype::Float => 4.0,
                OutputDtype::Int8 | OutputDtype::Uint8 => 1.0,
                OutputDtype::Binary | OutputDtype::Ubinary => 0.125, // 1 bit = 1/8 byte
            }
        }

        let dims = 512;
        assert_eq!(storage_per_dimension(OutputDtype::Float) * dims as f64, 2048.0);
        assert_eq!(storage_per_dimension(OutputDtype::Int8) * dims as f64, 512.0);
        assert_eq!(storage_per_dimension(OutputDtype::Binary) * dims as f64, 64.0);
    }

    #[test]
    fn recommend_dtype_for_use_case() {
        fn recommend(storage_critical: bool, quality_critical: bool) -> OutputDtype {
            match (storage_critical, quality_critical) {
                (true, true) => OutputDtype::Int8,
                (true, false) => OutputDtype::Binary,
                (false, true) => OutputDtype::Float,
                (false, false) => OutputDtype::Int8,
            }
        }

        assert_eq!(recommend(true, true), OutputDtype::Int8);
        assert_eq!(recommend(true, false), OutputDtype::Binary);
        assert_eq!(recommend(false, true), OutputDtype::Float);
        assert_eq!(recommend(false, false), OutputDtype::Int8);
    }
}
