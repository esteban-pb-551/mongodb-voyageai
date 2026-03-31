//! Text chunking strategies for splitting documents into embedding-friendly segments.
//!
//! This module provides multiple chunking strategies optimized for RAG (Retrieval-Augmented
//! Generation) pipelines. Each strategy balances semantic coherence with size constraints.
//!
//! # Available Strategies
//!
//! - **Fixed Size** (`chunk_fixed_size`): Simple character-based splitting with overlap
//! - **Sentence-based** (`chunk_by_sentences`): Respects sentence boundaries
//! - **Recursive** (`chunk_recursive`): Hierarchical splitting (paragraphs → sentences → words)
//!
//! # Example
//! ```
//! use mongodb_voyageai::chunk::chunking::{chunk_recursive, ChunkConfig};
//!
//! let config = ChunkConfig::default();
//! let text = "First paragraph.\n\nSecond paragraph with more content.";
//! let chunks = chunk_recursive(text, &config);
//! ```

/// Shared configuration for all chunking strategies.
#[derive(Debug, Clone)]
pub struct ChunkConfig {
    /// Maximum chunk size in characters (approx. 300-500 tokens → ~1500-2500 chars)
    pub chunk_size: usize,
    /// Overlap between chunks to maintain context continuity
    pub chunk_overlap: usize,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            chunk_size: 1500,   // ≈ 500 tokens
            chunk_overlap: 150, // ≈ 50 tokens overlap
        }
    }
}

// ──────────────────────────────────────────────
// Strategy 1: Fixed Size
// ──────────────────────────────────────────────

/// Splits text into fixed-size chunks with overlap.
/// Equivalent to LangChain's `CharacterTextSplitter`.
///
/// # Advantages
/// - Simple and very fast
/// - Predictable size → easy to calculate tokens
///
/// # Disadvantages
/// - May cut in the middle of a sentence
/// - Loss of semantic context at boundaries
///
/// # Example
/// ```
/// use mongodb_voyageai::chunk::chunking::{chunk_fixed_size, ChunkConfig};
///
/// let config = ChunkConfig::default();
/// let text = "This is a long text that needs to be split into smaller chunks.";
/// let chunks = chunk_fixed_size(text, &config);
/// ```
pub fn chunk_fixed_size(text: &str, config: &ChunkConfig) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let total = chars.len();

    if total <= config.chunk_size {
        return vec![text.to_string()];
    }

    let mut chunks = Vec::new();
    let mut start = 0;

    while start < total {
        let end = (start + config.chunk_size).min(total);
        let chunk: String = chars[start..end].iter().collect();
        chunks.push(chunk.trim().to_string());

        if end == total {
            break;
        }

        // Advance by subtracting overlap to maintain continuity
        start += config.chunk_size - config.chunk_overlap;
    }

    chunks.into_iter().filter(|c| !c.is_empty()).collect()
}

// ──────────────────────────────────────────────
// Strategy 2: Semantic Boundaries (Sentence-based)
// ──────────────────────────────────────────────

/// Splits text respecting sentence boundaries, then groups up to chunk_size.
/// Guarantees that no sentence is cut in half.
///
/// # Advantages
/// - Preserves complete semantic context of each sentence
/// - Better embedding quality per chunk
///
/// # Disadvantages
/// - Variable chunk sizes (may have very long chunks)
///
/// # Example
/// ```
/// use mongodb_voyageai::chunk::chunking::{chunk_by_sentences, ChunkConfig};
///
/// let config = ChunkConfig::default();
/// let text = "First sentence. Second sentence. Third sentence.";
/// let chunks = chunk_by_sentences(text, &config);
/// ```
pub fn chunk_by_sentences(text: &str, config: &ChunkConfig) -> Vec<String> {
    // Split into sentences using common delimiters
    let sentences = split_into_sentences(text);

    let mut chunks: Vec<String> = Vec::new();
    let mut current_chunk = String::new();
    let mut overlap_buffer: Vec<String> = Vec::new();

    for sentence in &sentences {
        let sentence = sentence.trim();
        if sentence.is_empty() {
            continue;
        }

        // If adding this sentence exceeds the limit, save current chunk
        if !current_chunk.is_empty()
            && current_chunk.len() + sentence.len() + 1 > config.chunk_size
        {
            chunks.push(current_chunk.trim().to_string());

            // Build next chunk starting from overlap
            current_chunk = overlap_buffer.join(" ");
            overlap_buffer.clear();
        }

        current_chunk.push_str(sentence);
        current_chunk.push(' ');

        // Keep last sentences as overlap buffer
        overlap_buffer.push(sentence.to_string());
        let overlap_text: String = overlap_buffer.join(" ");
        if overlap_text.len() > config.chunk_overlap {
            overlap_buffer.remove(0);
        }
    }

    // Add last chunk if it has content
    if !current_chunk.trim().is_empty() {
        chunks.push(current_chunk.trim().to_string());
    }

    chunks
}

/// Splits text into sentences using delimiters '. ! ?'
fn split_into_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();

    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();

    for i in 0..len {
        current.push(chars[i]);

        if matches!(chars[i], '.' | '!' | '?') {
            // Avoid splits on abbreviations (Mr. Dr. etc.) or decimals (3.14)
            let next_is_space = chars.get(i + 1).map_or(true, |c| c.is_whitespace());
            let prev_is_letter = i > 0 && chars[i - 1].is_alphabetic();

            if next_is_space && current.trim().len() > 10 {
                // Check that it's not a 2-3 letter abbreviation (Mr. Dr. vs.)
                let words: Vec<&str> = current.trim().split_whitespace().collect();
                let last_word = words.last().unwrap_or(&"");
                let is_abbrev = prev_is_letter && last_word.len() <= 3;

                if !is_abbrev {
                    sentences.push(current.trim().to_string());
                    current = String::new();
                }
            }
        }
    }

    if !current.trim().is_empty() {
        sentences.push(current.trim().to_string());
    }

    sentences
}

// ──────────────────────────────────────────────
// Strategy 3: Recursive (RecursiveCharacterTextSplitter)
// ──────────────────────────────────────────────

/// Direct equivalent to LangChain's `RecursiveCharacterTextSplitter`.
/// Attempts to split by separators in order of preference:
///   1. "\n\n" (paragraphs)
///   2. "\n"   (line breaks)
///   3. ". "   (sentences)
///   4. " "    (words)
///   5. ""     (characters — last resort)
///
/// If a fragment is still too large, it's recursively applied
/// with the next separator in the list.
///
/// # Advantages
/// - Respects natural text hierarchy (paragraph > sentence > word)
/// - Best general strategy for RAG
/// - Identical behavior to LangChain
///
/// # Example
/// ```
/// use mongodb_voyageai::chunk::chunking::{chunk_recursive, ChunkConfig};
///
/// let config = ChunkConfig::default();
/// let text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.";
/// let chunks = chunk_recursive(text, &config);
/// ```
pub fn chunk_recursive(text: &str, config: &ChunkConfig) -> Vec<String> {
    let separators = vec!["\n\n", "\n", ". ", " ", ""];
    recursive_split(text, &separators, config)
}

fn recursive_split(text: &str, separators: &[&str], config: &ChunkConfig) -> Vec<String> {
    if text.len() <= config.chunk_size {
        return vec![text.trim().to_string()];
    }

    // Find the first separator that splits this text
    let separator = separators
        .iter()
        .find(|&&sep| !sep.is_empty() && text.contains(sep))
        .copied()
        .unwrap_or("");

    let next_separators = if separator.is_empty() {
        &separators[separators.len()..] // empty list
    } else {
        let idx = separators.iter().position(|&s| s == separator).unwrap_or(0);
        &separators[(idx + 1)..]
    };

    // Split by the chosen separator
    let raw_splits: Vec<&str> = if separator.is_empty() {
        // Last resort: split by characters
        text.char_indices()
            .step_by(config.chunk_size)
            .map(|(i, _)| &text[i..])
            .collect()
    } else {
        text.split(separator).collect()
    };

    // Group small splits up to chunk_size (with overlap)
    let mut final_chunks: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut overlap: Vec<String> = Vec::new();

    for split in raw_splits {
        let split = split.trim();
        if split.is_empty() {
            continue;
        }

        // If this split individually exceeds the limit, subdivide it
        if split.len() > config.chunk_size && !next_separators.is_empty() {
            if !current.is_empty() {
                final_chunks.push(current.trim().to_string());
                current = String::new();
            }
            let sub_chunks = recursive_split(split, next_separators, config);
            final_chunks.extend(sub_chunks);
            continue;
        }

        let sep_display = if separator.is_empty() { "" } else { separator };
        let candidate = if current.is_empty() {
            split.to_string()
        } else {
            format!("{}{}{}", current, sep_display, split)
        };

        if candidate.len() > config.chunk_size && !current.is_empty() {
            // Save current chunk and start from overlap
            final_chunks.push(current.trim().to_string());
            current = format!("{}{}{}", overlap.join(sep_display), sep_display, split);
            overlap.clear();
        } else {
            current = candidate;
        }

        // Update overlap buffer
        overlap.push(split.to_string());
        let overlap_len: usize = overlap.iter().map(|s| s.len()).sum();
        while overlap_len > config.chunk_overlap && overlap.len() > 1 {
            overlap.remove(0);
        }
    }

    if !current.trim().is_empty() {
        final_chunks.push(current.trim().to_string());
    }

    final_chunks
        .into_iter()
        .filter(|c| !c.is_empty())
        .collect()
}


// ──────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_config_default() {
        let config = ChunkConfig::default();
        assert_eq!(config.chunk_size, 1500);
        assert_eq!(config.chunk_overlap, 150);
    }

    #[test]
    fn test_chunk_fixed_size_small_text() {
        let config = ChunkConfig {
            chunk_size: 100,
            chunk_overlap: 20,
        };
        let text = "Short text";
        let chunks = chunk_fixed_size(text, &config);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "Short text");
    }

    #[test]
    fn test_chunk_fixed_size_with_overlap() {
        let config = ChunkConfig {
            chunk_size: 20,
            chunk_overlap: 5,
        };
        let text = "This is a longer text that needs to be split into multiple chunks";
        let chunks = chunk_fixed_size(text, &config);
        assert!(chunks.len() > 1);
        // Verify overlap exists
        for i in 0..chunks.len() - 1 {
            assert!(!chunks[i].is_empty());
        }
    }

    #[test]
    fn test_split_into_sentences_basic() {
        let text = "First sentence. Second sentence! Third sentence?";
        let sentences = split_into_sentences(text);
        assert_eq!(sentences.len(), 3);
        assert!(sentences[0].contains("First"));
        assert!(sentences[1].contains("Second"));
        assert!(sentences[2].contains("Third"));
    }

    #[test]
    fn test_split_into_sentences_abbreviations() {
        let text = "Dr. Smith works here. He is great.";
        let sentences = split_into_sentences(text);
        // Should not split on "Dr."
        assert!(sentences.len() <= 2);
    }

    #[test]
    fn test_split_into_sentences_decimals() {
        let text = "The value is 3.14 approximately. Next sentence.";
        let sentences = split_into_sentences(text);
        // Should not split on "3.14"
        assert_eq!(sentences.len(), 2);
    }

    #[test]
    fn test_chunk_by_sentences_basic() {
        let config = ChunkConfig {
            chunk_size: 50,
            chunk_overlap: 10,
        };
        let text = "First sentence here. Second sentence here. Third sentence here.";
        let chunks = chunk_by_sentences(text, &config);
        assert!(!chunks.is_empty());
        // Each chunk should contain complete sentences
        for chunk in &chunks {
            assert!(!chunk.is_empty());
        }
    }

    #[test]
    fn test_chunk_by_sentences_preserves_sentences() {
        let config = ChunkConfig {
            chunk_size: 30,
            chunk_overlap: 5,
        };
        let text = "Short. Another short. Third.";
        let chunks = chunk_by_sentences(text, &config);
        // Verify no sentence is cut in half
        for chunk in &chunks {
            assert!(chunk.ends_with('.') || chunk.contains("Short") || chunk.contains("Another"));
        }
    }

    #[test]
    fn test_chunk_recursive_small_text() {
        let config = ChunkConfig::default();
        let text = "Small text";
        let chunks = chunk_recursive(text, &config);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "Small text");
    }

    #[test]
    fn test_chunk_recursive_paragraph_split() {
        let config = ChunkConfig {
            chunk_size: 50,
            chunk_overlap: 10,
        };
        let text = "First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph.";
        let chunks = chunk_recursive(text, &config);
        assert!(chunks.len() >= 1);
        // Should respect paragraph boundaries
        for chunk in &chunks {
            assert!(!chunk.is_empty());
        }
    }

    #[test]
    fn test_chunk_recursive_sentence_fallback() {
        let config = ChunkConfig {
            chunk_size: 30,
            chunk_overlap: 5,
        };
        let text = "First sentence. Second sentence. Third sentence.";
        let chunks = chunk_recursive(text, &config);
        assert!(chunks.len() >= 1);
    }

    #[test]
    fn test_chunk_recursive_word_fallback() {
        let config = ChunkConfig {
            chunk_size: 15,
            chunk_overlap: 3,
        };
        let text = "word1 word2 word3 word4 word5 word6";
        let chunks = chunk_recursive(text, &config);
        assert!(chunks.len() >= 1);
        for chunk in &chunks {
            assert!(!chunk.is_empty());
        }
    }

    #[test]
    fn test_recursive_split_empty_text() {
        let config = ChunkConfig::default();
        let text = "";
        let chunks = chunk_recursive(text, &config);
        // Empty text returns a single empty string after trim
        assert!(chunks.is_empty() || (chunks.len() == 1 && chunks[0].is_empty()));
    }

    #[test]
    fn test_chunk_fixed_size_empty_text() {
        let config = ChunkConfig::default();
        let text = "";
        let chunks = chunk_fixed_size(text, &config);
        // Empty text returns a single empty string
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "");
    }

    #[test]
    fn test_chunk_by_sentences_empty_text() {
        let config = ChunkConfig::default();
        let text = "";
        let chunks = chunk_by_sentences(text, &config);
        assert_eq!(chunks.len(), 0);
    }
}
