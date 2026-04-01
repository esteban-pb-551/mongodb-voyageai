//! Text normalization utilities for cleaning and standardizing text input.
//!
//! This module provides a configurable pipeline for normalizing text before
//! embedding or processing. It handles common issues like inconsistent whitespace,
//! Unicode punctuation, hyphenation, and markdown formatting.
//!
//! # Example
//! ```
//! use mongodb_voyageai::chunk::normalizer::{normalize, NormalizerConfig};
//!
//! let config = NormalizerConfig::default();
//! let clean = normalize("  Hello,\r\n\n\n  world!  ", &config);
//! assert_eq!(clean, "Hello,\n\nworld!");
//! ```

/// Configuration for the text normalization pipeline.
/// Each flag enables or disables a specific normalization step.
#[derive(Debug, Clone)]
pub struct NormalizerConfig {
    /// Strip leading/trailing whitespace from each line
    pub trim_lines: bool,
    /// Collapse multiple blank lines into a single paragraph break
    pub collapse_blank_lines: bool,
    /// Collapse multiple spaces/tabs within a line into a single space
    pub collapse_inline_whitespace: bool,
    /// Remove soft hyphens and line-break hyphens (e.g. "infor-\nmation" → "information")
    pub fix_hyphenation: bool,
    /// Normalize Unicode punctuation to ASCII equivalents
    /// e.g. curly quotes → straight quotes, em-dash → "--", ellipsis → "..."
    pub normalize_unicode_punctuation: bool,
    /// Strip control characters (except \n and \t)
    pub strip_control_chars: bool,
    /// Normalize line endings: convert \r\n and \r to \n
    pub normalize_line_endings: bool,
    /// Join single newlines within a paragraph into a single space,
    /// while preserving double newlines as paragraph boundaries.
    /// Should be true for most RAG / embedding use cases.
    pub join_soft_line_breaks: bool,
    /// Collapse all whitespace (including newlines) into a single space.
    /// Useful when the downstream splitter handles structure itself.
    /// Overrides `collapse_blank_lines` and `join_soft_line_breaks` when true.
    pub flatten_to_single_line: bool,
    /// Remove URLs from text (useful for cleaner embeddings)
    pub strip_urls: bool,
    /// Remove markdown formatting symbols (**, __, ##, etc.)
    pub strip_markdown: bool,
}

impl Default for NormalizerConfig {
    fn default() -> Self {
        Self {
            trim_lines: true,
            collapse_blank_lines: true,
            collapse_inline_whitespace: true,
            fix_hyphenation: true,
            normalize_unicode_punctuation: true,
            strip_control_chars: true,
            normalize_line_endings: true,
            join_soft_line_breaks: true,
            flatten_to_single_line: false,
            strip_urls: false,
            strip_markdown: false,
        }
    }
}

impl NormalizerConfig {
    /// Preset for plain prose documents (articles, books, reports).
    pub fn prose() -> Self {
        Self::default()
    }

    /// Preset for code-adjacent documents (READMEs, docstrings, changelogs).
    /// Keeps structure but strips markdown symbols and URLs.
    pub fn code_docs() -> Self {
        Self {
            strip_markdown: true,
            strip_urls: true,
            // Preserve intra-paragraph newlines so code examples stay readable
            join_soft_line_breaks: false,
            ..Self::default()
        }
    }

    /// Preset for single-field inputs (search queries, short descriptions).
    /// Flattens everything to one clean line.
    pub fn single_line() -> Self {
        Self {
            flatten_to_single_line: true,
            ..Self::default()
        }
    }

    /// Preset for web-scraped text with heavy noise.
    pub fn web_scraped() -> Self {
        Self {
            strip_urls: true,
            strip_markdown: true,
            ..Self::default()
        }
    }
}

/// Normalize `text` according to `config`, applying each enabled step in order.
///
/// Steps are applied in a deliberate sequence so that earlier transformations
/// do not interfere with later ones:
///
///  1. Strip control characters
///  2. Normalize line endings
///  3. Fix hyphenation          (must run before collapsing newlines)
///  4. Strip URLs
///  5. Strip Markdown
///  6. Normalize Unicode punctuation
///  7. Trim each line
///  8. Collapse inline whitespace
///  9. Collapse blank lines     (or flatten to single line)
/// 10. Join soft line breaks    (single \n → space within paragraphs)
/// 11. Final trim of the entire string
///
/// # Example
/// ```
/// let config = NormalizerConfig::default();
/// let clean = normalize("  Hello,\r\n\n\n  world!  ", &config);
/// assert_eq!(clean, "Hello,\n\nworld!");
/// ```
pub fn normalize(text: &str, config: &NormalizerConfig) -> String {
    let mut s = text.to_string();

    if config.strip_control_chars {
        s = strip_control_chars(&s);
    }
    if config.normalize_line_endings {
        s = normalize_line_endings(&s);
    }
    if config.fix_hyphenation {
        s = fix_hyphenation(&s);
    }
    if config.strip_urls {
        s = strip_urls(&s);
    }
    if config.strip_markdown {
        s = strip_markdown(&s);
    }
    if config.normalize_unicode_punctuation {
        s = normalize_unicode_punctuation(&s);
    }
    if config.trim_lines {
        s = trim_lines(&s);
    }
    if config.collapse_inline_whitespace {
        s = collapse_inline_whitespace(&s);
    }

    if config.flatten_to_single_line {
        s = flatten_to_single_line(&s);
    } else {
        if config.collapse_blank_lines {
            s = collapse_blank_lines(&s);
        }
        if config.join_soft_line_breaks {
            s = join_soft_line_breaks(&s);
        }
    }

    s.trim().to_string()
}

/// Convenience wrapper using `NormalizerConfig::default()`.
pub fn normalize_default(text: &str) -> String {
    normalize(text, &NormalizerConfig::default())
}

// ─────────────────────────────────────────────────────────────────────────────
// Individual normalization steps
// ─────────────────────────────────────────────────────────────────────────────

/// Remove ASCII control characters (0x00–0x1F and 0x7F) except:
/// - `\n` (0x0A) — preserved as structural separator
/// - `\r` (0x0D) — handled separately by `normalize_line_endings`
/// - `\t` (0x09) — preserved; collapsed later by `collapse_inline_whitespace`
fn strip_control_chars(text: &str) -> String {
    text.chars()
        .filter(|&c| !c.is_control() || matches!(c, '\n' | '\r' | '\t'))
        .collect()
}

/// Convert `\r\n` (Windows) and bare `\r` (old Mac) to `\n`.
fn normalize_line_endings(text: &str) -> String {
    text.replace("\r\n", "\n").replace('\r', "\n")
}

/// Rejoin words broken by soft hyphens at line boundaries.
///
/// Handles two patterns:
/// - Hard hyphen at end of line:  "infor-\nmation" → "information"
/// - Soft hyphen (U+00AD):        "infor\u{00AD}mation" → "information"
///
/// Intentionally hyphenated words on the same line (e.g. "well-known") are
/// preserved because the hyphen-removal only fires when the character
/// immediately after the newline is a lowercase letter.
fn fix_hyphenation(text: &str) -> String {
    // Remove soft hyphen (U+00AD) unconditionally
    let s = text.replace('\u{00AD}', "");

    let mut result = String::with_capacity(s.len());
    let chars: Vec<char> = s.chars().collect();
    let len = chars.len();
    let mut i = 0;

    while i < len {
        let c = chars[i];

        // Detect: alphabetic + '-' + '\n' + lowercase
        if c == '-'
            && i + 2 < len
            && chars[i + 1] == '\n'
            && chars[i + 2].is_lowercase()
            && i > 0
            && chars[i - 1].is_alphabetic()
        {
            // Drop the hyphen and the newline
            i += 2;
            continue;
        }

        result.push(c);
        i += 1;
    }

    result
}

/// Replace common Unicode punctuation with ASCII equivalents.
///
/// | Input                        | Output    |
/// |------------------------------|-----------|
/// | `"` `"` `„` `«` `»`         | `"`       |
/// | `'` `'` `‚` `` ` `` `‹` `›` | `'`       |
/// | `–` (en-dash)                | `-`       |
/// | `—` (em-dash)                | `--`      |
/// | `…` (ellipsis)               | `...`     |
/// | `•` `·` `‣` `◦`             | `-`       |
/// | non-breaking / thin space    | ` `       |
/// | zero-width chars             | (removed) |
fn normalize_unicode_punctuation(text: &str) -> String {
    text.chars()
        .fold(String::with_capacity(text.len()), |mut acc, c| {
            match c {
                // Curly / angled double quotes → straight double quote
                '\u{201C}' | '\u{201D}' | '\u{201E}' | '\u{00AB}' | '\u{00BB}' => acc.push('"'),
                // Curly / angled single quotes, backtick → straight single quote
                '\u{2018}' | '\u{2019}' | '\u{201A}' | '\u{2039}' | '\u{203A}' | '`' => {
                    acc.push('\'')
                }
                // En-dash → hyphen-minus
                '\u{2013}' => acc.push('-'),
                // Em-dash → double hyphen
                '\u{2014}' => acc.push_str("--"),
                // Horizontal ellipsis → three dots
                '\u{2026}' => acc.push_str("..."),
                // Bullet variants → hyphen-minus
                '\u{2022}' | '\u{00B7}' | '\u{2023}' | '\u{25E6}' => acc.push('-'),
                // Non-breaking space / narrow NBSP / thin space → regular space
                '\u{00A0}' | '\u{202F}' | '\u{2009}' => acc.push(' '),
                // Zero-width space / ZWNJ / ZWJ / BOM → remove
                '\u{200B}' | '\u{200C}' | '\u{200D}' | '\u{FEFF}' => {}
                other => acc.push(other),
            }
            acc
        })
}

/// Trim leading and trailing whitespace from every line.
fn trim_lines(text: &str) -> String {
    text.lines()
        .map(|line| line.trim())
        .collect::<Vec<_>>()
        .join("\n")
}

/// Collapse runs of spaces and tabs within a line to a single space.
/// Newlines are left untouched so paragraph structure is preserved.
fn collapse_inline_whitespace(text: &str) -> String {
    text.lines()
        .map(|line| {
            let mut prev_space = false;
            line.chars()
                .filter_map(|c| {
                    if c == ' ' || c == '\t' {
                        if prev_space {
                            None
                        } else {
                            prev_space = true;
                            Some(' ')
                        }
                    } else {
                        prev_space = false;
                        Some(c)
                    }
                })
                .collect::<String>()
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Collapse runs of more than one consecutive blank line into exactly one
/// blank line (i.e. a single `\n\n` paragraph break).
fn collapse_blank_lines(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut blank_run = 0usize;
    let lines: Vec<&str> = text.lines().collect();

    for (idx, line) in lines.iter().enumerate() {
        if line.trim().is_empty() {
            blank_run += 1;
            if blank_run == 1 {
                result.push('\n');
            }
        } else {
            blank_run = 0;
            result.push_str(line);
            // Only add newline if not the last line
            if idx < lines.len() - 1 {
                result.push('\n');
            }
        }
    }

    result.trim_end().to_string()
}

/// Join single newlines within a paragraph into a single space,
/// while preserving double newlines as paragraph boundaries.
///
/// This is the key step for RAG pipelines: embedding models receive
/// dense, unbroken semantic context per chunk instead of fragmented
/// multi-line strings that carry accidental line-break positions.
///
/// Must run AFTER `collapse_blank_lines` so that `\n\n` boundaries
/// are already clean before single `\n` breaks are merged.
///
/// # Example
/// ```
/// let input = "first paragraph\nsecond line\n\nnew paragraph\nits second line";
/// let out = join_soft_line_breaks(input);
/// assert_eq!(out, "first paragraph second line\n\nnew paragraph its second line");
/// ```
fn join_soft_line_breaks(text: &str) -> String {
    text.split("\n\n")
        .map(|paragraph| {
            paragraph
                .lines()
                .map(|l| l.trim())
                .filter(|l| !l.is_empty())
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// Collapse all whitespace (spaces, tabs, newlines) into a single space.
/// Useful for normalizing query strings or single-field inputs.
fn flatten_to_single_line(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Remove URLs from text, replacing them with a single space.
///
/// Matches `http://`, `https://`, and `www.` prefixed URLs up to the next
/// whitespace character. Lightweight but sufficient for most web-scraped text.
///
/// Note: URLs are replaced with a space to avoid joining adjacent words.
/// Use `collapse_inline_whitespace` afterward to clean up any double spaces.
fn strip_urls(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();
    let mut i = 0;

    while i < len {
        let remaining: String = chars[i..].iter().collect();

        let is_url = remaining.starts_with("https://")
            || remaining.starts_with("http://")
            || remaining.starts_with("www.");

        if is_url {
            // Skip the URL
            while i < len && !chars[i].is_whitespace() {
                i += 1;
            }
            // Add space only if we're not at the start and previous char wasn't whitespace
            if !result.is_empty() && !result.ends_with(char::is_whitespace) {
                result.push(' ');
            }
        } else {
            result.push(chars[i]);
            i += 1;
        }
    }

    result
}

/// Remove common Markdown formatting symbols while preserving the underlying text.
///
/// Handles:
/// - ATX headings:      `## Title`        → `Title`
/// - Bold/italic:       `**text**`        → `text`
/// - Inline code:       `` `code` ``      → `code`
/// - Fenced blocks:     ` ```\n...\n``` ` → inner content preserved
/// - Blockquotes:       `> text`          → `text`
/// - Strikethrough:     `~~text~~`        → `text`
/// - Horizontal rules:  `---`, `***`      → removed
/// - HTML tags:         `<br>`, `<p>`     → removed
fn strip_markdown(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut in_fenced_block = false;

    for line in text.lines() {
        let trimmed = line.trim();

        // Toggle fenced code block
        if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
            in_fenced_block = !in_fenced_block;
            result.push('\n');
            continue;
        }

        if in_fenced_block {
            result.push_str(line);
            result.push('\n');
            continue;
        }

        if is_horizontal_rule(trimmed) {
            result.push('\n');
            continue;
        }

        let mut processed = line.to_string();

        // ATX headings
        if processed.trim_start().starts_with('#') {
            processed = processed.trim_start_matches(['#', ' ']).to_string();
        }

        // Blockquotes
        if processed.trim_start().starts_with('>') {
            processed = processed.trim_start_matches(['>', ' ']).to_string();
        }

        // Inline patterns — longer delimiters first to avoid partial matches
        processed = strip_inline_delimiters(&processed, "~~");
        processed = strip_inline_delimiters(&processed, "**");
        processed = strip_inline_delimiters(&processed, "__");
        processed = strip_inline_delimiters(&processed, "*");
        processed = strip_inline_delimiters(&processed, "_");
        processed = strip_inline_delimiters(&processed, "`");

        // HTML tags
        processed = strip_html_tags(&processed);

        result.push_str(&processed);
        result.push('\n');
    }

    result.trim_end().to_string()
}

/// Returns true if `line` is a Markdown horizontal rule (`---`, `***`, `===`).
fn is_horizontal_rule(line: &str) -> bool {
    if line.len() < 3 {
        return false;
    }
    let chars: Vec<char> = line.chars().filter(|c| !c.is_whitespace()).collect();
    if chars.len() < 3 {
        return false;
    }
    let first = chars[0];
    matches!(first, '-' | '*' | '=') && chars.iter().all(|&c| c == first)
}

/// Remove paired inline delimiters (e.g. `**bold**` → `bold`).
fn strip_inline_delimiters(text: &str, delimiter: &str) -> String {
    let mut result = text.to_string();
    while let Some(start) = result.find(delimiter) {
        let after_open = start + delimiter.len();
        match result[after_open..].find(delimiter) {
            Some(rel_end) => {
                let end = after_open + rel_end;
                let content = result[after_open..end].to_string();
                result = format!(
                    "{}{}{}",
                    &result[..start],
                    content,
                    &result[end + delimiter.len()..]
                );
            }
            None => break,
        }
    }
    result
}

/// Remove simple HTML tags from text.
fn strip_html_tags(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut in_tag = false;

    for c in text.chars() {
        match c {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => result.push(c),
            _ => {}
        }
    }

    result
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Individual steps ────────────────────────────────────────────────────

    #[test]
    fn test_strip_control_chars() {
        let input = "hello\x00world\x01!\n\tnext";
        assert_eq!(strip_control_chars(input), "helloworld!\n\tnext");
    }

    #[test]
    fn test_normalize_line_endings() {
        assert_eq!(normalize_line_endings("a\r\nb\rc"), "a\nb\nc");
    }

    #[test]
    fn test_fix_hyphenation_line_break() {
        assert_eq!(
            fix_hyphenation("infor-\nmation is power"),
            "information is power"
        );
    }

    #[test]
    fn test_fix_hyphenation_preserves_intentional() {
        // "well-known" on the same line must NOT be touched
        assert_eq!(
            fix_hyphenation("well-known\nnext line"),
            "well-known\nnext line"
        );
    }

    #[test]
    fn test_fix_hyphenation_soft_hyphen() {
        assert_eq!(fix_hyphenation("sug\u{00AD}gest"), "suggest");
    }

    #[test]
    fn test_unicode_punctuation_quotes() {
        let input = "\u{201C}Hello\u{201D} it\u{2019}s fine\u{2026}";
        assert_eq!(
            normalize_unicode_punctuation(input),
            "\"Hello\" it's fine..."
        );
    }

    #[test]
    fn test_unicode_punctuation_em_dash() {
        assert_eq!(normalize_unicode_punctuation("one\u{2014}two"), "one--two");
    }

    #[test]
    fn test_unicode_punctuation_nbsp() {
        assert_eq!(normalize_unicode_punctuation("a\u{00A0}b"), "a b");
    }

    #[test]
    fn test_unicode_punctuation_zero_width() {
        assert_eq!(normalize_unicode_punctuation("a\u{200B}b"), "ab");
    }

    #[test]
    fn test_trim_lines() {
        assert_eq!(trim_lines("  hello  \n  world  "), "hello\nworld");
    }

    #[test]
    fn test_collapse_inline_whitespace() {
        let input = "one   two\t\tthree\nfour  five";
        assert_eq!(
            collapse_inline_whitespace(input),
            "one two three\nfour five"
        );
    }

    #[test]
    fn test_collapse_blank_lines() {
        let input = "para one\n\n\n\npara two\n\n\npara three";
        assert_eq!(
            collapse_blank_lines(input),
            "para one\n\npara two\n\npara three"
        );
    }

    #[test]
    fn test_join_soft_line_breaks_within_paragraph() {
        let input = "line one\nline two\nline three";
        assert_eq!(join_soft_line_breaks(input), "line one line two line three");
    }

    #[test]
    fn test_join_soft_line_breaks_preserves_paragraph_boundary() {
        let input = "first paragraph\nsecond line\n\nnew paragraph\nits second line";
        assert_eq!(
            join_soft_line_breaks(input),
            "first paragraph second line\n\nnew paragraph its second line"
        );
    }

    #[test]
    fn test_join_soft_line_breaks_multiple_paragraphs() {
        let input = "para one\nline two\n\npara two\nline two\n\npara three";
        assert_eq!(
            join_soft_line_breaks(input),
            "para one line two\n\npara two line two\n\npara three"
        );
    }

    #[test]
    fn test_flatten_to_single_line() {
        assert_eq!(
            flatten_to_single_line("  hello\n  world  \n  again  "),
            "hello world again"
        );
    }

    #[test]
    fn test_strip_urls_https() {
        let out = strip_urls("Visit https://example.com for details.");
        assert!(!out.contains("https://"));
        assert!(out.contains("Visit"));
        assert!(out.contains("for details"));
    }

    #[test]
    fn test_strip_urls_www() {
        let out = strip_urls("See www.foo.org too.");
        assert!(!out.contains("www.foo.org"));
    }

    #[test]
    fn test_strip_markdown_headings() {
        assert_eq!(strip_markdown("## Section Title").trim(), "Section Title");
    }

    #[test]
    fn test_strip_markdown_bold_italic() {
        let out = strip_markdown("This is **bold** and *italic* and __also bold__.");
        assert_eq!(out.trim(), "This is bold and italic and also bold.");
    }

    #[test]
    fn test_strip_markdown_inline_code() {
        assert_eq!(
            strip_markdown("Use `println!()` here.").trim(),
            "Use println!() here."
        );
    }

    #[test]
    fn test_strip_markdown_blockquote() {
        assert_eq!(
            strip_markdown("> This is a quote").trim(),
            "This is a quote"
        );
    }

    #[test]
    fn test_strip_markdown_horizontal_rule() {
        let out = strip_markdown("before\n---\nafter");
        assert!(!out.contains("---"));
        assert!(out.contains("before"));
        assert!(out.contains("after"));
    }

    #[test]
    fn test_strip_markdown_fenced_block_content_preserved() {
        let input = "Intro\n```rust\nlet x = 1;\n```\nOutro";
        let out = strip_markdown(input);
        assert!(
            out.contains("Intro"),
            "Should preserve text before code block"
        );
        assert!(
            out.contains("let x = 1;"),
            "Should preserve code block content"
        );
        assert!(
            out.contains("Outro"),
            "Should preserve text after code block"
        );
        assert!(!out.contains("```"), "Should remove fence markers");
    }

    #[test]
    fn test_strip_html_tags() {
        assert_eq!(
            strip_html_tags("Hello <br> world <p>test</p>"),
            "Hello  world test"
        );
    }

    // ── Full pipeline ────────────────────────────────────────────────────────

    #[test]
    fn test_full_pipeline_prose_indented_raw_string() {
        // Simulates the exact case from the chunking output
        let input = r#"
            The Mediterranean diet emphasizes fish, olive oil, and vegetables,
            believed to reduce chronic diseases. Studies show a significant reduction
            in cardiovascular risk among adherents.

            Photosynthesis in plants converts light energy into glucose and produces
            essential oxygen.
        "#;

        let out = normalize(input, &NormalizerConfig::prose());

        // No leading/trailing whitespace
        assert!(!out.starts_with(' '));
        assert!(!out.ends_with(' '));

        // Intra-paragraph lines joined into a single space
        assert!(out.contains("vegetables, believed"), "got: {}", out);

        // Paragraph boundary preserved as \n\n
        assert!(out.contains("adherents.\n\nPhotosynthesis"), "got: {}", out);

        // No indentation remnants
        assert!(!out.contains("            "));
    }

    #[test]
    fn test_full_pipeline_unicode_and_line_breaks() {
        let input = "The \u{201C}quick\u{201D}\r\n\n\n  brown   fox\u{00A0}jumps.";
        let out = normalize(input, &NormalizerConfig::prose());
        assert_eq!(out, "The \"quick\"\n\nbrown fox jumps.");
    }

    #[test]
    fn test_preset_single_line() {
        let input = "  Hello\n\n  World  \n  Again  ";
        assert_eq!(
            normalize(input, &NormalizerConfig::single_line()),
            "Hello World Again"
        );
    }

    #[test]
    fn test_preset_code_docs_preserves_newlines() {
        let input = "## Title\nSee https://docs.rs for **more**.\nNext line.";
        let out = normalize(input, &NormalizerConfig::code_docs());
        assert!(!out.contains("##"));
        assert!(!out.contains("https://"));
        assert!(!out.contains("**"));
        assert!(out.contains("Title"));
        assert!(out.contains("more"));
        // code_docs keeps join_soft_line_breaks: false — newlines preserved
        assert!(out.contains('\n'));
    }

    #[test]
    fn test_preset_web_scraped() {
        let input = "## Hot take\nVisit https://spam.com **now**!\n\nReal content here.";
        let out = normalize(input, &NormalizerConfig::web_scraped());
        assert!(!out.contains("##"));
        assert!(!out.contains("https://"));
        assert!(!out.contains("**"));
        assert!(out.contains("Hot take"));
        assert!(out.contains("Real content here"));
    }

    #[test]
    fn test_hyphenation_runs_before_line_collapse() {
        // fix_hyphenation must run before join_soft_line_breaks
        // so the rejoined word is clean, not "knowl- edgeledge"
        let input = "knowl-\nedge is power";
        let out = normalize(input, &NormalizerConfig::prose());
        assert!(out.contains("knowledge is power"), "got: {}", out);
    }
}
