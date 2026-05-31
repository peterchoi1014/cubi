//! Strips `<think>...</think>` reasoning blocks from assistant content.
//!
//! Qwen3 (and a few other "thinking-mode" models) emit chain-of-thought
//! wrapped in `<think>...</think>` tags as part of the visible content
//! stream. We never want that text shown to the user or persisted into
//! history: it's both privacy-sensitive (the model is reasoning out loud
//! about what to do with the user's data) and confusing context for the
//! next turn (the model treats prior `<think>` text as a continuation
//! prompt rather than its own reasoning).
//!
//! Two surfaces:
//!
//! - [`strip_thinking_blocks`] for the non-streaming code path: feed it
//!   the assembled assistant content and get back the user-visible
//!   substring.
//! - [`ThinkStripper`] for the streaming code path: feed it raw deltas
//!   and emit the user-visible portion. Tag boundaries can fall across
//!   chunk boundaries (e.g. `…<thi` then `nk>…`); the stripper buffers
//!   the minimum suffix needed to handle that without delaying plain
//!   text output.

const OPEN: &str = "<think>";
const CLOSE: &str = "</think>";

/// One-shot stripper for non-streaming assistant content.
///
/// Removes every `<think>…</think>` block (inclusive of tags). Unclosed
/// `<think>` at the end of input is dropped along with everything after
/// it — that's the desired behavior for the case where a model crashed
/// or was cancelled mid-thought.
///
/// Honors the `CUBI_KEEP_THINKING=1` escape hatch: set it to disable
/// stripping entirely (useful when the user explicitly wants to see the
/// model's reasoning, or when their content legitimately contains
/// literal `<think>` tags in e.g. code samples about prompt formats).
pub fn strip_thinking_blocks(s: &str) -> String {
    if keep_thinking() {
        return s.to_string();
    }
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(start) = rest.find(OPEN) {
        out.push_str(&rest[..start]);
        let after_open = &rest[start + OPEN.len()..];
        match after_open.find(CLOSE) {
            Some(end) => {
                rest = &after_open[end + CLOSE.len()..];
            }
            None => {
                // Unclosed — drop the rest.
                return out;
            }
        }
    }
    out.push_str(rest);
    out
}

fn keep_thinking() -> bool {
    std::env::var("CUBI_KEEP_THINKING")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// Streaming stripper. Construct one per stream, call [`feed`] for each
/// delta, and call [`flush`] when the stream ends.
///
/// Internally tracks whether we're inside a `<think>` block and buffers
/// the smallest possible suffix that might be a partial tag, so that
/// plain text streams are emitted with at most ~8 characters of
/// latency.
#[derive(Default)]
pub struct ThinkStripper {
    inside: bool,
    pending: String,
}

impl ThinkStripper {
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether the stripper is currently inside a `<think>` block. The
    /// caller can use this to suppress UI side-effects (spinners,
    /// timers) while reasoning is in progress.
    #[allow(dead_code)]
    pub fn is_thinking(&self) -> bool {
        self.inside
    }

    /// Feeds the next streaming delta and returns the portion that's
    /// safe to emit to the user. Always returns the empty string when
    /// the delta is fully inside a `<think>` block (or buffered as a
    /// possible partial tag).
    ///
    /// Honors `CUBI_KEEP_THINKING=1`: passes input through verbatim.
    pub fn feed(&mut self, chunk: &str) -> String {
        if keep_thinking() {
            return chunk.to_string();
        }
        let mut input = std::mem::take(&mut self.pending);
        input.push_str(chunk);
        let mut out = String::with_capacity(input.len());
        let mut rest = input.as_str();
        loop {
            if self.inside {
                if let Some(end) = rest.find(CLOSE) {
                    rest = &rest[end + CLOSE.len()..];
                    self.inside = false;
                    // Continue the loop to check for a new <think> in
                    // the remainder.
                } else {
                    // Keep a suffix that might be the start of </think>.
                    self.pending = tail_after_last_lt(rest, CLOSE).to_string();
                    return out;
                }
            } else if let Some(start) = rest.find(OPEN) {
                out.push_str(&rest[..start]);
                rest = &rest[start + OPEN.len()..];
                self.inside = true;
            } else {
                // No tag in this slice. Emit everything except a
                // suffix that might be the start of <think>.
                let tail = tail_after_last_lt(rest, OPEN);
                let safe_len = rest.len() - tail.len();
                out.push_str(&rest[..safe_len]);
                self.pending = tail.to_string();
                return out;
            }
        }
    }

    /// Flushes any final buffered bytes when the stream ends. Buffered
    /// bytes that were inside a `<think>` block are discarded.
    pub fn flush(&mut self) -> String {
        let pending = std::mem::take(&mut self.pending);
        if self.inside {
            // Unclosed <think> at end-of-stream — drop it.
            self.inside = false;
            String::new()
        } else {
            pending
        }
    }
}

/// Returns the suffix of `s` starting at the last `'<'` if and only if
/// that suffix is a strict prefix of `tag` and shorter than `tag` (so
/// it could still grow into the complete tag). Otherwise returns an
/// empty suffix.
///
/// This is the workhorse that keeps streaming latency minimal: plain
/// text without a `'<'` near the tail — or with a `'<'` that already
/// can't be the start of the tag we're hunting — is emitted
/// immediately.
fn tail_after_last_lt<'a>(s: &'a str, tag: &str) -> &'a str {
    if s.is_empty() {
        return "";
    }
    if let Some(pos) = s.rfind('<') {
        let suffix = &s[pos..];
        if suffix.len() < tag.len() && tag.starts_with(suffix) {
            return suffix;
        }
    }
    ""
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_shot_strips_single_block() {
        assert_eq!(
            strip_thinking_blocks("hi <think>secret</think> there"),
            "hi  there"
        );
    }

    #[test]
    fn one_shot_strips_multiple_blocks() {
        assert_eq!(
            strip_thinking_blocks("<think>a</think>X<think>b</think>Y"),
            "XY"
        );
    }

    #[test]
    fn one_shot_drops_unclosed_block() {
        assert_eq!(strip_thinking_blocks("ok <think>incomplete..."), "ok ");
    }

    #[test]
    fn one_shot_passes_through_plain_text() {
        assert_eq!(strip_thinking_blocks("just <code>"), "just <code>");
        assert_eq!(strip_thinking_blocks(""), "");
    }

    #[test]
    fn streaming_emits_clean_text_immediately() {
        let mut s = ThinkStripper::new();
        // No '<' in the tail → nothing buffered.
        assert_eq!(s.feed("hello world\n"), "hello world\n");
        assert_eq!(s.flush(), "");
    }

    #[test]
    fn streaming_strips_block_split_across_chunks() {
        let mut s = ThinkStripper::new();
        assert_eq!(s.feed("hi <th"), "hi ");
        assert_eq!(s.feed("ink>secret reason"), "");
        assert_eq!(s.feed("ing</think> there"), " there");
        assert_eq!(s.flush(), "");
    }

    #[test]
    fn streaming_handles_close_tag_split() {
        let mut s = ThinkStripper::new();
        assert_eq!(s.feed("<think>foo</th"), "");
        assert_eq!(s.feed("ink>bar"), "bar");
    }

    #[test]
    fn streaming_handles_block_in_single_chunk() {
        let mut s = ThinkStripper::new();
        assert_eq!(s.feed("a<think>x</think>b"), "ab");
    }

    #[test]
    fn streaming_handles_multiple_blocks_in_one_chunk() {
        let mut s = ThinkStripper::new();
        assert_eq!(s.feed("<think>1</think>X<think>2</think>Y"), "XY");
    }

    #[test]
    fn streaming_flush_drops_unclosed_block() {
        let mut s = ThinkStripper::new();
        assert_eq!(s.feed("ok <think>still thinking..."), "ok ");
        assert_eq!(s.flush(), "");
        assert!(!s.is_thinking());
    }

    #[test]
    fn streaming_flush_emits_pending_when_not_inside() {
        let mut s = ThinkStripper::new();
        // "<" alone might be a partial tag → buffered.
        assert_eq!(s.feed("hello <"), "hello ");
        // Stream ends without more bytes → emit the buffered '<'.
        assert_eq!(s.flush(), "<");
    }

    #[test]
    fn streaming_preserves_lone_lt_when_clearly_not_a_tag() {
        let mut s = ThinkStripper::new();
        // '<' followed by non-'t' chars is not the start of <think>; the
        // next chunk should disambiguate. We do buffer briefly while
        // the suffix is still short enough to be ambiguous.
        let mut got = String::new();
        got.push_str(&s.feed("a < b"));
        got.push_str(&s.feed(" c\n"));
        got.push_str(&s.flush());
        assert_eq!(got, "a < b c\n");
    }

    #[test]
    fn streaming_handles_utf8_safely() {
        // Multi-byte chars don't appear inside the tag, but must be
        // preserved verbatim in surrounding text.
        let mut s = ThinkStripper::new();
        let out = s.feed("café <think>π</think> 你好");
        assert_eq!(out, "café  你好");
    }

    #[test]
    fn one_shot_is_no_op_when_no_tags() {
        let input = "Just a regular response with no thinking tags.";
        assert_eq!(strip_thinking_blocks(input), input);
    }
}
