"""The bundled Qwen3 tokenizer wrapper.

This module wraps the Qwen3 tokenizer shipped inside the SDK package
(byte-level BPE with good coverage of 100+ languages including CJK,
Thai, and Arabic). It performs all local text handling for jevner:
token counting, sliding-window splitting, and token-level character
anchoring.

What callers and maintainers need to know first:
- The tokenizer file lives at assets/qwen3_tokenizer.json inside the
  package, installed together with the SDK — users never touch
  HuggingFace and never need transformers or torch.
- Candidate spans do not depend on fixed n-grams: the detection stage
  judges token by token, and both region reassembly and sub-span
  generation work on token character offsets, treating unspaced
  scripts (Chinese/Thai) and agglutinative languages (Arabic/Malay)
  identically.
- Sliding windows mirror BertNER: windows are cut on token sequences
  and decoded back per window, so window boundaries always land on
  token edges and never split a multi-token entity mid-character.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

from tokenizers import Encoding, Tokenizer

_ASSET_PATH = Path(__file__).parent / "assets" / "qwen3_tokenizer.json"
"""Path to the bundled Qwen3 tokenizer file."""


def _is_edge_junk(ch: str) -> bool:
    """True for whitespace or punctuation stripped from entity edges
    (currency symbols and other non-P characters are kept)."""
    return ch.isspace() or unicodedata.category(ch)[0] == "P"


def strip_edge_punctuation(text: str) -> str:
    """Strip leading/trailing whitespace and punctuation from a string.

    Token-offset slices often pull adjacent punctuation into a
    candidate span (e.g. a name plus a comma), which pollutes entity text and
    breaks cross-span containment tests if left in place.

    Args:
        text: The candidate span text to clean.

    Returns:
        The cleaned text; possibly an empty string.
    """
    start, end = 0, len(text)
    while start < end and _is_edge_junk(text[start]):
        start += 1
    while end > start and _is_edge_junk(text[end - 1]):
        end -= 1
    return text[start:end]


def strip_span(s: str, start: int, end: int) -> tuple[str, int, int]:
    """Strip edge whitespace/punctuation from a range, moving offsets with it.

    Unlike strip_edge_punctuation, this operates on a source range and
    returns (cleaned text, new start, new end). The new offsets keep
    the invariant text == s[new_start:new_end], so entity text stays
    an exact substring of the source — the anchor for nesting-by-
    containment must track the text exactly.

    Args:
        s: The source text.
        start: Range start (inclusive), clamped to [0, len(s)].
        end: Range end (exclusive), clamped to [0, len(s)].

    Returns:
        A (cleaned text, cleaned start, cleaned end) triple.
    """
    start = max(0, min(start, len(s)))
    end = max(0, min(end, len(s)))
    while start < end and _is_edge_junk(s[start]):
        start += 1
    while end > start and _is_edge_junk(s[end - 1]):
        end -= 1
    return s[start:end], start, end


class BundledTokenizer:
    """A lightweight wrapper over the bundled Qwen3 tokenizer.

    Provides the local capabilities the jevner pipeline needs: token
    serialization, sliding-window splitting, and token character
    anchoring. Everything is deterministic, CPU-only local computation;
    no network requests are ever made.
    """

    def __init__(self, asset_path: Path = _ASSET_PATH) -> None:
        """Load the bundled tokenizer file.

        Args:
            asset_path: Path to tokenizer.json; defaults to the
                in-package asset.

        Raises:
            FileNotFoundError: The tokenizer asset is missing
                (corrupted distribution).
            Exception: Whatever the tokenizers library raises if the
                file fails to parse.
        """
        if not asset_path.exists():
            raise FileNotFoundError(f"Bundled tokenizer asset not found: {asset_path}")
        self._tokenizer = Tokenizer.from_file(str(asset_path))

    def encode(self, s: str) -> Encoding:
        """Encode text into a token sequence (no special tokens).

        Args:
            s: The text to encode.

        Returns:
            A tokenizers.Encoding with .ids and character-level .offsets.
        """
        return self._tokenizer.encode(s, add_special_tokens=False)

    def decode(self, ids: list[int]) -> str:
        """Decode a token sequence back into text.

        Args:
            ids: Token id list.

        Returns:
            The decoded text (special tokens skipped).
        """
        return self._tokenizer.decode(ids, skip_special_tokens=True)

    def count_tokens(self, s: str) -> int:
        """Count the tokens in a text.

        Args:
            s: The text to measure.

        Returns:
            The token count.
        """
        return len(self.encode(s).ids)

    def window_texts(self, s: str, window_size: int, stride: int) -> list[str]:
        """Split long text into sliding windows over the token sequence.

        Mirrors BertNER's _slow_extract: text whose token count fits
        within one window is returned as a single whole window;
        otherwise token slices advance by stride and each window is
        decoded back to text, with adjacent windows overlapping so an
        entity crossing a boundary appears fully in at least one.

        Args:
            s: The text to split.
            window_size: Window size in tokens.
            stride: Slide step in tokens; keep below window_size to
                retain overlap.

        Returns:
            Window texts in source order.
        """
        enc = self.encode(s)
        if len(enc.ids) <= window_size:
            return [s]
        texts: list[str] = []
        for start in range(0, len(enc.ids), stride):
            chunk = enc.ids[start : start + window_size]
            if not chunk:
                break
            texts.append(self.decode(chunk))
        return texts

    def window_tokens(self, s: str) -> list[tuple[int, int, str]]:
        """Return every token's character offsets and text.

        After token-level detection (one Noul per token), the code side
        uses this to rebuild contiguous entity regions at the character
        level — the language-agnostic foundation replacing fixed
        n-grams.

        Args:
            s: The window text.

        Returns:
            (char start, char end, token text) triples in text order.
        """
        enc = self.encode(s)
        return [(start, end, s[start:end]) for (start, end) in enc.offsets]

    def windows(
        self, s: str, window_size: int, stride: int
    ) -> list[tuple[str, list[tuple[int, int, str]]]]:
        """Split long text into sliding windows over the token sequence.

        Unlike window_texts, windows are cut directly from the source
        character range s[c_start:c_end] with no decode round-trip,
        and each window's token offsets are global source offsets
        (invariant: text == s[start:end]). Downstream spans therefore
        carry source offsets, and nesting derives reliably from offset
        containment.

        Token texts must never be re-joined into window text: Qwen3
        emits overlapping tokens for space-prefixed tokens (e.g. " 我")
        and joining duplicates characters, misaligning the detection
        state against its tokens. Window text must be sliced from the
        source character range.

        Splitting matches window_texts: token count within one window
        returns a single whole window; otherwise slices advance by
        stride, window text covers the first through last token's
        character range, and adjacent windows overlap.

        Args:
            s: The text to split.
            window_size: Window size in tokens.
            stride: Slide step in tokens; keep below window_size to
                retain overlap.

        Returns:
            (window text, window tokens) pairs; tokens carry source
            offsets (start, end, text).
        """
        enc = self.encode(s)
        offsets = enc.offsets
        if len(offsets) <= window_size:
            return [(s, [(start, end, s[start:end]) for start, end in offsets])]
        result: list[tuple[str, list[tuple[int, int, str]]]] = []
        for base in range(0, len(offsets), stride):
            chunk = offsets[base : base + window_size]
            if not chunk:
                break
            c_start, c_end = chunk[0][0], chunk[-1][1]
            window_text = s[c_start:c_end]
            result.append((window_text, [(start, end, s[start:end]) for start, end in chunk]))
        return result
