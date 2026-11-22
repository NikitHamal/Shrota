"""Canonical NFC Devanagari & Sanskrit normalization + punctuation hygiene.

Exact 1:1 parity with docs/norm.js for browser WASM inference.
"""
import re
import unicodedata

# Allowed clean punctuation set for <punct> mode
ALLOWED_PUNCT = {"।", ",", "?", "!", "-"}

# Map variant dandas, quotes, ASCII periods at clause end, and zero-width junk
_CHAR_MAP = {
    "\u200b": "",  # ZWSP
    "\u200c": "",  # ZWNJ
    "\u200d": "",  # ZWJ (strip stray ZWJ for consistent acoustic-grapheme mapping)
    "\ufeff": "",  # BOM
    "\u00a0": " ",
    "\u0965": "।",  # Double danda -> single danda for standard ASR punctuation
    "|": "।",
    "॥": "।",
    "–": "-",
    "—": "-",
    "’": "'",
    "‘": "'",
    "“": '"',
    "”": '"',
}

# Precompiled regexes
_MULTI_SPACE_RE = re.compile(r"\s+")
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
# Remove Vedic pitch accents (udatta/anudatta/svarita) for unified acoustic ASR text
_VEDIC_ACCENT_RE = re.compile(r"[\u0951-\u0954\u1CD0-\u1CFF\uA8E0-\uA8FF]")
_PUNCT_REPEAT_RE = re.compile(r"([।,?!\-])\s*\1+")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([।,?!])")
_PUNCT_NO_SPACE_AFTER_RE = re.compile(r"([।,?!])([^\s।,?!])")


def normalize_text(text: str, keep_punct: bool = True, strip_accents: bool = True) -> str:
    """Canonical NFC normalization for Nepali & Sanskrit ASR transcripts."""
    if not text:
        return ""
    s = unicodedata.normalize("NFC", str(text))
    for k, v in _CHAR_MAP.items():
        if k in s:
            s = s.replace(k, v)
    if strip_accents:
        s = _VEDIC_ACCENT_RE.sub("", s)

    # Convert trailing period after Devanagari character to danda
    s = re.sub(r"([\u0900-\u097F])\s*\.(?=\s|$)", r"\1।", s)

    out_chars = []
    for ch in s:
        cat = unicodedata.category(ch)
        if ch in ALLOWED_PUNCT:
            if keep_punct:
                out_chars.append(ch)
            else:
                out_chars.append(" ")
        elif cat.startswith(("L", "M", "N")):
            # Normalize ASCII digits 0-9 to canonical Devanagari digits ०-९
            if "0" <= ch <= "9":
                out_chars.append(chr(0x0966 + ord(ch) - ord("0")))
            else:
                out_chars.append(ch)
        elif ch.isspace():
            out_chars.append(" ")
        else:
            # Drop stray symbols/brackets/quotes
            out_chars.append(" ")

    s = "".join(out_chars)
    if keep_punct:
        s = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", s)
        s = _PUNCT_REPEAT_RE.sub(r"\1", s)
        s = _PUNCT_NO_SPACE_AFTER_RE.sub(r"\1 \2", s)
    s = _MULTI_SPACE_RE.sub(" ", s).strip()
    if not keep_punct:
        # Strip any leftover hyphens
        s = s.replace("-", " ")
        s = _MULTI_SPACE_RE.sub(" ", s).strip()
    return s


def devanagari_ratio(text: str) -> float:
    """Return fraction of alphabetic/syllabic characters that are Devanagari."""
    s = normalize_text(text, keep_punct=False)
    letters = [c for c in s if not c.isspace()]
    if not letters:
        return 0.0
    dev = sum(1 for c in letters if "\u0900" <= c <= "\u097f")
    return dev / len(letters)


def has_punctuation(text: str) -> bool:
    """Return True if text contains genuine sentence/clause punctuation."""
    return any(p in text for p in ("।", "?", "!", ","))


def tag_utterance(text: str, lang: str = "ne", force_mode: str | None = None) -> str:
    """Prefix normalized text with mode tokens: <ne>/<sa> and <punct>/<nopunct>."""
    raw_has_punct = has_punctuation(text)
    mode = force_mode or ("punct" if raw_has_punct else "nopunct")
    normed = normalize_text(text, keep_punct=(mode == "punct"))
    lang_tok = "<sa>" if lang == "sa" else "<ne>"
    punct_tok = "<punct>" if mode == "punct" else "<nopunct>"
    return f"{lang_tok} {punct_tok} {normed}".strip()
