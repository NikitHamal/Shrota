"""Devanagari Akshara-Aware 1,024-Token Subword Tokenizer for Nepali + Sanskrit ASR.

Supports deterministic greedy longest-match / Unigram-style subword encoding with
zero external runtime dependencies in Browser WASM / Android Kotlin / Python, plus
optional SentencePiece export.
"""
from __future__ import annotations
import json
import os
import re
from collections import Counter
from typing import Iterable

from shrota.norm import normalize_text, ALLOWED_PUNCT

SPECIAL_TOKENS = [
    "<blk>",      # 0: CTC / TDT blank token
    "<unk>",      # 1: unknown fallback
    "<bos>",      # 2: start of sequence
    "<eos>",      # 3: end of sequence
    "<ne>",       # 4: Nepali mode
    "<sa>",       # 5: Sanskrit / Vedic mode
    "<punct>",    # 6: Punctuated output mode
    "<nopunct>",  # 7: Unpunctuated output mode
]

WORD_BOUNDARY = "▁"

# Devanagari orthographic akshara regex: optional consonant+halant clusters + base char + dependent matras/signs
_AKSHARA_RE = re.compile(
    r"(?:[\u0915-\u0939\u0958-\u095F]\u094D)*"
    r"[\u0904-\u0939\u0958-\u0961\u0972-\u097F]"
    r"[\u0900-\u0903\u093A-\u094F\u0955-\u0957\u0962-\u0963]*"
    r"|[\u0966-\u096F]"
    r"|[a-zA-Z0-9]"
    r"|."
)


def split_aksharas(word: str) -> list[str]:
    """Split a word into orthographic Devanagari aksharas (syllabic clusters) or chars."""
    return [m.group(0) for m in _AKSHARA_RE.finditer(word) if m.group(0)]


class ShrotaTokenizer:
    """Compact 1,024-token Akshara-aware subword tokenizer."""

    def __init__(self, vocab: list[str]):
        self.vocab = list(vocab)
        self.token_to_id = {t: i for i, t in enumerate(self.vocab)}
        self.id_to_token = {i: t for i, t in enumerate(self.vocab)}
        self.blank_id = self.token_to_id["<blk>"]
        self.unk_id = self.token_to_id["<unk>"]
        self.bos_id = self.token_to_id["<bos>"]
        self.eos_id = self.token_to_id["<eos>"]
        self.special_set = set(SPECIAL_TOKENS)
        self.max_piece_len = max((len(t) for t in self.vocab if t not in self.special_set), default=1)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @classmethod
    def build_from_texts(cls, texts: Iterable[str], vocab_size: int = 1024) -> "ShrotaTokenizer":
        """Train an Akshara-aligned 1,024-token vocabulary from Nepali & Sanskrit corpora."""
        vocab_list = list(SPECIAL_TOKENS)
        seen = set(vocab_list)

        def add_tok(t: str):
            if t and t not in seen and len(vocab_list) < vocab_size:
                seen.add(t)
                vocab_list.append(t)

        # 1. Guarantee all core Devanagari codepoints, matras, digits, punctuation, and word-boundary prefix
        add_tok(WORD_BOUNDARY)
        for p in sorted(ALLOWED_PUNCT):
            add_tok(p)
            add_tok(WORD_BOUNDARY + p)
        for cp in range(0x0900, 0x0980):
            ch = chr(cp)
            add_tok(ch)
        for ch in "abcdefghijklmnopqrstuvwxyz0123456789":
            add_tok(ch)
            add_tok(WORD_BOUNDARY + ch)

        # 2. Count word-initial and word-internal aksharas, bigrams, and frequent whole words/postpositions
        piece_counts: Counter[str] = Counter()
        word_counts: Counter[str] = Counter()

        # High-priority Nepali grammatical suffixes/postpositions & common function words
        priority_pieces = [
            "को", "का", "की", "मा", "ले", "लाई", "बाट", "देखि", "सँग", "द्वारा",
            "हरू", "हरूको", "हरूलाई", "हरूमा", "छ", "छन्", "थियो", "थिए", "हुन्",
            "हुन्छ", "गर्नु", "गरेको", "गर्दै", "भने", "पनि", "तर", "अनि", "र", "वा",
            "▁नेपाल", "▁सरकार", "▁काठमाडौँ", "▁भने", "▁पनि", "▁गरेको", "▁भएको",
            "▁छ।", "▁छन्।", "▁थियो।", "▁हो।", "▁गर्छन्।", "▁बताए।", "▁जानकारी",
            "▁विद्यालय", "▁तिमी", "▁हामी", "▁तपाईँ", "▁उहाँ", "▁मानिस", "▁कारण",
            "▁अनुसार", "▁अवस्था", "▁विकास", "▁शिक्षा", "▁स्वास्थ्य", "▁समाज",
            "▁ओम्", "▁नमः", "▁स्वाहा", "▁भवति", "▁उवाच", "▁अग्निमीळे", "▁पुरोहितं",
        ]
        for pp in priority_pieces:
            add_tok(pp)

        for raw in texts:
            norm = normalize_text(raw, keep_punct=True)
            if not norm:
                continue
            for w in norm.split():
                if w in ALLOWED_PUNCT or w in SPECIAL_TOKENS:
                    continue
                # Strip trailing punct from word for clean subword counting
                while w and w[-1] in ALLOWED_PUNCT:
                    w = w[:-1]
                if not w:
                    continue
                pw = WORD_BOUNDARY + w
                word_counts[pw] += 1
                aks = split_aksharas(w)
                if not aks:
                    continue
                aks_pref = [WORD_BOUNDARY + aks[0]] + aks[1:]
                for a in aks_pref:
                    piece_counts[a] += 2
                for i in range(len(aks_pref) - 1):
                    bg = aks_pref[i] + aks_pref[i + 1]
                    if len(bg) <= 8:
                        piece_counts[bg] += 1
                if len(aks_pref) >= 3:
                    for i in range(len(aks_pref) - 2):
                        tg = aks_pref[i] + aks_pref[i + 1] + aks_pref[i + 2]
                        if len(tg) <= 10:
                            piece_counts[tg] += 1

        # Add top frequent whole words (up to 260 slots) then top akshara n-grams
        for w, _ in word_counts.most_common(400):
            if 2 <= len(w) <= 10:
                add_tok(w)
            if len(vocab_list) >= vocab_size - 420:
                break

        for piece, _ in piece_counts.most_common():
            if len(vocab_list) >= vocab_size:
                break
            add_tok(piece)

        # Pad deterministically if corpus was small
        idx = 0
        while len(vocab_list) < vocab_size:
            add_tok(f"<extra_{idx}>")
            idx += 1

        return cls(vocab_list[:vocab_size])

    def encode(
        self,
        text: str,
        lang: str | None = None,
        punct_mode: str | None = None,
        add_bos_eos: bool = False,
    ) -> list[int]:
        """Encode text into token IDs."""
        ids: list[int] = []
        if add_bos_eos:
            ids.append(self.bos_id)
        if lang in ("ne", "sa"):
            ids.append(self.token_to_id[f"<{lang}>"])
        if punct_mode in ("punct", "nopunct"):
            ids.append(self.token_to_id[f"<{punct_mode}>"])

        keep_p = punct_mode != "nopunct"
        s = normalize_text(text, keep_punct=keep_p)
        if not s:
            if add_bos_eos:
                ids.append(self.eos_id)
            return ids

        for word in s.split():
            if word in self.special_set:
                ids.append(self.token_to_id[word])
                continue
            pw = WORD_BOUNDARY + word
            pos = 0
            n = len(pw)
            while pos < n:
                end = min(n, pos + self.max_piece_len)
                matched = None
                while end > pos:
                    cand = pw[pos:end]
                    if cand in self.token_to_id:
                        matched = self.token_to_id[cand]
                        pos = end
                        break
                    end -= 1
                if matched is not None:
                    ids.append(matched)
                else:
                    ch = pw[pos]
                    tid = self.token_to_id.get(ch)
                    if tid is None and ch.lower() in self.token_to_id:
                        tid = self.token_to_id[ch.lower()]
                    ids.append(tid if tid is not None else self.unk_id)
                    pos += 1

        if add_bos_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        """Decode token IDs back into normalized Devanagari text."""
        pieces: list[str] = []
        for idx in ids:
            t = self.id_to_token.get(int(idx), "")
            if not t:
                continue
            if t in self.special_set or t.startswith("<extra_"):
                if not skip_special and t not in ("<blk>",):
                    pieces.append(f" {t} ")
                continue
            pieces.append(t)
        raw = "".join(pieces).replace(WORD_BOUNDARY, " ").strip()
        raw = re.sub(r"\s+([।,?!])", r"\1", raw)
        raw = re.sub(r"\s+", " ", raw)
        return raw.strip()

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"vocab": self.vocab}, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "ShrotaTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(data["vocab"])
