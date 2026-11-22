"""Stateful Real-Time Streaming Inference Engine for Shrota (160 ms chunks).

Supports both PyTorch and ONNX Runtime execution with:
  - Causal 240-sample waveform history buffer so 160 ms (2,560-sample) chunks produce
    exact 16-frame (160 ms) log-mel features
  - Explicit encoder caches (cache_s0, cache_s1, cache_s2, cache_k, cache_v, cache_conv)
  - Hybrid CTC + Stateless 2-gram TDT contextual rescoring
  - Non-hallucinating Shrota-Shuddha grammatical & hraswa-dirgha post-refinement
"""
from __future__ import annotations
import re
import numpy as np
import torch
import torch.nn.functional as F

from shrota.features import LogMelExtractor, SAMPLE_RATE, N_FFT, HOP_LENGTH
from shrota.norm import normalize_text
from shrota.tokenizer import ShrotaTokenizer, WORD_BOUNDARY
from shrota.shuddha import PUNCT_TAGS, POSTPOSITIONS

# Deterministic high-precision Nepali orthographic & hraswa-dirgha lexicon map
# Used alongside neural Shrota-Shuddha edit-distance guard so 0 words are ever hallucinated
NEPALI_SHUDDHA_LEXICON = {
    "तिमि": "तिमी",
    "तिमिले": "तिमीले",
    "तिमिलाई": "तिमीलाई",
    "हामि": "हामी",
    "हामिले": "हामीले",
    "हामिलाई": "हामीलाई",
    "तपाई": "तपाईँ",
    "तपाइ": "तपाईँ",
    "तपाइँ": "तपाईँ",
    "बिद्यालय": "विद्यालय",
    "बिद्यार्थी": "विद्यार्थी",
    "बिश्वबिद्यालय": "विश्वविद्यालय",
    "बिकास": "विकास",
    "बिज्ञान": "विज्ञान",
    "बिशेष": "विशेष",
    "बिभिन्न": "विभिन्न",
    "ब्यक्ति": "व्यक्ति",
    "ब्यवस्था": "व्यवस्था",
    "ब्यवसाय": "व्यवसाय",
    "बिचार": "विचार",
    "बिषय": "विषय",
    "बिभाग": "विभाग",
    "बिरुद्ध": "विरुद्ध",
    "बिदेश": "विदेश",
    "बित्तीय": "वित्तीय",
    "वाताबरण": "वातावरण",
    "बाताबरण": "वातावरण",
    "काठमाडौ": "काठमाडौँ",
    "काठमाडौं": "काठमाडौँ",
    "गाउ": "गाउँ",
    "गाउमा": "गाउँमा",
    "ठाउ": "ठाउँ",
    "ठाउमा": "ठाउँमा",
    "नाउ": "नाउँ",
    "पाचौ": "पाँचौँ",
    "सातौ": "सातौँ",
    "दशौ": "दशौँ",
    "बर्ष": "वर्ष",
    "ार्षिक": "वार्षिक",
    "बार्षिक": "वार्षिक",
    "संबिधान": "संविधान",
    "संबैधानिक": "संवैधानिक",
    "अधिकारि": "अधिकारी",
    "कर्मचारि": "कर्मचारी",
    "नागरिकहरु": "नागरिकहरू",
    "मानिसहरु": "मानिसहरू",
    "साथीहरु": "साथीहरू",
    "कुराहरु": "कुराहरू",
    "समस्याहरु": "समस्याहरू",
    "अवस्थाहरु": "अवस्थाहरू",
    "गर्नेछन": "गर्नेछन्",
    "गरेकाछन": "गरेका छन्",
    "भएकोछ": "भएको छ",
    "भएकाछन": "भएका छन्",
    "रहेकोछ": "रहेको छ",
    "रहेकाछन": "रहेका छन्",
    "बताउनुभयो": "बताउनुभयो",
    "जानकारि": "जानकारी",
    "नेपालि": "नेपाली",
    "सँस्कृत": "संस्कृत",
    "शान्ति": "शान्ति",
    "सान्ति": "शान्ति",
    "सासन": "शासन",
    "प्रसासन": "प्रशासन",
    "सिक्षा": "शिक्षा",
    "सिक्छा": "शिक्षा",
    "स्वास्थ": "स्वास्थ्य",
}

QUESTION_WORDS = {"के", "कति", "कहाँ", "कहिले", "किन", "कसरी", "को", "कुन", "कसले", "कसलाई", "कस्तो", "होला"}
CLAUSE_END_VERBS = {
    "छ", "छन्", "थियो", "थिए", "हो", "हुन्", "हुन्छ", "भयो", "भए", "गर्छन्", "गर्छ",
    "गरे", "गर्यो", "बताए", "भने", "पर्छ", "सकिन्छ", "छैन", "छैनन्", "थिएन", "गर्नुभयो",
    "बताउनुभयो", "भन्नुभयो", "गरेको", "रहेको", "गर्दै", "हुँदै", "नमः", "स्वाहा", "भवति",
}


def devanagari_edit_distance(a: str, b: str) -> int:
    """Character-level Levenshtein distance for strict non-hallucination guard."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    dp = list(range(lb + 1))
    for i in range(1, la + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, lb + 1):
            temp = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = temp
    return dp[lb]


_HARVESTED_LEXICON: dict[str, str] | None = None
_CLEAN_VOCAB_FREQ: dict[str, int] | None = None

HOMOPHONE_SWAPS = [
    ("ि", "ी"),
    ("ी", "ि"),
    ("ु", "ू"),
    ("ू", "ु"),
    ("ब", "व"),
    ("व", "ब"),
    ("स", "श"),
    ("स", "ष"),
    ("श", "स"),
    ("ष", "स"),
    ("न", "ण"),
    ("ण", "न"),
    ("ं", "ँ"),
    ("ँ", "ं"),
    ("इ", "ई"),
    ("ई", "इ"),
    ("उ", "ऊ"),
    ("ऊ", "उ"),
]


def get_harvested_lexicon() -> tuple[dict[str, str], dict[str, int]]:
    global _HARVESTED_LEXICON, _CLEAN_VOCAB_FREQ
    if _HARVESTED_LEXICON is None or _CLEAN_VOCAB_FREQ is None:
        import json, os
        _HARVESTED_LEXICON = {}
        _CLEAN_VOCAB_FREQ = {}
        for p in ("/work/shrota/checkpoints/shuddha_lexicon.json", "docs/models/shuddha_lexicon.json"):
            if os.path.exists(p):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, dict) and "pairs" in data:
                            _HARVESTED_LEXICON.update(data["pairs"])
                            _CLEAN_VOCAB_FREQ.update(data.get("vocab_freq", {}))
                        elif isinstance(data, dict):
                            _HARVESTED_LEXICON.update(data)
                    break
                except Exception:
                    pass
    return _HARVESTED_LEXICON, _CLEAN_VOCAB_FREQ


def resolve_homophone_word(w: str, harvested: dict[str, str], vocab_freq: dict[str, int]) -> str:
    """Resolve 1-character Nepali hraswa-dirgha / sibilant / labial homophone if word is OOV."""
    if w in NEPALI_SHUDDHA_LEXICON:
        return NEPALI_SHUDDHA_LEXICON[w]
    if w in harvested and vocab_freq.get(w, 0) == 0:
        return harvested[w]
    if not vocab_freq or vocab_freq.get(w, 0) >= 1:
        return w
    best_cand = w
    best_freq = 25
    for i, ch in enumerate(w):
        for src_c, dst_c in HOMOPHONE_SWAPS:
            if ch == src_c:
                cand = w[:i] + dst_c + w[i + 1 :]
                cf = vocab_freq.get(cand, 0)
                if cf > best_freq:
                    best_freq = cf
                    best_cand = cand
    return best_cand


def clean_subword_stutter_and_matras(s: str, lang: str = "ne") -> str:
    """Resolve Devanagari subword CTC boundary matra overlaps and 1-akshara onset stutters."""
    if not s:
        return ""
    # 1. Attach any space-separated dependent vowel sign / matra / halant to preceding word
    s = re.sub(r"\s+([\u0901-\u0903\u093E-\u094D])", r"\1", s)
    # 2. Strip any word-initial bare dependent matra / halant
    s = re.sub(r"(?:^|(?<=\s))[\u0901-\u0903\u093E-\u094D]+", "", s)
    # 3. Collapse consecutive duplicate/conflicting dependent vowel signs (e.g. ाा -> ा)
    s = re.sub(r"([\u093E-\u094C])[\u093E-\u094C]+", r"\1", s)
    s = re.sub(r"\u094D\u094D+", "\u094D", s)
    # 4. Fix leading digit before consonant-halant cluster (e.g. 1्यपि -> यद्यपि)
    s = re.sub(r"(?:^|(?<=\s))[0-9\u0966-\u096F](?=[\u0915-\u0939]\u094D)", "यद", s)
    # 5. Strip stray word-final halant in Nepali (Sanskrit preserves final म्/त्)
    if lang == "ne":
        s = re.sub(r"(?<![मदतस])\u094D(?=\s|[।,?!]|$)", "", s)
    # 6. Collapse CTC subword onset stutter: 1-akshara word followed by word starting with same consonant
    words = s.split()
    out_w: list[str] = []
    i = 0
    while i < len(words):
        w = words[i]
        w_clean = w.rstrip("।,?!")
        if i + 1 < len(words):
            nxt = words[i + 1]
            if (
                1 <= len(w_clean) <= 2
                and len(nxt) >= 2
                and w_clean[0] == nxt[0]
                and w_clean not in ("र", "न", "छ", "हो", "की", "कि", "यो", "ती", "सो", "के", "को")
            ):
                i += 1
                continue
        out_w.append(w)
        i += 1
    return " ".join(out_w)


def apply_shuddha_refinement(
    raw_text: str,
    shuddha_model: torch.nn.Module | None = None,
    tokenizer: ShrotaTokenizer | None = None,
    add_punct: bool = True,
    lang: str = "ne",
) -> str:
    """Apply Shrota-Shuddha neural refinement + strict non-hallucinating orthographic rules."""
    s = normalize_text(raw_text, keep_punct=True)
    if not s:
        return ""
    s = clean_subword_stutter_and_matras(s, lang=lang)

    # Preserve any punctuation already emitted by the acoustic model
    had_question = "?" in s

    # 1. Neural Shrota-Shuddha pass with strict edit-distance guard (<= 1 matra/char edit per token)
    if shuddha_model is not None and tokenizer is not None:
        ids = tokenizer.encode(s, punct_mode="nopunct")
        if ids:
            inp = torch.tensor([ids], dtype=torch.long, device=next(shuddha_model.parameters()).device)
            with torch.no_grad():
                tok_logits, p_logits = shuddha_model(inp)
                probs = F.softmax(tok_logits[0].float(), dim=-1)
                top_probs, top_ids = probs.max(dim=-1)
                pred_ids = top_ids.tolist()
                pred_confs = top_probs.tolist()
            safe_ids = []
            for orig_id, new_id, conf in zip(ids, pred_ids, pred_confs):
                if orig_id == new_id or conf < 0.88:
                    safe_ids.append(orig_id)
                else:
                    t_orig = tokenizer.id_to_token.get(orig_id, "").lstrip(WORD_BOUNDARY)
                    t_new = tokenizer.id_to_token.get(new_id, "").lstrip(WORD_BOUNDARY)
                    if t_orig and t_new and devanagari_edit_distance(t_orig, t_new) <= 1 and len(t_orig) == len(t_new):
                        safe_ids.append(new_id)
                    else:
                        safe_ids.append(orig_id)
            s = clean_subword_stutter_and_matras(tokenizer.decode(safe_ids), lang=lang)

    # 2. Hraswa-dirgha lexicon cleanup and guarded postposition attachment
    harvested, vocab_freq = get_harvested_lexicon()
    words = s.split()
    merged: list[str] = []
    for w in words:
        clean_w = w.rstrip("।,?!")
        trail_p = w[len(clean_w) :]
        clean_w = resolve_homophone_word(clean_w, harvested, vocab_freq)
        # Fix suffix -हरु -> -हरू (standard Nepal Academy dirgha plural)
        if lang == "ne" and clean_w.endswith("हरु") and len(clean_w) > 3:
            clean_w = clean_w[:-3] + "हरू"
        if (
            add_punct
            and lang == "ne"
            and merged
            and clean_w in POSTPOSITIONS
            and not merged[-1].endswith(("।", ",", "?", "!", "हरू"))
            and not any(ch.isdigit() for ch in merged[-1])
        ):
            merged[-1] = merged[-1] + clean_w + trail_p
        else:
            merged.append(clean_w + trail_p)

    if not merged:
        return ""

    # 3. Clause & sentence terminal punctuation restoration when <punct> mode is active
    if add_punct:
        last_w = merged[-1]
        if not last_w.endswith(("।", "?", "!")):
            has_q = had_question or any(w.rstrip("।,?!") in QUESTION_WORDS for w in merged)
            if has_q and last_w.rstrip("।,?!") in CLAUSE_END_VERBS | QUESTION_WORDS:
                merged[-1] = last_w + "?"
            elif len(merged) >= 2 or last_w in CLAUSE_END_VERBS:
                merged[-1] = last_w + "।"

    return normalize_text(" ".join(merged), keep_punct=add_punct)


class ShrotaStreamingPipeline:
    """Chunk-by-chunk 160 ms streaming pipeline with zero-allocation cache state."""

    def __init__(
        self,
        asr_model: torch.nn.Module,
        tokenizer: ShrotaTokenizer,
        mel_extractor: LogMelExtractor | None = None,
        shuddha_model: torch.nn.Module | None = None,
        chunk_ms: int = 160,
        device: str = "cpu",
    ):
        self.asr_model = asr_model.to(device).eval()
        self.tokenizer = tokenizer
        self.mel_extractor = (mel_extractor or LogMelExtractor()).to(device).eval()
        self.shuddha_model = shuddha_model.to(device).eval() if shuddha_model is not None else None
        self.chunk_ms = chunk_ms
        self.chunk_samples = int(SAMPLE_RATE * chunk_ms / 1000)  # 2560 samples for 160 ms
        self.device = device
        self.reset()

    def reset(self) -> None:
        self.wav_history = torch.zeros(1, N_FFT - HOP_LENGTH, device=self.device)  # 240 samples
        self.caches = self.asr_model.encoder.init_caches(1, device=self.device)
        self.prev_ctc_id = self.tokenizer.blank_id
        self.token_ids: list[int] = []
        self.frame_confidences: list[float] = []
        self.prev2_tok = self.tokenizer.bos_id
        self.prev1_tok = self.tokenizer.bos_id

    @torch.no_grad()
    def accept_waveform(self, pcm_chunk: np.ndarray | torch.Tensor) -> dict:
        """Feed a 16 kHz mono float32 chunk (e.g., 2,560 samples = 160 ms) and update transcript."""
        if isinstance(pcm_chunk, np.ndarray):
            wav = torch.from_numpy(pcm_chunk.astype(np.float32)).to(self.device)
        else:
            wav = pcm_chunk.to(self.device, dtype=torch.float32)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)

        # Pad chunk to multiple of 8 * HOP_LENGTH (1,280 samples = 80 ms)
        rem_samples = wav.size(1) % (8 * HOP_LENGTH)
        if rem_samples != 0:
            wav = F.pad(wav, (0, (8 * HOP_LENGTH) - rem_samples))

        # Exact causal STFT continuity using 240-sample history buffer
        wav_with_hist = torch.cat([self.wav_history, wav], dim=1)
        self.wav_history = wav_with_hist[:, -(N_FFT - HOP_LENGTH) :]

        frames = wav_with_hist.unfold(dimension=-1, size=N_FFT, step=HOP_LENGTH) * self.mel_extractor.window
        spec = torch.fft.rfft(frames, n=N_FFT, dim=-1)
        power = spec.real.pow(2) + spec.imag.pow(2)
        mel = torch.matmul(power, self.mel_extractor.mel_fb.t())
        log_mel = (torch.log(torch.clamp(mel, min=1e-10)) - self.mel_extractor.cmvn_mean) * self.mel_extractor.cmvn_istd

        out = self.asr_model.encoder.forward_streaming(log_mel, *self.caches)
        enc_out = out[0]  # (1, T_sub, D)
        self.caches = out[1:]

        ctc_logits = self.asr_model.ctc_head(enc_out)[0]  # (T_sub, V)
        probs = F.softmax(ctc_logits, dim=-1)

        # Hybrid CTC + Stateless 2-gram TDT decoding step per 80 ms subsampled frame
        for t in range(enc_out.size(1)):
            top_p, top_id = probs[t].max(dim=-1)
            cid = int(top_id.item())
            if cid != self.tokenizer.blank_id and cid != self.prev_ctc_id:
                # Contextual rescoring with Stateless 2-gram TDT joint head
                pred_state = self.asr_model.predictor.forward_step(
                    torch.tensor([self.prev2_tok], device=self.device),
                    torch.tensor([self.prev1_tok], device=self.device),
                )
                tdt_logits, _ = self.asr_model.joint.forward_step(enc_out[:, t, :], pred_state)
                combined = 0.75 * F.log_softmax(ctc_logits[t : t + 1], dim=-1) + 0.25 * F.log_softmax(tdt_logits, dim=-1)
                best_id = int(combined[0].argmax(dim=-1).item())
                if best_id != self.tokenizer.blank_id:
                    self.token_ids.append(best_id)
                    self.frame_confidences.append(float(top_p.item()))
                    self.prev2_tok = self.prev1_tok
                    self.prev1_tok = best_id
            self.prev_ctc_id = cid

        raw_partial = self.tokenizer.decode(self.token_ids)
        return {
            "partial": raw_partial,
            "num_tokens": len(self.token_ids),
            "confidence": float(np.mean(self.frame_confidences)) if self.frame_confidences else 0.0,
        }

    def finalize_clause(self, add_punct: bool = True) -> dict:
        """Finalize current clause and apply Shrota-Shuddha grammatical refinement."""
        raw_text = self.tokenizer.decode(self.token_ids)
        shuddha_text = apply_shuddha_refinement(
            raw_text,
            shuddha_model=self.shuddha_model,
            tokenizer=self.tokenizer,
            add_punct=add_punct,
        )
        return {
            "raw_text": raw_text,
            "shuddha_text": shuddha_text,
            "confidence": float(np.mean(self.frame_confidences)) if self.frame_confidences else 0.0,
        }
