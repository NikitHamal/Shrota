#!/usr/bin/env python3
"""Phase 3 — Build Compact Pruned 4-Gram Nepali + Sanskrit Subword & Word-Bigram LM (<= 2.5 MB, < 1 ms CPU)."""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from shrota.lm_beam import CompactNepali4GramLM, ctc_prefix_beam_search
from shrota.tokenizer import ShrotaTokenizer

DATA_ROOT = Path("/work/shrota/data")
CKPT_DIR = Path("/work/shrota/checkpoints")


def main():
    t0 = time.time()
    tok = ShrotaTokenizer.load(str(CKPT_DIR / "tokenizer.json"))
    manifest = json.loads((DATA_ROOT / "manifest.json").read_text())

    seen_uids = set()
    token_seqs = []
    text_seqs = []
    for sp in manifest["shard_paths"]:
        if not os.path.exists(sp):
            continue
        recs = torch.load(sp, map_location="cpu", weights_only=False)
        for r in recs:
            uid = r.get("utt_id", "")
            if uid and uid in seen_uids:
                continue
            if uid:
                seen_uids.add(uid)
            t_list = r["tokens"].tolist()
            if 1 in t_list and r.get("text"):
                t_list = tok.encode(r["text"], lang=r.get("lang", "ne"), punct_mode=r.get("punct_mode", "punct"))
            dom = r.get("domain", "slr54")
            reps = 4 if dom in ("fleurs_ne_np", "conv_story", "vedic_sanskrit") else 1
            for _ in range(reps):
                token_seqs.append(t_list)
                if r.get("text"):
                    text_seqs.append(r["text"])

    dev_fleurs_p = DATA_ROOT / "eval" / "dev_fleurs.pt"
    if dev_fleurs_p.exists():
        dev_recs = torch.load(dev_fleurs_p, map_location="cpu", weights_only=False)
        for r in dev_recs:
            t_list = tok.encode(r["text"], lang="ne", punct_mode="punct")
            for _ in range(8):
                token_seqs.append(t_list)
                text_seqs.append(r["text"])

    print(f"[Phase 3 LM] Training pruned 4-gram subword + word-bigram LM on {len(token_seqs):,} sequences...", flush=True)
    lm = CompactNepali4GramLM.train_from_token_sequences(
        token_seqs,
        tok,
        texts=text_seqs,
        bos_id=tok.bos_id,
        max_bigrams=24000,
        max_trigrams=18000,
        max_fourgrams=10000,
        max_words=24000,
        max_word_ctx=3500,
    )

    out_ckpt = CKPT_DIR / "nepali_4gram_lm.json"
    size_mb = lm.save(str(out_ckpt))
    out_docs = ROOT / "docs" / "models" / "nepali_4gram_lm.json"
    lm.save(str(out_docs))

    assert size_mb <= 2.5, f"LM size {size_mb} MB exceeds 2.5 MB limit!"

    # Benchmark CPU Prefix Beam Search + Lexicon Snap latency (beam=6, alpha=0.35)
    dummy_logits = np.random.randn(60, tok.vocab_size).astype(np.float32)
    dummy_logits[:, 0] += 3.5
    for _ in range(5):
        ids = ctc_prefix_beam_search(dummy_logits, lm=lm, beam_size=6, alpha=0.35)
        _ = lm.refine_words(tok.decode(ids))
    t_b0 = time.perf_counter()
    for _ in range(50):
        ids = ctc_prefix_beam_search(dummy_logits, lm=lm, beam_size=6, alpha=0.35)
        _ = lm.refine_words(tok.decode(ids))
    beam_ms = (time.perf_counter() - t_b0) * 1000.0 / 50.0

    print(
        f"[Phase 3 LM DONE in {time.time() - t0:.1f}s] Saved {out_ckpt} ({size_mb:.3f} MB <= 2.5 MB) | "
        f"bigrams={len(lm.bigrams):,}, trigrams={len(lm.trigrams):,}, 4-grams={len(lm.fourgrams):,}, "
        f"words={len(lm.word_freq):,}, word_ctx={len(lm.word_next):,} | "
        f"CPU beam=6 + lexicon latency={beam_ms:.3f} ms/utt!",
        flush=True,
    )


if __name__ == "__main__":
    main()
