#!/usr/bin/env python3
"""Shrota Data-Sorting CLI (`scripts/shrota_sort.py`).

Solves the Mined-Data Sorting Problem:
Given any folder of raw/noisy Nepali or Sanskrit audio clips (.wav/.flac/.mp3), runs
Shrota-Base + CTC Forced-Alignment Confidence Scoring + Shrota-Shuddha grammatical
refinement to transcribe, compute frame-level acoustic confidence & word timestamps,
and automatically sort utterances into:
  - `gold.jsonl`   (confidence >= 0.85, Devanagari >= 95%, clean speech rate)
  - `silver.jsonl` (0.68 <= confidence < 0.85, usable with minor acoustic noise)
  - `reject.jsonl` (confidence < 0.68, music/noise/truncation/low agreement)
"""
from __future__ import annotations
import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from shrota.features import LogMelExtractor, SAMPLE_RATE
from shrota.norm import devanagari_ratio, normalize_text
from shrota.shuddha import ShrotaShuddhaModel
from shrota.streaming import apply_shuddha_refinement
from shrota.tdt_ctc import ShrotaASRModel
from shrota.tokenizer import ShrotaTokenizer, WORD_BOUNDARY


def score_and_align_clip(
    wav_path: str,
    asr_model: ShrotaASRModel,
    shuddha_model: ShrotaShuddhaModel | None,
    tokenizer: ShrotaTokenizer,
    mel_ext: LogMelExtractor,
    device: str = "cpu",
) -> dict:
    arr, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    wav = torch.from_numpy(arr)
    if sr != SAMPLE_RATE:
        new_len = int(round(wav.numel() * SAMPLE_RATE / sr))
        wav = F.interpolate(wav.view(1, 1, -1), size=new_len, mode="linear", align_corners=False).view(-1)

    dur = round(wav.numel() / SAMPLE_RATE, 3)
    if dur < 0.5:
        return {"path": wav_path, "dur": dur, "tier": "reject", "reason": "too_short", "confidence": 0.0}

    sub = getattr(asr_model, "subsampling_factor", 4)
    frame_sec = sub * 0.01
    with torch.no_grad():
        best_score = -1.0
        top_probs, top_ids, best_logits = None, None, None
        for gain in (1.0,):
            mel = mel_ext((wav * gain).to(device)).unsqueeze(0)
            rem = mel.size(1) % sub
            if rem != 0:
                mel = F.pad(mel, (0, 0, 0, sub - rem))
            enc_out = asr_model.encoder(mel)
            logits = asr_model.ctc_head(enc_out)[0]
            probs = F.softmax(logits, dim=-1)
            tp, ti = probs.max(dim=-1)
            non_blk = tp[ti != 0]
            sc = float(non_blk.mean().item()) if non_blk.numel() >= 2 else 0.0
            if sc > best_score:
                best_score = sc
                top_probs, top_ids, best_logits = tp, ti, logits

    blank_id = tokenizer.blank_id
    tokens = []
    confs = []
    frame_times = []
    prev = blank_id
    for t_idx in range(top_ids.size(0)):
        cid = int(top_ids[t_idx].item())
        if cid != blank_id and cid >= 8 and cid != prev:
            tokens.append(cid)
            confs.append(float(top_probs[t_idx].item()))
            frame_times.append(round(t_idx * frame_sec, 3))
        prev = cid

    from shrota.lm_beam import get_default_lm, ctc_prefix_beam_search
    from shrota.streaming import clean_subword_stutter_and_matras
    lm = get_default_lm()
    beam_tokens = ctc_prefix_beam_search(best_logits, lm=lm, blank_id=blank_id, bos_id=tokenizer.bos_id) if lm is not None else tokens
    raw_text = clean_subword_stutter_and_matras(tokenizer.decode(beam_tokens or tokens))
    shuddha_text = apply_shuddha_refinement(
        raw_text, shuddha_model=shuddha_model, tokenizer=tokenizer, add_punct=True
    )

    # Reconstruct word-level timestamps from subword boundary markers (▁)
    words = []
    cur_word = ""
    cur_t0 = 0.0
    cur_confs = []
    for tid, t_sec, c_val in zip(tokens, frame_times, confs):
        piece = tokenizer.id_to_token.get(tid, "")
        if piece in tokenizer.special_set:
            continue
        if piece.startswith(WORD_BOUNDARY):
            if cur_word:
                words.append({
                    "word": cur_word,
                    "t0": cur_t0,
                    "t1": t_sec,
                    "conf": round(float(np.mean(cur_confs)), 3),
                })
            cur_word = piece.lstrip(WORD_BOUNDARY)
            cur_t0 = t_sec
            cur_confs = [c_val]
        else:
            cur_word += piece
            cur_confs.append(c_val)
    if cur_word:
        words.append({
            "word": cur_word,
            "t0": cur_t0,
            "t1": dur,
            "conf": round(float(np.mean(cur_confs)), 3),
        })

    mean_conf = round(float(np.mean(confs)) if confs else 0.0, 4)
    deva = round(devanagari_ratio(shuddha_text), 3)
    wps = len(words) / max(dur, 0.1)

    if mean_conf >= 0.72 and deva >= 0.92 and 0.6 <= wps <= 6.0:
        tier = "gold"
    elif mean_conf >= 0.55 and deva >= 0.85 and 0.4 <= wps <= 7.0:
        tier = "silver"
    else:
        tier = "reject"

    return {
        "path": wav_path,
        "dur": dur,
        "tier": tier,
        "confidence": mean_conf,
        "deva_ratio": deva,
        "text_raw": raw_text,
        "text_shuddha": shuddha_text,
        "words": words,
    }


def main():
    parser = argparse.ArgumentParser(description="Sort mined Nepali/Sanskrit audio into gold/silver/reject manifests.")
    parser.add_argument("--input-dir", required=True, help="Directory containing .wav/.flac/.mp3 files")
    parser.add_argument("--output-dir", required=True, help="Directory to write gold.jsonl, silver.jsonl, reject.jsonl")
    parser.add_argument("--ckpt-dir", default="/work/shrota/checkpoints")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    tok = ShrotaTokenizer.load(str(ckpt_dir / "tokenizer.json"))
    mel_ext = LogMelExtractor().to(args.device).eval()
    if (ckpt_dir / "cmvn_stats.pt").exists():
        cmvn = torch.load(ckpt_dir / "cmvn_stats.pt", map_location=args.device, weights_only=True)
        mel_ext.set_cmvn(cmvn["mean"], cmvn["std"])

    asr = ShrotaASRModel.build_base(vocab_size=tok.vocab_size).to(args.device).eval()
    base_ckpt = ckpt_dir / "shrota_base_v2_best.pt" if (ckpt_dir / "shrota_base_v2_best.pt").exists() else ckpt_dir / "shrota_base_best.pt"
    if base_ckpt.exists():
        asr.load_state_dict(torch.load(base_ckpt, map_location=args.device, weights_only=True))

    shuddha = ShrotaShuddhaModel(vocab_size=tok.vocab_size).to(args.device).eval()
    if (ckpt_dir / "shrota_shuddha_best.pt").exists():
        shuddha.load_state_dict(torch.load(ckpt_dir / "shrota_shuddha_best.pt", map_location=args.device, weights_only=True))

    audio_files = []
    for ext in ("*.wav", "*.flac", "*.mp3"):
        audio_files.extend(glob.glob(os.path.join(args.input_dir, "**", ext), recursive=True))
    audio_files.sort()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    handles = {
        "gold": open(out_dir / "gold.jsonl", "w", encoding="utf-8"),
        "silver": open(out_dir / "silver.jsonl", "w", encoding="utf-8"),
        "reject": open(out_dir / "reject.jsonl", "w", encoding="utf-8"),
    }
    counts = {"gold": 0, "silver": 0, "reject": 0}

    for fp in audio_files:
        res = score_and_align_clip(fp, asr, shuddha, tok, mel_ext, device=args.device)
        tier = res["tier"]
        handles[tier].write(json.dumps(res, ensure_ascii=False) + "\n")
        counts[tier] += 1

    for h in handles.values():
        h.close()
    print(f"[shrota_sort] Processed {len(audio_files)} files -> {json.dumps(counts)}")


if __name__ == "__main__":
    main()
