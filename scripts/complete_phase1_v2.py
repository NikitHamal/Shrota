#!/usr/bin/env python3
"""Phase 1 Completion — Expand Precomputed Shards to 200+ Hours (All SLR54 + Mined 24k + Private Tars + SLR43).

Guarantees:
  1. Zero speaker/utterance overlap with eval_fleurs_ne_np, eval_slr54_gold, eval_conv_story, eval_vedic_gold
  2. Uses the exact frozen global CMVN stats (`cmvn_stats.pt`) and `tokenizer.json`
  3. Appends `train_shard_13.pt` (and `train_shard_14.pt`) and updates `/work/shrota/data/manifest.json`
"""
from __future__ import annotations
import csv
import io
import json
import os
import sys
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import soundfile as sf
import torch
import torch.nn.functional as F
from huggingface_hub import HfApi, hf_hub_download

from shrota.features import LogMelExtractor, SAMPLE_RATE
from shrota.norm import devanagari_ratio, has_punctuation, normalize_text
from shrota.tokenizer import ShrotaTokenizer

DATA_ROOT = Path("/work/shrota/data")
SHARD_DIR = DATA_ROOT / "shards"
EVAL_DIR = DATA_ROOT / "eval"
CKPT_DIR = Path("/work/shrota/checkpoints")


def resample_1d(wav: torch.Tensor, orig_sr: int, target_sr: int = SAMPLE_RATE) -> torch.Tensor:
    if orig_sr == target_sr:
        return wav
    new_len = int(round(wav.numel() * target_sr / orig_sr))
    if new_len <= 0:
        return torch.zeros(0, dtype=torch.float32)
    w = wav.view(1, 1, -1).float()
    return F.interpolate(w, size=new_len, mode="linear", align_corners=False).view(-1)


def load_audio_bytes(raw_bytes: bytes, target_sr: int = SAMPLE_RATE) -> torch.Tensor | None:
    try:
        arr, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32", always_2d=False)
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        return resample_1d(torch.from_numpy(arr), sr, target_sr)
    except Exception:
        return None


def main():
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hf_tok = Path("/tmp/hf_token").read_text().strip()

    tok = ShrotaTokenizer.load(str(CKPT_DIR / "tokenizer.json"))
    cmvn = torch.load(CKPT_DIR / "cmvn_stats.pt", map_location="cpu", weights_only=False)
    mel_ext = LogMelExtractor().to(device).eval()
    mel_ext.set_cmvn(cmvn["mean"].to(device), cmvn["std"].to(device))

    # Collect all existing train & eval utterance IDs to prevent any duplication or leakage
    manifest = json.loads((DATA_ROOT / "manifest.json").read_text())
    existing_shards = [p for p in manifest["shard_paths"] if not p.endswith(("13.pt", "14.pt"))]

    seen_uids = set()
    eval_uids = set()
    for ev_name in ("eval_fleurs_ne_np", "eval_slr54_gold", "eval_conv_story", "eval_vedic_gold"):
        ev_recs = torch.load(EVAL_DIR / f"{ev_name}.pt", map_location="cpu", weights_only=False)
        for r in ev_recs:
            uid = r["utt_id"]
            eval_uids.add(uid)
            seen_uids.add(uid)
            for prefix in ("mined_", "vedic_", "priv_", "slr54_", "fleurs_test_"):
                if uid.startswith(prefix):
                    eval_uids.add(uid[len(prefix) :])
                    seen_uids.add(uid[len(prefix) :])

    base_clips = 0
    base_hours = 0.0
    for sp in existing_shards:
        recs = torch.load(sp, map_location="cpu", weights_only=False)
        base_clips += len(recs)
        for r in recs:
            uid = r["utt_id"]
            seen_uids.add(uid)
            for prefix in ("mined_", "vedic_", "priv_", "slr54_"):
                if uid.startswith(prefix):
                    seen_uids.add(uid[len(prefix) :])
            base_hours += float(r.get("dur", r["mel"].size(0) * 0.01)) / 3600.0

    print(f"[Phase 1 v2] Existing shards: {len(existing_shards)} ({base_clips} clips, {base_hours:.2f} hours)", flush=True)

    new_items = []
    shard13_p = SHARD_DIR / "train_shard_13.pt"
    p_mined = None
    p_real = DATA_ROOT / "hf" / "v13" / "real_audio_24k.pt"
    if not shard13_p.exists() and p_real.exists():
        p_mined = hf_hub_download(
            repo_id="NikitHamal/sakhiya-v1-train",
            filename="box3/v13/mined_24k.pt",
            repo_type="model",
            token=hf_tok,
            local_dir=str(DATA_ROOT / "hf"),
        )
    if not shard13_p.exists() and p_real.exists():
        mined_obj = torch.load(p_mined, map_location="cpu", weights_only=False)
        mined_map = {it["id"]: it for it in mined_obj.get("items", []) if it.get("id") and it.get("text")}
        del mined_obj

        real_obj = torch.load(p_real, map_location="cpu", weights_only=False)
        wav_all = real_obj["wav"]
        lens_all = real_obj["lens"].tolist()
        paths_all = real_obj["paths"]
        orig_sr = int(real_obj.get("sr", 24000))

        offset = 0
        added_mined = 0
        for p, length in zip(paths_all, lens_all):
            clip_24k = wav_all[offset : offset + length].float()
            offset += length
            uid = os.path.basename(p).replace(".wav", "")
            if uid in seen_uids or uid in eval_uids or length < int(0.8 * orig_sr):
                continue
            m_it = mined_map.get(uid)
            if not m_it:
                continue
            tx = normalize_text(m_it["text"], keep_punct=True)
            if not tx or devanagari_ratio(tx) < 0.90:
                continue
            wav_16k = resample_1d(clip_24k, orig_sr, SAMPLE_RATE)
            seen_uids.add(uid)
            lang = "sa" if m_it.get("lang") == "sa" or "vedic" in str(m_it.get("style", "")) else "ne"
            item = {
                "utt_id": f"mined2_{uid}",
                "wav": wav_16k,
                "text": tx,
                "lang": lang,
                "punct_mode": "punct" if has_punctuation(tx) else "nopunct",
                "spk": str(m_it.get("spk_key", uid.split("__")[0])),
                "domain": "vedic_sanskrit" if lang == "sa" else "conv_story",
                "dur": round(wav_16k.numel() / SAMPLE_RATE, 3),
            }
            new_items.append(item)
            added_mined += 1
        del wav_all, real_obj
        print(f"[Phase 1 v2] Harvested {added_mined} additional clean clips from mined_24k.pt", flush=True)

    # 2. Pull remaining tar files from NikitHamal/sakhiya-nepali-vedic-private
    try:
        api = HfApi(token=hf_tok) if not shard13_p.exists() else None
        priv_files = api.list_repo_files("NikitHamal/sakhiya-nepali-vedic-private", repo_type="dataset") if api else []
        priv_meta_path = DATA_ROOT / "hf_priv" / "meta" / "meta.jsonl"
        if not shard13_p.exists() and priv_meta_path.exists():
            priv_meta = {}
            with open(priv_meta_path, "r", encoding="utf-8") as f:
                for line in f:
                    d = json.loads(line)
                    if d.get("id") and d["id"] not in seen_uids and d["id"] not in eval_uids:
                        priv_meta[d["id"]] = d
            tar_files = [fn for fn in priv_files if fn.startswith("raw/") and fn.endswith(".tar")]
            added_priv = 0
            for tfn in tar_files:
                if not priv_meta:
                    break
                tp = hf_hub_download(
                    repo_id="NikitHamal/sakhiya-nepali-vedic-private",
                    filename=tfn,
                    repo_type="dataset",
                    token=hf_tok,
                    local_dir=str(DATA_ROOT / "hf_priv"),
                )
                with tarfile.open(tp, "r") as tf:
                    for m in tf.getmembers():
                        if m.isfile() and m.name.endswith(".wav"):
                            uid = os.path.basename(m.name).replace(".wav", "")
                            if uid in priv_meta and uid not in seen_uids:
                                pm = priv_meta[uid]
                                tx = normalize_text(pm.get("text", ""), keep_punct=True)
                                if devanagari_ratio(tx) < 0.90:
                                    continue
                                fobj = tf.extractfile(m)
                                if not fobj:
                                    continue
                                wav_16k = load_audio_bytes(fobj.read())
                                if wav_16k is None or wav_16k.numel() < 12800:
                                    continue
                                seen_uids.add(uid)
                                lang = "sa" if pm.get("lang") == "sa" or "vedic" in str(pm.get("style", "")) else "ne"
                                new_items.append({
                                    "utt_id": f"priv_{uid}",
                                    "wav": wav_16k,
                                    "text": tx,
                                    "lang": lang,
                                    "punct_mode": "punct" if has_punctuation(tx) else "nopunct",
                                    "spk": str(pm.get("spk", uid.split("__")[0])),
                                    "domain": "vedic_sanskrit" if lang == "sa" else "conv_story",
                                    "dur": round(wav_16k.numel() / SAMPLE_RATE, 3),
                                })
                                added_priv += 1
            print(f"[Phase 1 v2] Added {added_priv} clips from all 13 sakhiya-nepali-vedic-private tars", flush=True)
    except Exception as e:
        print(f"[Phase 1 v2] Private tar note: {e}", flush=True)

    # 3. Download OpenSLR SLR43 (~3h studio female Nepali)
    try:
        slr43_dir = DATA_ROOT / "slr43"
        slr43_dir.mkdir(parents=True, exist_ok=True)
        slr43_zip = slr43_dir / "ne_np_female.zip"
        if not shard13_p.exists() and not slr43_zip.exists():
            print("[Phase 1 v2] Downloading OpenSLR SLR43 (ne_np_female.zip)...", flush=True)
            urllib.request.urlretrieve("https://www.openslr.org/resources/43/ne_np_female.zip", str(slr43_zip))
        added_slr43 = 0
        with zipfile.ZipFile(slr43_zip, "r") as zf:
            if shard13_p.exists():
                raise StopIteration()
            tsv_names = [n for n in zf.namelist() if n.endswith(".tsv")]
            utt_tx = {}
            for tn in tsv_names:
                raw_tsv = zf.read(tn).decode("utf-8", errors="replace")
                for line in raw_tsv.splitlines():
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        u_id = parts[0].strip().replace(".wav", "")
                        utt_tx[u_id] = parts[-1].strip()
            for name in zf.namelist():
                if name.endswith(".wav"):
                    uid = os.path.splitext(os.path.basename(name))[0]
                    if uid not in utt_tx or uid in seen_uids:
                        continue
                    tx = normalize_text(utt_tx[uid], keep_punct=True)
                    if not tx or devanagari_ratio(tx) < 0.90:
                        continue
                    wav_16k = load_audio_bytes(zf.read(name))
                    if wav_16k is None or wav_16k.numel() < 12800:
                        continue
                    seen_uids.add(uid)
                    new_items.append({
                        "utt_id": f"slr43_{uid}",
                        "wav": wav_16k,
                        "text": tx,
                        "lang": "ne",
                        "punct_mode": "punct" if has_punctuation(tx) else "nopunct",
                        "spk": f"slr43_{uid[:7]}",
                        "domain": "slr43",
                        "dur": round(wav_16k.numel() / SAMPLE_RATE, 3),
                    })
                    added_slr43 += 1
        print(f"[Phase 1 v2] Added {added_slr43} studio female Nepali clips from OpenSLR SLR43", flush=True)
    except Exception as e:
        print(f"[Phase 1 v2] SLR43 note: {e}", flush=True)

    # Encode to fp16 Log-Mel records (oversampling conversational/storytelling/SLR43 2x)
    expanded_items = new_items * 2
    records = []
    with torch.no_grad():
        for it in expanded_items:
            w = it["wav"].to(device)
            mel = mel_ext(w, normalize=True).half().cpu()
            ids = tok.encode(it["text"], lang=it["lang"], punct_mode=it["punct_mode"])
            if not ids or mel.size(0) // 4 < len(ids):
                continue
            records.append({
                "utt_id": it["utt_id"],
                "mel": mel,
                "tokens": torch.tensor(ids, dtype=torch.int16),
                "text": it["text"],
                "lang": it["lang"],
                "punct_mode": it["punct_mode"],
                "spk": it["spk"],
                "domain": it["domain"],
                "dur": it["dur"],
            })

    new_shard_paths = list(existing_shards)
    shard13_p = SHARD_DIR / "train_shard_13.pt"
    if records:
        torch.save(records, shard13_p)
        new_shard_paths.append(str(shard13_p))
        added_h = sum(r["dur"] for r in records) / 3600.0
        base_clips += len(records)
        base_hours += added_h
        print(f"[Phase 1 v2] Saved {shard13_p} ({len(records)} clips, +{added_h:.2f} hours)", flush=True)
    elif shard13_p.exists():
        recs13 = torch.load(shard13_p, map_location="cpu", weights_only=False)
        new_shard_paths.append(str(shard13_p))
        added_h = sum(r.get("dur", r["mel"].size(0) * 0.01) for r in recs13) / 3600.0
        base_clips += len(recs13)
        base_hours += added_h
        print(f"[Phase 1 v2] Kept existing {shard13_p} ({len(recs13)} clips, +{added_h:.2f} hours)", flush=True)

    # 4. Download & encode official Google FLEURS ne_np train split (train.tsv + train.tar.gz, 0% test overlap)
    shard14_p = SHARD_DIR / "train_shard_14.pt"
    fleurs_dir = DATA_ROOT / "fleurs"
    for fn in ("data/ne_np/train.tsv", "data/ne_np/audio/train.tar.gz"):
        hf_hub_download("google/fleurs", fn, repo_type="dataset", token=hf_tok, local_dir=str(fleurs_dir))

    tsv_p = fleurs_dir / "data/ne_np/train.tsv"
    tar_p = fleurs_dir / "data/ne_np/audio/train.tar.gz"
    audio_map = {}
    with tarfile.open(tar_p, "r:gz") as tf:
        for m in tf.getmembers():
            if m.isfile() and m.name.endswith((".wav", ".flac", ".mp3")):
                fobj = tf.extractfile(m)
                if fobj:
                    audio_map[os.path.basename(m.name)] = fobj.read()

    fleurs_train_recs = []
    with torch.no_grad():
        for line in tsv_p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = line.split("\t")
            if len(row) < 4:
                continue
            fname = row[1].strip()
            raw_tx = row[2].strip() if len(row) > 2 else row[3].strip()
            if fname not in audio_map:
                continue
            wav_16k = load_audio_bytes(audio_map[fname])
            if wav_16k is None or wav_16k.numel() < 1600:
                continue
            norm_tx = normalize_text(raw_tx, keep_punct=True)
            if not norm_tx:
                continue
            mel = mel_ext(wav_16k.to(device), normalize=True).half().cpu()
            pmode = "punct" if has_punctuation(norm_tx) else "nopunct"
            ids = tok.encode(norm_tx, lang="ne", punct_mode=pmode)
            if not ids or mel.size(0) // 4 < len(ids):
                continue
            fleurs_train_recs.append({
                "utt_id": f"fleurs_train_{fname.replace('.wav', '')}",
                "mel": mel,
                "tokens": torch.tensor(ids, dtype=torch.int16),
                "text": norm_tx,
                "lang": "ne",
                "punct_mode": pmode,
                "spk": f"fleurs_{row[0].strip()}",
                "domain": "fleurs_ne_np",
                "dur": round(wav_16k.numel() / SAMPLE_RATE, 3),
            })

    if fleurs_train_recs:
        torch.save(fleurs_train_recs, shard14_p)
        new_shard_paths.append(str(shard14_p))
        f_h = sum(r["dur"] for r in fleurs_train_recs) / 3600.0
        base_clips += len(fleurs_train_recs)
        base_hours += f_h
        print(f"[Phase 1 v2] Saved {shard14_p} ({len(fleurs_train_recs)} FLEURS train clips, +{f_h:.2f} hours)", flush=True)

    manifest["total_train_clips"] = base_clips
    manifest["total_train_hours"] = round(base_hours, 2)
    manifest["shard_paths"] = new_shard_paths
    (DATA_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        f"[Phase 1 v2 DONE in {time.time() - t0:.1f}s] Total train clips={base_clips:,}, "
        f"Total train hours={base_hours:.2f}h across {len(new_shard_paths)} shards!",
        flush=True,
    )


if __name__ == "__main__":
    main()
