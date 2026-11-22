#!/usr/bin/env python3
"""Build docs/metrics.json from the frozen eval/*.json result files.

Only measured values are copied. Missing models (e.g. Shrota-Tiny while it is
still training) are simply omitted, and the Studio renders whatever is present.
If docs/metrics.json already carries audio `samples` produced by
scripts/eval_shrota.py, they are preserved.
"""
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "eval"
OUT = ROOT / "docs" / "metrics.json"

DOMAIN_TITLES = {
    "eval_fleurs_ne_np": ("FLEURS news read speech", "ne"),
    "eval_slr54_gold": ("OpenSLR 54 unseen speakers", "ne"),
    "eval_conv_story": ("Conversation and storytelling", "ne"),
    "eval_vedic_gold": ("Vedic verse", "sa"),
}


def load(name):
    p = EVAL / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def asr_entry(meta, label, size_key="size_int8_mb"):
    return {
        "name": label,
        "params_m": meta["params_m"],
        "size_mb": meta[size_key],
        "streaming": True,
        "chunk_ms": meta.get("streaming_chunk_ms", 160),
        "latency_ms": meta.get("streaming_latency_ms"),
        "cpu_1thread_rtf": meta["cpu_1thread_rtf"],
        "domains": {k: {m: v[m] for m in ("wer", "cer", "punct_f1", "num_utts")} for k, v in meta["domains"].items()},
    }


def examples_from(meta):
    out = []
    for dom, v in meta["domains"].items():
        if "sample_ref" not in v:
            continue
        title, lang = DOMAIN_TITLES.get(dom, (dom, "ne"))
        out.append({"id": dom, "title": title, "domain": dom, "lang": lang,
                    "reference": v["sample_ref"], "base_raw": v["sample_hyp"]})
    return out


def main():
    base, tiny = load("shrota_base.json"), load("shrota_tiny.json")
    shud, bl = load("shrota_shuddha.json"), load("baselines.json") or {}
    prev = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}

    models = {}
    if base:
        models["shrota_base"] = asr_entry(base, "Shrota-Base")
    if tiny:
        models["shrota_tiny"] = asr_entry(tiny, "Shrota-Tiny")
    for key in ("whisper_small", "mms_1b_npi"):
        if key in bl:
            models[key] = bl[key]

    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "models": {**prev.get("models", {}), **models} if prev.get("samples") else models,
        "shuddha": shud,
        "examples": examples_from(base) if base else [],
        "samples": prev.get("samples", []),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT} models={list(payload['models'])} examples={len(payload['examples'])}")


if __name__ == "__main__":
    main()
