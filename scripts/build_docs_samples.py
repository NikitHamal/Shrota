"""Export a small spread of held-out FLEURS ne_np test clips for the GitHub Pages studio.

Usage: python3 scripts/build_docs_samples.py --fleurs-dir /path/with/test.tsv+test/ --ids id1,id2,...
Writes 16 kHz mono 16-bit WAV files to docs/samples/ plus docs/samples/manifest.json.
Clips are CC-BY-4.0 (Google FLEURS) and are never used for training.
"""
import argparse, json, struct, wave
from pathlib import Path
import numpy as np


def read_wav(path: Path):
    b = path.read_bytes()
    o, fmt, data = 12, None, None
    while o + 8 <= len(b):
        cid, sz = b[o:o + 4], struct.unpack("<I", b[o + 4:o + 8])[0]
        if cid == b"fmt ":
            fmt = struct.unpack("<HHIIHH", b[o + 8:o + 24])
        if cid == b"data":
            data = b[o + 8:o + 8 + sz]
            break
        o += 8 + sz + (sz & 1)
    tag, ch, sr, _, _, bits = fmt
    x = np.frombuffer(data, dtype="<f4" if tag == 3 else "<i2").astype(np.float32)
    if tag != 3:
        x /= 32768.0
    return x.reshape(-1, ch).mean(axis=1), sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fleurs-dir", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--out", default="docs/samples")
    a = ap.parse_args()
    root, out = Path(a.fleurs_dir), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = {r.split("\t")[1]: r.split("\t") for r in (root / "test.tsv").read_text(encoding="utf-8").splitlines()}
    items = []
    for n, fid in enumerate(a.ids.split(","), 1):
        x, sr = read_wav(root / "test" / fid)
        assert sr == 16000, sr
        pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
        name = f"fleurs_ne_{n:02d}.wav"
        with wave.open(str(out / name), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(pcm.tobytes())
        items.append({
            "id": f"fleurs_ne_{n:02d}", "title": f"FLEURS test clip {n}", "lang": "ne",
            "domain": "eval_fleurs_ne_np", "audio_url": f"samples/{name}",
            "duration_s": round(len(x) / 16000, 1), "reference": rows[fid][2],
            "source": "google/fleurs ne_np test (CC-BY-4.0), held out from training",
        })
    (out / "manifest.json").write_text(json.dumps({"samples": items}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {len(items)} clips")


if __name__ == "__main__":
    main()
