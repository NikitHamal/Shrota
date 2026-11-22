#!/usr/bin/env python3
"""Real-Time Desktop Microphone & File Streaming Dictation (`scripts/shrota_live_desktop.py`).

Streams 16 kHz audio in 160 ms chunks (2,560 samples) through Shrota ONNX Runtime or PyTorch
with live terminal partial updates and automatic Shrota-Shuddha grammatical & hraswa-dirgha
commit on micro-pauses!
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from shrota.features import LogMelExtractor, SAMPLE_RATE
from shrota.shuddha import ShrotaShuddhaModel
from shrota.streaming import ShrotaStreamingPipeline
from shrota.tdt_ctc import ShrotaASRModel
from shrota.tokenizer import ShrotaTokenizer


def stream_audio_file(pipe: ShrotaStreamingPipeline, wav_path: str):
    arr, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if sr != SAMPLE_RATE:
        w = torch.from_numpy(arr).view(1, 1, -1)
        new_l = int(round(len(arr) * SAMPLE_RATE / sr))
        arr = torch.nn.functional.interpolate(w, size=new_l, mode="linear", align_corners=False).view(-1).numpy()

    pipe.reset()
    chunk_samples = 2560  # 160 ms at 16 kHz
    t0 = time.perf_counter()
    for start in range(0, len(arr), chunk_samples):
        chunk = arr[start : start + chunk_samples]
        info = pipe.accept_waveform(chunk)
        sys.stdout.write(f"\r\033[90m[Live Partial 160ms]: {info['partial']}\033[0m")
        sys.stdout.flush()

    final = pipe.finalize_clause(add_punct=True)
    dt = time.perf_counter() - t0
    dur = len(arr) / SAMPLE_RATE
    print(f"\n\033[92m[Committed Shuddha]: {final['shuddha_text']}\033[0m")
    print(f"[Stats] Audio={dur:.2f}s | Decode={dt*1000:.1f}ms | RTF={dt/max(dur,0.1):.4f} | Conf={final['confidence']:.3f}")


def stream_live_microphone(pipe: ShrotaStreamingPipeline):
    import sounddevice as sd
    print("=== Shrota Live Nepali + Sanskrit Microphone Dictation (16 kHz, 160ms chunks) ===")
    print("Speak in Nepali or Sanskrit. Pause for 0.8s to commit clause with Shrota-Shuddha. Ctrl+C to stop.")
    pipe.reset()
    silent_chunks = 0
    has_speech = False

    def callback(indata, frames, time_info, status):
        nonlocal silent_chunks, has_speech
        pcm = indata[:, 0].copy()
        rms = float(np.sqrt(np.mean(pcm * pcm)))
        if rms > 0.008:
            has_speech = True
            silent_chunks = 0
            info = pipe.accept_waveform(pcm)
            sys.stdout.write(f"\r\033[90m[Partial]: {info['partial']}\033[0m")
            sys.stdout.flush()
        elif has_speech:
            pipe.accept_waveform(pcm)
            silent_chunks += 1
            if silent_chunks >= 5:  # 5 * 160ms = 800ms pause -> commit with Shrota-Shuddha
                final = pipe.finalize_clause(add_punct=True)
                if final["shuddha_text"]:
                    sys.stdout.write(f"\r\033[K\033[92m✓ {final['shuddha_text']}\033[0m\n")
                    sys.stdout.flush()
                pipe.reset()
                has_speech = False
                silent_chunks = 0

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=2560, callback=callback):
        while True:
            time.sleep(0.1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="Optional WAV/FLAC file to stream in 160ms chunks")
    parser.add_argument("--model", choices=["base", "tiny"], default="base")
    parser.add_argument("--ckpt-dir", default="/work/shrota/checkpoints")
    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    tok = ShrotaTokenizer.load(str(ckpt_dir / "tokenizer.json"))
    mel_ext = LogMelExtractor()
    if (ckpt_dir / "cmvn_stats.pt").exists():
        cmvn = torch.load(ckpt_dir / "cmvn_stats.pt", map_location="cpu", weights_only=True)
        mel_ext.set_cmvn(cmvn["mean"], cmvn["std"])

    if args.model == "base":
        asr = ShrotaASRModel.build_base(vocab_size=tok.vocab_size)
        ckpt_p = ckpt_dir / "shrota_base_v2_best.pt" if (ckpt_dir / "shrota_base_v2_best.pt").exists() else ckpt_dir / "shrota_base_best.pt"
    else:
        asr = ShrotaASRModel.build_tiny(vocab_size=tok.vocab_size)
        ckpt_p = ckpt_dir / "shrota_tiny_v2_best.pt" if (ckpt_dir / "shrota_tiny_v2_best.pt").exists() else ckpt_dir / "shrota_tiny_best.pt"

    if ckpt_p.exists():
        asr.load_state_dict(torch.load(ckpt_p, map_location="cpu", weights_only=True))

    shuddha = ShrotaShuddhaModel(vocab_size=tok.vocab_size)
    if (ckpt_dir / "shrota_shuddha_best.pt").exists():
        shuddha.load_state_dict(torch.load(ckpt_dir / "shrota_shuddha_best.pt", map_location="cpu", weights_only=True))

    pipe = ShrotaStreamingPipeline(asr, tok, mel_ext, shuddha, chunk_ms=160, device="cpu")
    if args.file:
        stream_audio_file(pipe, args.file)
    else:
        stream_live_microphone(pipe)


if __name__ == "__main__":
    main()
