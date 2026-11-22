# Shrota (श्रोता) v2

**Ultra-Tiny, Real-Time Streaming & Grammatically Accurate Nepali + Sanskrit ASR for Android, Desktop, and Browser**

Live Studio & Interactive Demo: **[https://nikithamal.github.io/Shrota/](https://nikithamal.github.io/Shrota/)**

Shrota v2 is a **100% from-scratch, sovereign Nepali and Vedic Sanskrit Automatic Speech Recognition (ASR)** suite trained on **NVIDIA RTX PRO 6000 Blackwell (98 GB)** and engineered for **zero-unsupported-op ONNX Runtime (Opset 17, zero `ConvInteger`)** deployment across **Android (XNNPACK / NNAPI)**, **Desktop CPU**, and **Browser WebAssembly (`onnxruntime-web`)**.

---

## 1. What Changed in Shrota v2 (Forensic Fix & Breakthroughs)

See [`eval/v1_autopsy.md`](eval/v1_autopsy.md) for the complete 5-check forensic autopsy of v1:
1. **Complete 16-Shard OpenSLR SLR54 + Multi-Domain Corpus (`159,067` clips / `122.3` hours)** with **Strict Global Fixed CMVN** (`(mel - mean[80]) / std[80]`) baked across training, evaluation, and ONNX streaming.
2. **4× Causal Depthwise Subsampling (`40 ms/frame` = `25 Hz`, replacing v1's 8× `80 ms/frame`)**: Eliminates CTC/TDT frame starvation on rapid Devanagari subwords.
3. **Self-Conditioned Intermediate CTC (SC-InterCTC at Layers 4 & 8)**: Applies shared CTC projection at layers 4 and 8 (`0.15` aux loss each) and projects `Linear(softmax(ctc_logits))` back into the Conformer residual stream (`< 0.3 MB` extra weights after ONNX initializer deduplication).
4. **Verified Hybrid CTC + Differentiable Pruned Stateless 2-Gram TDT/RNN-T** with unit-tested `rnnt_loss_exact` parity and strict `max_abs_diff < 1e-4` chunk-vs-offline streaming equivalence.
5. **Compact Pruned 4-Gram Nepali Subword + Word-Bigram LM (`2.07 MB`) & Prefix Beam Search (`beam=6`, `< 1 ms` CPU/WASM/Kotlin)**: Shared across Python (`shrota/lm_beam.py`), Browser JS (`docs/asr/text.js`), and Android Kotlin (`android/ShrotaStreamingEngine.kt`).

---

## 2. Model Family (100% From-Scratch v2 Weights)

| Model | Role | Architecture | Params | INT8 ONNX | Streaming Chunk | 1-Thread CPU RTF |
|---|---|---|---:|---:|---:|---:|
| **Shrota-Base v2** | Flagship Streaming + Offline Sorting ASR | 12L Cache-Aware FastConformer (`d=256`, 4× causal subsampling + SC-InterCTC) + Hybrid Stateless 2-Gram TDT & CTC | **13.42M** | **13.82 MB** | **160 ms** (2.64 ms/chunk) | **0.0165** (~61× real-time) |
| **Shrota-Tiny v2** | Ultra-Tiny Mobile & Browser WASM ASR | 8L Distilled Cache-Aware FastConformer (`d=128`, 4× causal subsampling + SC-InterCTC) + Hybrid Stateless TDT & CTC | **3.08M** | **3.51 MB** | **160 ms** (1.09 ms/chunk) | **0.0068** (~147× real-time) |
| **Shrota-Shuddha v2** | Real-Time Grammar, Hraswa-Dirgha & Punctuation Refiner | 4L Bidirectional Edit-Constrained Transformer (`d=192`, shared 1,024-token Devanagari vocab) | **1.59M** | **1.81 MB** | Clause / 1.2s window | **0.54 ms / clause** |
| **Nepali 4-Gram LM** | Subword + Word-Bigram Shallow Fusion Trie | Pruned 4-gram subword log-probs + 16k word lexicon & contextual bigrams | — | **2.07 MB** | Per-frame (`beam=6`) | **< 0.8 ms / clause** |

---

## 3. Four-Domain Human Gold Benchmark (Zero Speaker Leakage)

Evaluated on strictly held-out human-verified gold benchmarks (**Google FLEURS `ne_np` test**, **OpenSLR 54 Unseen Speakers**, **High-Agreement Conversational & Storytelling Nepali**, and **Vedic Sanskrit Verse Chants**):

| Model | INT8 Size | Streaming | 1-Thread RTF | SLR54 Unseen (WER / CER) | FLEURS `ne_np` (WER / CER) | Conversational (WER / CER) | Vedic Sanskrit (WER / CER) |
|---|---:|:---:|---:|---:|---:|---:|---:|
| **Shrota-Base v2 + Shuddha** | **15.6 MB** | **160 ms** | **0.0165** | **31.86% / 10.03%** | **51.85% / 20.55%** (Punct F1 **72.5%**) | **70.98% / 22.21%** | **65.92% / 9.21%** (Punct F1 **80.0%**) |
| **Shrota-Base v2 (Raw + 4-gram LM)** | **13.8 MB** | **160 ms** | **0.0165** | **31.99% / 10.05%** | **51.99% / 20.55%** | **70.87% / 22.05%** | **65.87% / 8.96%** |
| **Shrota-Tiny v2 + Shuddha** | **5.3 MB** | **160 ms** | **0.0068** | **42.90% / 14.66%** | **53.41% / 21.57%** (Punct F1 **71.5%**) | **74.16% / 24.81%** | **72.70% / 9.90%** |
| **Shrota-Tiny v2 (Raw + 4-gram LM)** | **3.5 MB** | **160 ms** | **0.0068** | **43.04% / 14.69%** | **53.43% / 21.54%** | **74.09% / 24.66%** | **72.66% / 9.67%** |
| `openai/whisper-small` | 483.4 MB | No (30s) | 0.7350 | 244.58% / 108.72% | 103.41% / 63.64% | 90.33% / 52.41% | 121.96% / 59.26% |
| `facebook/mms-1b-all (npi)` | 1,931.0 MB | No | 0.7800 | 73.56% / 33.67% | 31.31% / 8.95% | 73.93% / 23.39% | 115.27% / 72.58% |

*Note: On OpenSLR 54 Unseen Speakers, Conversational Nepali, and Vedic Sanskrit Verse, **Shrota-Base v2 (13.8 MB INT8)** outperforms both **Whisper-Small (483 MB)** and **Meta MMS-1b (1,931 MB)** while streaming in 160 ms chunks at 61× real-time on a single CPU thread.*

### Shrota-Shuddha v2 Ablation (Grammar, Hraswa-Dirgha & Punctuation Restoration)

| Pipeline Stage | WER | CER | Hraswa-Dirgha Exact-Match | Punctuation F1 | Hallucinated Words | Latency |
|---|---:|---:|---:|---:|---:|---:|
| Before Shuddha (Real Acoustic/Orthographic Errors) | 29.78% | 1.44% | 76.55% | 0.00% | 0 | — |
| **After Shrota-Shuddha v2 (1.81 MB INT8)** | **11.07%** | **1.83%** | **90.62%** | **69.23%** | **0** | **0.54 ms** |

---

## 4. Quickstart: Desktop Dictation, Dataset Sorting & Android

### Real-Time Desktop Microphone & File Streaming (`scripts/shrota_live_desktop.py`)
```bash
# Live 16 kHz microphone dictation in 160 ms chunks + automatic Shrota-Shuddha clause polish
python3 scripts/shrota_live_desktop.py --model base

# Stream an existing WAV/FLAC file in 160 ms chunks
python3 scripts/shrota_live_desktop.py --file docs/audio/sample_04_eval_slr54_gold.wav --model tiny
```

### Automated Noisy Dataset Sorting CLI (`scripts/shrota_sort.py`)
```bash
# Transcribe, force-align word timestamps, score CTC confidence, and sort into gold/silver/reject manifests
python3 scripts/shrota_sort.py --input-dir /path/to/raw_wavs --output-dir /path/to/sorted_manifests
```

---

## 5. Repository Structure

- `shrota/` — Core PyTorch & ONNX modules (`norm.py`, `tokenizer.py`, `features.py`, `conformer.py`, `tdt_ctc.py`, `lm_beam.py`, `shuddha.py`, `streaming.py`, `export_onnx.py`)
- `scripts/` — Forensic v1 audit (`forensic_audit_v1.py`), complete 16-shard Phase 1 builder (`complete_phase1_v2.py`), from-scratch Blackwell v2 trainer (`train_base_v2.py`), 4-gram Nepali LM builder (`build_nepali_lm.py`), knowledge distillation (`distill_tiny.py`), Shuddha v2 trainer (`train_shuddha.py`), end-to-end evaluator (`eval_shrota.py`), dataset sorting CLI (`shrota_sort.py`), and live desktop dictation (`shrota_live_desktop.py`)
- `android/` — Drop-in Kotlin `ShrotaStreamingEngine.kt` + `README.md` for Android `onnxruntime-android` (4× subsampling + 4-gram Prefix Beam Search + stateful cache reuse)
- `docs/` — M3X Design System GitHub Pages Live Streaming Studio (`onnxruntime-web` WASM + 18 real Nepali/Sanskrit samples + benchmark comparator)
- `eval/` — Forensic v1 autopsy (`v1_autopsy.md`) and frozen 4-domain evaluation JSONs
- `tests/` — Unit tests, Python <-> JS normalization/tokenizer parity, and PyTorch <-> ONNX Opset 17 streaming equivalence tests
