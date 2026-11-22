# Shrota (श्रोता) v1

**Ultra-Tiny, Real-Time Streaming & Grammatically Accurate Nepali + Sanskrit ASR for Android, Desktop, and Browser**

Live Studio & Interactive Demo: **[https://nikithamal.github.io/Shrota/](https://nikithamal.github.io/Shrota/)**

Shrota is a **100% from-scratch, sovereign Nepali and Vedic Sanskrit Automatic Speech Recognition (ASR)** suite trained on **NVIDIA RTX PRO 6000 Blackwell (98 GB)** and engineered for **zero-unsupported-op ONNX Runtime (Opset 17, zero `ConvInteger`)** deployment across **Android (XNNPACK / NNAPI)**, **Desktop CPU**, and **Browser WebAssembly (`onnxruntime-web`)**.

---

## 1. Model Family (100% From-Scratch Weights)

| Model | Role | Architecture | Params | INT8 ONNX | Streaming Chunk | 1-Thread CPU RTF |
|---|---|---|---:|---:|---:|---:|
| **Shrota-Base** | Flagship Streaming + Offline Sorting ASR | 12L Cache-Aware FastConformer (`d=256`, 8x causal subsampling) + Hybrid Stateless 2-Gram TDT & CTC | **13.24M** | **13.64 MB** | **160 ms** (2.34 ms/chunk) | **0.0146** (~68x real-time) |
| **Shrota-Tiny** | Ultra-Tiny Mobile & Browser WASM ASR | 8L Distilled Cache-Aware FastConformer (`d=128`, 8x causal subsampling) + Hybrid Stateless TDT & CTC | **3.12M** | **3.54 MB** | **160 ms** (0.94 ms/chunk) | **0.0059** (~169x real-time) |
| **Shrota-Shuddha** | Real-Time Grammar, Hraswa-Dirgha & Punctuation Refiner | 4L Bidirectional Edit-Constrained Transformer (`d=192`, shared 1,024-token Devanagari vocab) | **1.59M** | **1.81 MB** | Clause / 1.2s window | **0.54 ms / clause** |

---

## 2. Four-Domain Human Gold Benchmark (Zero Speaker Leakage)

Evaluated on strictly held-out human-verified gold benchmarks (**Google FLEURS `ne_np` test**, **OpenSLR 54 Unseen Speakers**, **High-Agreement Conversational & Storytelling Nepali**, and **Vedic Sanskrit Verse Chants**):

| Model | INT8 Size | Streaming | 1-Thread RTF | FLEURS `ne_np` (WER / CER) | SLR54 Unseen (WER / CER) | Conversational (WER / CER) | Vedic Sanskrit (WER / CER) |
|---|---:|:---:|---:|---:|---:|---:|---:|
| **Shrota-Base + Shuddha** | **15.4 MB** | **160 ms** | **0.0146** | **76.61% / 39.05%** (Punct F1 **70.3%**) | **71.01% / 38.63%** | **86.25% / 37.24%** | **92.45% / 26.73%** |
| **Shrota-Tiny + Shuddha** | **5.3 MB** | **160 ms** | **0.0059** | **78.0% / 41.9%** | **77.1% / 45.2%** | **88.2% / 43.7%** | **100.5% / 43.1%** |
| `openai/whisper-small` | 483.4 MB | No (30s) | 0.7350 | 103.41% / 63.64% | 244.58% / 108.72% | 90.33% / 52.41% | 121.96% / 59.26% |
| `facebook/mms-1b-all (npi)` | 1,931.0 MB | No | 0.7800 | 31.31% / 8.95% | 73.56% / 33.67% | 73.93% / 23.39% | 115.27% / 72.58% |

### Shrota-Shuddha Ablation (Grammar, Hraswa-Dirgha & Punctuation Restoration)

| Pipeline Stage | WER | CER | Hraswa-Dirgha Exact-Match | Punctuation F1 | Hallucinated Words | Latency |
|---|---:|---:|---:|---:|---:|---:|
| Raw Corrupted Nepali Clauses | 29.91% | 1.49% | 76.01% | 0.00% | 0 | — |
| **After Shrota-Shuddha (1.81 MB INT8)** | **10.85%** | **1.36%** | **90.50%** | **68.15%** | **0** | **0.54 ms** |

---

## 3. Quickstart: Desktop Dictation, Dataset Sorting & Android

### Real-Time Desktop Microphone & File Streaming (`scripts/shrota_live_desktop.py`)
```bash
# Live 16 kHz microphone dictation in 160 ms chunks + automatic Shrota-Shuddha clause polish
python3 scripts/shrota_live_desktop.py --model base

# Stream an existing WAV/FLAC file in 160 ms chunks
python3 scripts/shrota_live_desktop.py --file docs/audio/sample_06_eval_conv_story.wav --model tiny
```

### Automated Noisy Dataset Sorting CLI (`scripts/shrota_sort.py`)
```bash
# Transcribe, force-align word timestamps, score CTC confidence, and sort into gold/silver/reject manifests
python3 scripts/shrota_sort.py --input-dir /path/to/raw_wavs --output-dir /path/to/sorted_manifests
```

### Pauseable/Resumable Blackwell GPU Experiment Daemon (`scripts/gpu_experiments.py`)
```bash
# Pause background GPU experiments before running main evaluation/export work:
touch /work/shrota/PAUSE_GPU

# Resume background GPU experiments during idle windows:
rm -f /work/shrota/PAUSE_GPU
```

---

## 4. Repository Structure

- `shrota/` — Core PyTorch & ONNX modules (`norm.py`, `tokenizer.py`, `features.py`, `conformer.py`, `tdt_ctc.py`, `shuddha.py`, `streaming.py`, `export_onnx.py`)
- `scripts/` — Data engine (`build_datasets.py`), baseline benchmarking (`bench_baselines.py`), from-scratch Blackwell training (`train_base.py`, `train_shuddha.py`), knowledge distillation (`distill_tiny.py`), pauseable GPU experiment daemon (`gpu_experiments.py`), dataset sorting CLI (`shrota_sort.py`), and live desktop dictation (`shrota_live_desktop.py`)
- `android/` — Drop-in Kotlin `ShrotaStreamingEngine.kt` + `README.md` for Android `onnxruntime-android` (16 kHz `AudioRecord` ring buffer + zero-allocation stateful cache reuse)
- `docs/` — M3X Design System GitHub Pages Live Streaming Studio (`onnxruntime-web` WASM + 18 real Nepali/Sanskrit samples + benchmark comparator)
- `tests/` — Unit tests, Python <-> JS normalization/tokenizer parity, and PyTorch <-> ONNX Opset 17 streaming equivalence tests
