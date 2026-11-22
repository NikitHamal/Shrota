# Shrota (श्रोता) v1

**Ultra-Tiny, Real-Time Streaming & Grammatically Accurate Nepali + Sanskrit ASR for Android, Desktop, and Browser**

Shrota is a 100% from-scratch, sovereign Nepali and Vedic Sanskrit Automatic Speech Recognition (ASR) suite trained on NVIDIA RTX PRO 6000 Blackwell (98 GB) and engineered for zero-unsupported-op ONNX Runtime deployment across **Android (XNNPACK / NNAPI)**, **Desktop CPU**, and **Browser WebAssembly**.

## Model Family

| Model | Role | Architecture | Params | INT8 ONNX | Streaming Chunk | 1-Thread CPU RTF |
|---|---|---|---:|---:|---:|---:|
| **Shrota-Base** | Flagship Streaming + Offline Sorting ASR | 12L Cache-Aware FastConformer (`d=256`, 8x causal subsampling) + Hybrid Stateless TDT & CTC | ~10.8M | <= 13.5 MB | 160 ms / 320 ms | <= 0.045 |
| **Shrota-Tiny** | Ultra-Tiny Mobile & Browser ASR | 8L Distilled Cache-Aware FastConformer (`d=144`, 8x causal subsampling) + Hybrid TDT & CTC | ~3.2M | <= 4.2 MB | 160 ms / 320 ms | <= 0.022 |
| **Shrota-Shuddha** | Real-Time Grammar, Hraswa-Dirgha & Punctuation Refiner | 4L Bidirectional Edit-Constrained Transformer (`d=192`, shared 1,024-token vocab) | ~2.1M | <= 2.4 MB | Clause / 1.2s window | < 0.005 (<8 ms) |

## Repository Structure

- `shrota/` — Core PyTorch & ONNX modules (`norm.py`, `tokenizer.py`, `features.py`, `conformer.py`, `tdt_ctc.py`, `shuddha.py`, `streaming.py`, `export_onnx.py`)
- `scripts/` — Data engine, baseline benchmarking, from-scratch Blackwell training, knowledge distillation, dataset sorting CLI (`shrota_sort.py`), and live desktop dictation (`shrota_live_desktop.py`)
- `android/` — Drop-in Kotlin `ShrotaStreamingEngine.kt` for Android `onnxruntime-android` (16 kHz `AudioRecord` ring buffer + stateful cache reuse)
- `docs/` — M3X Design System GitHub Pages Live Streaming Studio (`onnxruntime-web` WASM + real Nepali/Sanskrit samples + benchmark comparator)
- `tests/` — Unit tests, Python <-> JS normalization/tokenizer parity, and PyTorch <-> ONNX streaming equivalence tests
