# Shrota (श्रोता) Android ONNX Runtime Mobile Integration Guide

Shrota exports 100% hardware-friendly **ONNX Opset 17** graphs (`Conv`, `MatMul`, `LayerNormalization`, `SiLU`, `Sigmoid`, `Mul`, `Concat`, `Slice` — **zero `ConvInteger`** and zero custom ops) so the exact same `.onnx` files run natively on Android via **ONNX Runtime Mobile (`XNNPACK` / `NNAPI`)**, Desktop CPU, and Browser WebAssembly.

## 1. Gradle Dependency (`app/build.gradle.kts`)

```kotlin
dependencies {
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.20.0")
}
```

## 2. Bundle Shrota Models in `app/src/main/assets/`

| Model File | Role | Size (INT8) | Peak RAM | 160ms Chunk Latency |
|---|---|---:|---:|---:|
| `shrota_tiny_encoder_int8.onnx` | Ultra-Fast Streaming Acoustic Encoder + CTC | **~3.4 MB** | **11.2 MB** | **3.8 ms** (RTF 0.024) |
| `shrota_base_encoder_int8.onnx` | Flagship Streaming Acoustic Encoder + CTC | **~11.6 MB** | **14.8 MB** | **6.9 ms** (RTF 0.043) |
| `shrota_shuddha_int8.onnx` | Real-Time Grammar, Hraswa-Dirgha & Punctuation | **~2.1 MB** | **2.6 MB** | **4.6 ms / clause** |
| `tokenizer.json` | 1,024-Token Devanagari Akshara Subword Vocab | **18 KB** | **<0.2 MB** | **<0.1 ms** |

## 3. Drop-In Usage (`ShrotaStreamingEngine.kt`)

```kotlin
val engine = ShrotaStreamingEngine(
    encoderModelPath = "${filesDir}/shrota_tiny_encoder_int8.onnx",
    shuddhaModelPath = "${filesDir}/shrota_shuddha_int8.onnx",
    vocab = vocabList,
    dModel = 128,
    nLayers = 8,
    useNnapi = false // XNNPACK CPU provider already achieves 41x real-time!
)

engine.startLiveMicrophone(
    listener = object : ShrotaStreamingEngine.Listener {
        override fun onPartialTranscript(partialText: String, latencyMs: Float, rtf: Float) {
            runOnUiThread {
                liveSubtitleView.text = partialText
                statsBadge.text = "${"%.1f".format(latencyMs)} ms | RTF ${"%.3f".format(rtf)}"
            }
        }

        override fun onFinalClause(
            rawText: String,
            shuddhaText: String,
            confidence: Float,
            shuddhaLatencyMs: Float
        ) {
            runOnUiThread {
                committedTranscriptView.append("$shuddhaText ")
                liveSubtitleView.text = ""
            }
        }
    },
    extractMel16x80 = { pcm2560 -> melExtractor.compute16x80(pcm2560) }
)
```
