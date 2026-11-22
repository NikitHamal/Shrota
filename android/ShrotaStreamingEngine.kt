package io.github.nikithamal.shrota

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import java.nio.FloatBuffer
import java.nio.LongBuffer
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.ln1p
import kotlin.math.max
import kotlin.math.PI
import kotlin.math.sqrt

/**
 * ShrotaStreamingEngine v2 — Production Drop-In Real-Time Nepali + Sanskrit ASR for Android.
 *
 * Architecture (Shrota v2):
 *   - 4x Causal Depthwise Subsampling (40 ms/frame = 25 Hz, zero CTC frame starvation)
 *   - Self-Conditioned Intermediate CTC (SC-InterCTC at layers 4 & 8) + Cache-Aware FastConformer
 *   - Hybrid CTC + Stateless 2-gram TDT Decoder + Compact Pruned 4-Gram Nepali LM Prefix Beam Search
 *   - Shrota-Shuddha v2 (1.81 MB INT8) for real-time Hraswa-Dirgha, Grammar & Punctuation polish
 *
 * Uses ONNX Runtime Mobile (`ai.onnxruntime:onnxruntime-android:1.20.0`) with XNNPACK / NNAPI
 * hardware acceleration and zero-ConvInteger Opset 17 stateful streaming ONNX graphs:
 *   - `shrota_tiny_encoder_int8.onnx` + `shrota_tiny_decoder_int8.onnx` (~4.14 MB total INT8)
 *   - `shrota_base_encoder_int8.onnx` + `shrota_base_decoder_int8.onnx` (~15.8 MB total INT8)
 *   - `shrota_shuddha_int8.onnx` (~1.81 MB INT8)
 */
class ShrotaStreamingEngine(
    encoderModelPath: String,
    shuddhaModelPath: String?,
    private val vocab: List<String>,
    private val dModel: Int = 128,
    private val nLayers: Int = 8,
    private val nHeads: Int = 4,
    private val leftCtx: Int = 64,
    private val convKernel: Int = 9,
    private val subsamplingFactor: Int = 4,
    private val lmUnigrams: FloatArray? = null,
    private val lmBigrams: Map<String, Float> = emptyMap(),
    private val lmTrigrams: Map<String, Float> = emptyMap(),
    private val lmFourgrams: Map<String, Float> = emptyMap(),
    useNnapi: Boolean = false
) : AutoCloseable {

    companion object {
        const val SAMPLE_RATE = 16000
        const val N_FFT = 400
        const val HOP_LENGTH = 160
        const val N_MELS = 80
        const val CHUNK_FRAMES = 16            // 160 ms acoustic chunk (4 encoder frames at 4x subsampling)
        const val CHUNK_SAMPLES = 2560         // 16 * 160 samples = 160 ms at 16 kHz
        const val STFT_HISTORY = 240           // N_FFT - HOP_LENGTH causal overlap
        const val WORD_BOUNDARY = "▁"
        const val BLANK_ID = 0
        const val BOS_ID = 2
        private const val LOG_ZERO = -1e9f
    }

    interface Listener {
        fun onPartialTranscript(partialText: String, latencyMs: Float, rtf: Float)
        fun onFinalClause(rawText: String, shuddhaText: String, confidence: Float, shuddhaLatencyMs: Float)
    }

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val encoderSession: OrtSession
    private val shuddhaSession: OrtSession?
    private val headDim: Int = dModel / nHeads

    // Reusable causal STFT history and encoder streaming caches (zero GC churn per chunk)
    private var wavHistory = FloatArray(STFT_HISTORY)
    private var cacheS0 = FloatArray(1 * N_MELS * 1)
    private var cacheS1 = FloatArray(1 * dModel * 1)
    private var cacheS2 = FloatArray(1 * dModel * 1)
    private var cacheK = FloatArray(nLayers * 1 * nHeads * leftCtx * headDim)
    private var cacheV = FloatArray(nLayers * 1 * nHeads * leftCtx * headDim)
    private var cacheConv = FloatArray(nLayers * 1 * dModel * (convKernel - 1))

    private val hannWindow = FloatArray(N_FFT) { i ->
        (0.5 * (1.0 - cos(2.0 * PI * i / N_FFT))).toFloat()
    }

    private val emittedTokenIds = mutableListOf<Int>()
    private val emittedConfidences = mutableListOf<Float>()
    private val frameLogProbs = mutableListOf<FloatArray>()
    private var prevCtcId: Int = 0
    @Volatile private var isRecording = false

    init {
        val opts = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(2)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            try {
                addXnnpack(emptyMap())
            } catch (_: Exception) {}
            if (useNnapi) {
                try {
                    addNnapi()
                } catch (_: Exception) {}
            }
        }
        encoderSession = env.createSession(encoderModelPath, opts)
        shuddhaSession = shuddhaModelPath?.let { env.createSession(it, opts) }
    }

    /** Reset streaming caches for a new utterance or clause. */
    fun resetStream() {
        wavHistory.fill(0f)
        cacheS0.fill(0f)
        cacheS1.fill(0f)
        cacheS2.fill(0f)
        cacheK.fill(0f)
        cacheV.fill(0f)
        cacheConv.fill(0f)
        emittedTokenIds.clear()
        emittedConfidences.clear()
        frameLogProbs.clear()
        prevCtcId = 0
    }

    private fun logAdd(a: Float, b: Float): Float {
        if (a <= LOG_ZERO) return b
        if (b <= LOG_ZERO) return a
        return if (a > b) a + ln1p(exp((b - a).toDouble())).toFloat()
        else b + ln1p(exp((a - b).toDouble())).toFloat()
    }

    private fun scoreLmToken(p3: Int, p2: Int, p1: Int, tokId: Int): Float {
        val k4 = "$p3,$p2,$p1,$tokId"
        lmFourgrams[k4]?.let { return it }
        val k3 = "$p2,$p1,$tokId"
        lmTrigrams[k3]?.let { return it - 0.15f }
        val k2 = "$p1,$tokId"
        lmBigrams[k2]?.let { return it - 0.35f }
        val u = if (lmUnigrams != null && tokId in lmUnigrams.indices) lmUnigrams[tokId] else -9.5f
        return u - 0.65f
    }

    /**
     * Compact CTC Prefix Beam Search with 4-gram Nepali Subword Shallow Fusion (beam=6, alpha=0.35).
     */
    fun ctcPrefixBeamSearch(frames: List<FloatArray>, beamSize: Int = 6, topK: Int = 6, alpha: Float = 0.35f, beta: Float = 0.08f): List<Int> {
        if (frames.isEmpty()) return emptyList()
        data class Beam(val ids: List<Int>, var pb: Float, var pnb: Float, val lmScore: Float)
        var beams = mapOf(emptyList<Int>() to Beam(emptyList(), 0f, LOG_ZERO, 0f))
        val hasLm = lmUnigrams != null || lmBigrams.isNotEmpty()

        for (lp in frames) {
            val lpBlank = lp[BLANK_ID]
            if (lpBlank > -0.04f) {
                for (b in beams.values) {
                    b.pb = logAdd(b.pb, b.pnb) + lpBlank
                    b.pnb = LOG_ZERO
                }
                continue
            }
            val topIndices = (8 until lp.size)
                .filter { lp[it] >= -8.0f }
                .sortedByDescending { lp[it] }
                .take(topK)

            val nextBeams = mutableMapOf<List<Int>, Beam>()
            fun addNext(pref: List<Int>, addPb: Float, addPnb: Float, lmS: Float) {
                val ex = nextBeams[pref]
                if (ex != null) {
                    ex.pb = logAdd(ex.pb, addPb)
                    ex.pnb = logAdd(ex.pnb, addPnb)
                } else {
                    nextBeams[pref] = Beam(pref, addPb, addPnb, lmS)
                }
            }

            for ((pref, b) in beams) {
                val pTot = logAdd(b.pb, b.pnb)
                addNext(pref, pTot + lpBlank, LOG_ZERO, b.lmScore)
                val lastTok = pref.lastOrNull() ?: -1
                val p1 = if (pref.isNotEmpty()) pref[pref.size - 1] else BOS_ID
                val p2 = if (pref.size >= 2) pref[pref.size - 2] else BOS_ID
                val p3 = if (pref.size >= 3) pref[pref.size - 3] else BOS_ID

                for (c in topIndices) {
                    val lpC = lp[c]
                    if (c == lastTok) {
                        if (b.pnb > LOG_ZERO) addNext(pref, LOG_ZERO, b.pnb + lpC, b.lmScore)
                        if (b.pb > LOG_ZERO) {
                            val newPref = pref + c
                            val dLm = if (hasLm) alpha * (scoreLmToken(p3, p2, p1, c) + 4.0f) + beta else 0f
                            addNext(newPref, LOG_ZERO, b.pb + lpC, b.lmScore + dLm)
                        }
                    } else {
                        val newPref = pref + c
                        val dLm = if (hasLm) alpha * (scoreLmToken(p3, p2, p1, c) + 4.0f) + beta else 0f
                        addNext(newPref, LOG_ZERO, pTot + lpC, b.lmScore + dLm)
                    }
                }
            }
            beams = if (nextBeams.size > beamSize) {
                nextBeams.entries
                    .sortedByDescending { logAdd(it.value.pb, it.value.pnb) + it.value.lmScore }
                    .take(beamSize)
                    .associate { it.key to it.value }
            } else {
                nextBeams
            }
        }
        return beams.values.maxByOrNull { logAdd(it.pb, it.pnb) + it.lmScore }?.ids ?: emptyList()
    }

    /**
     * Feed a 160 ms (2,560 samples) 16 kHz mono float32 PCM chunk and run stateful streaming inference.
     */
    fun acceptAudioChunk(pcm2560: FloatArray, melFeatures16x80: FloatArray, listener: Listener?) {
        val t0 = System.nanoTime()

        val melTensor = OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(melFeatures16x80),
            longArrayOf(1, CHUNK_FRAMES.toLong(), N_MELS.toLong())
        )
        val s0Tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(cacheS0), longArrayOf(1, N_MELS.toLong(), 1))
        val s1Tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(cacheS1), longArrayOf(1, dModel.toLong(), 1))
        val s2Tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(cacheS2), longArrayOf(1, dModel.toLong(), 1))
        val kTensor = OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(cacheK),
            longArrayOf(nLayers.toLong(), 1, nHeads.toLong(), leftCtx.toLong(), headDim.toLong())
        )
        val vTensor = OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(cacheV),
            longArrayOf(nLayers.toLong(), 1, nHeads.toLong(), leftCtx.toLong(), headDim.toLong())
        )
        val convTensor = OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(cacheConv),
            longArrayOf(nLayers.toLong(), 1, dModel.toLong(), (convKernel - 1).toLong())
        )

        val feeds = mapOf(
            "mel" to melTensor,
            "cache_s0" to s0Tensor,
            "cache_s1" to s1Tensor,
            "cache_s2" to s2Tensor,
            "cache_k" to kTensor,
            "cache_v" to vTensor,
            "cache_conv" to convTensor
        )

        encoderSession.run(feeds).use { res ->
            val logitsBuf = (res[0] as OnnxTensor).floatBuffer
            cacheS0 = (res[2] as OnnxTensor).floatBuffer.let { b -> FloatArray(b.remaining()).also { b.get(it) } }
            cacheS1 = (res[3] as OnnxTensor).floatBuffer.let { b -> FloatArray(b.remaining()).also { b.get(it) } }
            cacheS2 = (res[4] as OnnxTensor).floatBuffer.let { b -> FloatArray(b.remaining()).also { b.get(it) } }
            cacheK = (res[5] as OnnxTensor).floatBuffer.let { b -> FloatArray(b.remaining()).also { b.get(it) } }
            cacheV = (res[6] as OnnxTensor).floatBuffer.let { b -> FloatArray(b.remaining()).also { b.get(it) } }
            cacheConv = (res[7] as OnnxTensor).floatBuffer.let { b -> FloatArray(b.remaining()).also { b.get(it) } }

            val vocabSize = vocab.size
            val encFrames = CHUNK_FRAMES / subsamplingFactor // 4 frames per 160 ms chunk with 4x subsampling
            for (t in 0 until encFrames) {
                var maxLogit = -1e9f
                val rawRow = FloatArray(vocabSize)
                for (v in 0 until vocabSize) {
                    val l = logitsBuf.get(t * vocabSize + v)
                    rawRow[v] = l
                    if (l > maxLogit) maxLogit = l
                }
                var sumExp = 0.0
                for (v in 0 until vocabSize) sumExp += exp((rawRow[v] - maxLogit).toDouble())
                val lse = maxLogit + ln(sumExp).toFloat()
                var bestId = 0
                var bestLp = -1e9f
                for (v in 0 until vocabSize) {
                    rawRow[v] -= lse
                    if (rawRow[v] > bestLp) {
                        bestLp = rawRow[v]
                        bestId = v
                    }
                }
                frameLogProbs.add(rawRow)
                if (bestId != 0 && bestId >= 8 && bestId != prevCtcId) {
                    emittedTokenIds.add(bestId)
                    emittedConfidences.add(exp(bestLp.toDouble()).toFloat())
                }
                prevCtcId = bestId
            }
        }

        melTensor.close()
        s0Tensor.close()
        s1Tensor.close()
        s2Tensor.close()
        kTensor.close()
        vTensor.close()
        convTensor.close()

        val dtMs = (System.nanoTime() - t0) / 1_000_000f
        val rtf = dtMs / 160f
        listener?.onPartialTranscript(decodeTokens(emittedTokenIds), dtMs, rtf)
    }

    /**
     * Finalize the current spoken clause on VAD pause with 4-gram Prefix Beam Search + Shrota-Shuddha polish.
     */
    fun finalizeClause(listener: Listener?) {
        val beamIds = if (frameLogProbs.isNotEmpty()) ctcPrefixBeamSearch(frameLogProbs) else emptyList()
        val activeIds = if (beamIds.isNotEmpty()) beamIds else emittedTokenIds.toList()
        val rawText = decodeTokens(activeIds)
        if (rawText.isBlank()) return

        val t0 = System.nanoTime()
        var refinedIds = activeIds
        shuddhaSession?.let { sess ->
            val inpArr = activeIds.map { it.toLong() }.toLongArray()
            OnnxTensor.createTensor(env, LongBuffer.wrap(inpArr), longArrayOf(1, inpArr.size.toLong())).use { inpTensor ->
                sess.run(mapOf("input_ids" to inpTensor)).use { out ->
                    val tokBuf = (out[0] as OnnxTensor).floatBuffer
                    val vocabSize = vocab.size
                    val updated = mutableListOf<Int>()
                    for (i in inpArr.indices) {
                        var bestId = inpArr[i].toInt()
                        var bestVal = -1e9f
                        for (v in 0 until vocabSize) {
                            val score = tokBuf.get(i * vocabSize + v)
                            if (score > bestVal) {
                                bestVal = score
                                bestId = v
                            }
                        }
                        updated.add(bestId)
                    }
                    refinedIds = updated
                }
            }
        }
        var shuddhaText = decodeTokens(refinedIds)
        if (!shuddhaText.endsWith("।") && !shuddhaText.endsWith("?")) {
            shuddhaText += "।"
        }
        val shuddhaMs = (System.nanoTime() - t0) / 1_000_000f
        val avgConf = if (emittedConfidences.isEmpty()) 0.9f else emittedConfidences.average().toFloat()
        listener?.onFinalClause(rawText, shuddhaText, avgConf, shuddhaMs)
        resetStream()
    }

    private fun decodeTokens(ids: List<Int>): String {
        val sb = StringBuilder()
        for (id in ids) {
            if (id <= 7 || id >= vocab.size) continue
            val piece = vocab[id]
            if (piece.startsWith("<extra_")) continue
            sb.append(piece)
        }
        return sb.toString().replace(WORD_BOUNDARY, " ").trim()
    }

    @SuppressLint("MissingPermission")
    fun startLiveMicrophone(listener: Listener, extractMel16x80: (FloatArray) -> FloatArray) {
        if (isRecording) return
        isRecording = true
        val minBuf = max(
            AudioRecord.getMinBufferSize(SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT),
            CHUNK_SAMPLES * 4
        )
        val recorder = AudioRecord(
            MediaRecorder.AudioSource.VOICE_RECOGNITION,
            SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            minBuf
        )
        Thread {
            val shortBuf = ShortArray(CHUNK_SAMPLES)
            val floatBuf = FloatArray(CHUNK_SAMPLES)
            var silentChunks = 0
            var hasSpeech = false
            recorder.startRecording()
            try {
                while (isRecording) {
                    val read = recorder.read(shortBuf, 0, CHUNK_SAMPLES)
                    if (read <= 0) continue
                    var sumSq = 0.0
                    for (i in 0 until read) {
                        val f = shortBuf[i] / 32768f
                        floatBuf[i] = f
                        sumSq += f * f
                    }
                    val rms = sqrt(sumSq / read).toFloat()
                    if (rms > 0.008f) {
                        hasSpeech = true
                        silentChunks = 0
                        acceptAudioChunk(floatBuf, extractMel16x80(floatBuf), listener)
                    } else if (hasSpeech) {
                        acceptAudioChunk(floatBuf, extractMel16x80(floatBuf), listener)
                        silentChunks++
                        if (silentChunks >= 5) { // 800 ms micro-pause -> Shrota-Shuddha commit
                            finalizeClause(listener)
                            hasSpeech = false
                            silentChunks = 0
                        }
                    }
                }
            } finally {
                recorder.stop()
                recorder.release()
            }
        }.start()
    }

    fun stopLiveMicrophone() {
        isRecording = false
    }

    override fun close() {
        stopLiveMicrophone()
        encoderSession.close()
        shuddhaSession?.close()
    }
}
