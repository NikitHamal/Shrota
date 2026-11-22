// Shrota streaming ASR runtime for the browser (onnxruntime-web WASM) and Node (same code, same ONNX).
// encoder: (mel, caches...) -> (ctc_logits, enc_out, next caches...)
// decoder: (enc_frame, tok_prev2, tok_prev1) -> (tdt_tok_logits, tdt_dur_logits)
// Decoding mirrors scripts/train_base.py evaluate_asr_model: CTC greedy with 2-gram TDT rescoring.
import { LogMelStream, N_MELS } from "./features.js";
import { decodeTokens, cleanSubwordStutter, tokensToWords } from "./text.js";

const BLANK = 0;
const BOS = 2;
const CTC_W = 0.85;
const TDT_W = 0.15;
const FRAME_SEC = 0.08;
const CACHE_NAMES = ["s0", "s1", "s2", "k", "v", "conv"];

function logSoftmaxInto(src, off, n, dst) {
  let max = -Infinity;
  for (let i = 0; i < n; i++) if (src[off + i] > max) max = src[off + i];
  let sum = 0;
  for (let i = 0; i < n; i++) sum += Math.exp(src[off + i] - max);
  const lse = max + Math.log(sum);
  for (let i = 0; i < n; i++) dst[i] = src[off + i] - lse;
}

export class ShrotaEngine {
  /**
   * @param {object} opts
   * @param {object} opts.ort onnxruntime-web (or -node) module
   * @param {(path:string)=>Promise<ArrayBuffer|Uint8Array>} opts.loadBytes
   * @param {(path:string)=>Promise<object>} opts.loadJson
   * @param {object} opts.manifest models/manifest.json content
   */
  constructor({ ort, loadBytes, loadJson, manifest }) {
    this.ort = ort;
    this.loadBytes = loadBytes;
    this.loadJson = loadJson;
    this.manifest = manifest;
    this.sessions = {};
    this.vocab = null;
    this.cmvn = null;
  }

  async loadShared() {
    if (!this.vocab) {
      const tok = await this.loadJson(this.manifest.tokenizer);
      this.vocab = tok.vocab;
    }
    if (!this.cmvn) this.cmvn = await this.loadJson(this.manifest.cmvn);
  }

  loadModel(key, onProgress) {
    this.loading = this.loading || {};
    if (!this.loading[key]) {
      this.loading[key] = this.#loadModel(key, onProgress).catch((e) => {
        delete this.loading[key];
        throw e;
      });
    }
    return this.loading[key];
  }

  async #loadModel(key, onProgress) {
    await this.loadShared();
    if (this.sessions[key]) return this.sessions[key];
    const cfg = this.manifest.models[key];
    if (!cfg) throw new Error(`Unknown model: ${key}`);
    const opts = { executionProviders: ["wasm"], graphOptimizationLevel: "all" };
    onProgress?.("encoder");
    const enc = await this.ort.InferenceSession.create(new Uint8Array(await this.loadBytes(cfg.encoder)), opts);
    onProgress?.("decoder");
    const dec = await this.ort.InferenceSession.create(new Uint8Array(await this.loadBytes(cfg.decoder)), opts);
    this.sessions[key] = { cfg, enc, dec };
    return this.sessions[key];
  }

  createStream(key, { lang = "ne", chunkFrames = 16 } = {}) {
    const m = this.sessions[key];
    if (!m) throw new Error(`Model ${key} not loaded`);
    return new ShrotaStream(this, m, { lang, chunkFrames });
  }

  /** Offline transcription helper: runs the full waveform through a stream. */
  async transcribe(key, pcm, { lang = "ne", chunkFrames = 64, onProgress } = {}) {
    const stream = this.createStream(key, { lang, chunkFrames });
    const step = 16000 * 2;
    for (let i = 0; i < pcm.length; i += step) {
      await stream.feed(pcm.subarray(i, Math.min(pcm.length, i + step)));
      onProgress?.(Math.min(1, (i + step) / pcm.length), stream);
    }
    await stream.flush();
    return stream.result();
  }
}

export class ShrotaStream {
  constructor(engine, model, { lang, chunkFrames }) {
    this.engine = engine;
    this.model = model;
    this.lang = lang;
    this.chunkFrames = Math.max(8, Math.round(chunkFrames / 8) * 8);
    this.mel = new LogMelStream(engine.cmvn);
    this.pending = new Float32Array(0);
    this.caches = this.#initCaches();
    this.tokens = [];
    this.confs = [];
    this.prevCtc = BLANK;
    this.prev2 = BOS;
    this.prev1 = BOS;
    this.encFrames = 0;
    this.silentFrames = 0;
    this.audioSec = 0;
    this.computeMs = 0;
    this.lastChunkMs = 0;
    this.tokBuf = new Float32Array(engine.vocab.length);
    this.tdtBuf = new Float32Array(engine.vocab.length);
  }

  #initCaches() {
    const c = this.model.cfg;
    const T = this.engine.ort.Tensor;
    const z = (dims) => new T("float32", new Float32Array(dims.reduce((a, b) => a * b, 1)), dims);
    return {
      s0: z([1, N_MELS, 1]),
      s1: z([1, c.d_model, 1]),
      s2: z([1, c.d_model, 1]),
      k: z([c.n_layers, 1, c.n_heads, c.left_ctx, c.head_dim]),
      v: z([c.n_layers, 1, c.n_heads, c.left_ctx, c.head_dim]),
      conv: z([c.n_layers, 1, c.d_model, c.conv_cache]),
    };
  }

  /** Feed 16 kHz mono float32 PCM. Returns number of newly emitted tokens. */
  async feed(pcm) {
    this.audioSec += pcm.length / 16000;
    const rows = this.mel.push(pcm);
    return this.#consumeMel(rows, false);
  }

  /** Flush the tail (zero-pads mel to a multiple of 8 frames like the offline evaluator). */
  async flush() {
    return this.#consumeMel(new Float32Array(0), true);
  }

  async #consumeMel(rows, final) {
    if (rows.length) {
      const joined = new Float32Array(this.pending.length + rows.length);
      joined.set(this.pending, 0);
      joined.set(rows, this.pending.length);
      this.pending = joined;
    }
    let emitted = 0;
    const chunkLen = this.chunkFrames * N_MELS;
    while (this.pending.length >= chunkLen) {
      emitted += await this.#runChunk(this.pending.subarray(0, chunkLen), this.chunkFrames);
      this.pending = this.pending.slice(chunkLen);
    }
    if (final && this.pending.length) {
      const frames = this.pending.length / N_MELS;
      const padded = Math.ceil(frames / 8) * 8;
      const buf = new Float32Array(padded * N_MELS);
      buf.set(this.pending, 0);
      this.pending = new Float32Array(0);
      emitted += await this.#runChunk(buf, padded);
    }
    return emitted;
  }

  async #runChunk(melRows, nFrames) {
    const { ort } = this.engine;
    const { enc, dec } = this.model;
    const t0 = performance.now();
    const c = this.caches;
    const out = await enc.run({
      mel: new ort.Tensor("float32", Float32Array.from(melRows), [1, nFrames, N_MELS]),
      cache_s0: c.s0, cache_s1: c.s1, cache_s2: c.s2, cache_k: c.k, cache_v: c.v, cache_conv: c.conv,
    });
    this.caches = { s0: out.next_s0, s1: out.next_s1, s2: out.next_s2, k: out.next_k, v: out.next_v, conv: out.next_conv };
    const ctc = out.ctc_logits.data;
    const encOut = out.enc_out.data;
    const V = out.ctc_logits.dims[2];
    const D = out.enc_out.dims[2];
    const T = out.ctc_logits.dims[1];
    let emitted = 0;
    for (let t = 0; t < T; t++) {
      const off = t * V;
      let best = 0, bestV = -Infinity;
      for (let i = 0; i < V; i++) if (ctc[off + i] > bestV) { bestV = ctc[off + i]; best = i; }
      const frameIdx = this.encFrames + t;
      if (best !== BLANK) this.silentFrames = 0; else this.silentFrames++;
      if (best !== BLANK && best !== this.prevCtc) {
        const d = await dec.run({
          enc_frame: new ort.Tensor("float32", encOut.slice(t * D, (t + 1) * D), [1, D]),
          tok_prev2: new ort.Tensor("int64", BigInt64Array.from([BigInt(this.prev2)]), [1]),
          tok_prev1: new ort.Tensor("int64", BigInt64Array.from([BigInt(this.prev1)]), [1]),
        });
        logSoftmaxInto(ctc, off, V, this.tokBuf);
        logSoftmaxInto(d.tdt_tok_logits.data, 0, V, this.tdtBuf);
        let id = 0, score = -Infinity;
        for (let i = 0; i < V; i++) {
          if (i === BLANK) continue;
          const s = CTC_W * this.tokBuf[i] + TDT_W * this.tdtBuf[i];
          if (s > score) { score = s; id = i; }
        }
        this.tokens.push({ id, frame: frameIdx });
        this.confs.push(Math.exp(this.tokBuf[best]));
        this.prev2 = this.prev1;
        this.prev1 = id;
        emitted++;
      }
      this.prevCtc = best;
    }
    this.encFrames += T;
    this.lastChunkMs = performance.now() - t0;
    this.computeMs += this.lastChunkMs;
    return emitted;
  }

  rawText() {
    return cleanSubwordStutter(decodeTokens(this.tokens.map((t) => t.id), this.engine.vocab), this.lang);
  }

  words() {
    return tokensToWords(this.tokens, this.engine.vocab, FRAME_SEC);
  }

  /** Seconds of trailing blank frames since the last emitted token. */
  silenceSec() {
    return this.silentFrames * FRAME_SEC;
  }

  /** Drop the emitted tokens (clause commit) while keeping encoder caches and decoder context. */
  takeSegment() {
    const seg = { raw: this.rawText(), words: this.words(), tokens: this.tokens, confs: this.confs };
    this.tokens = [];
    this.confs = [];
    this.prev2 = BOS;
    this.prev1 = BOS;
    return seg;
  }

  result() {
    const conf = this.confs.length ? this.confs.reduce((a, b) => a + b, 0) / this.confs.length : 0;
    return {
      raw: this.rawText(),
      words: this.words(),
      confidence: conf,
      audioSec: this.audioSec,
      computeMs: this.computeMs,
      rtf: this.audioSec ? this.computeMs / 1000 / this.audioSec : 0,
    };
  }
}
