// Causal 80-bin log-mel extractor. 1:1 port of shrota/features.py (LogMelExtractor + streaming history).
// sr=16 kHz, n_fft=400, hop=160, 80 mels (20-7600 Hz), global CMVN. Frame k covers samples
// [160k-240, 160k+160) so chunked streaming and offline extraction are identical.
export const SAMPLE_RATE = 16000;
export const N_FFT = 400;
export const HOP = 160;
export const N_MELS = 80;
const N_FREQ = N_FFT / 2 + 1;
const FMIN = 20.0;
const FMAX = 7600.0;

const hzToMel = (f) => 2595.0 * Math.log10(1.0 + f / 700.0);
const melToHz = (m) => 700.0 * (10.0 ** (m / 2595.0) - 1.0);

function buildMelFilterbank() {
  const fftFreqs = Float64Array.from({ length: N_FREQ }, (_, i) => (i * SAMPLE_RATE) / 2 / (N_FREQ - 1));
  const melMin = hzToMel(FMIN);
  const melMax = hzToMel(FMAX);
  const hz = Float64Array.from({ length: N_MELS + 2 }, (_, i) => melToHz(melMin + ((melMax - melMin) * i) / (N_MELS + 1)));
  const fb = [];
  for (let i = 0; i < N_MELS; i++) {
    const left = hz[i], center = hz[i + 1], right = hz[i + 2];
    const enorm = 2.0 / Math.max(right - left, 1e-8);
    const idx = [];
    const w = [];
    for (let k = 0; k < N_FREQ; k++) {
      const up = (fftFreqs[k] - left) / Math.max(center - left, 1e-8);
      const down = (right - fftFreqs[k]) / Math.max(right - center, 1e-8);
      const v = Math.max(0, Math.min(up, down)) * enorm;
      if (v > 0) { idx.push(k); w.push(v); }
    }
    fb.push({ idx: Int32Array.from(idx), w: Float32Array.from(w) });
  }
  return fb;
}

let TABLES = null;
function tables() {
  if (TABLES) return TABLES;
  // Periodic Hann window folded into the real-DFT basis: 201 x 400 cos and sin matrices.
  const cos = new Float32Array(N_FREQ * N_FFT);
  const sin = new Float32Array(N_FREQ * N_FFT);
  for (let n = 0; n < N_FFT; n++) {
    const win = 0.5 - 0.5 * Math.cos((2 * Math.PI * n) / N_FFT);
    for (let k = 0; k < N_FREQ; k++) {
      const a = (2 * Math.PI * ((k * n) % N_FFT)) / N_FFT;
      cos[k * N_FFT + n] = win * Math.cos(a);
      sin[k * N_FFT + n] = -win * Math.sin(a);
    }
  }
  TABLES = { cos, sin, fb: buildMelFilterbank() };
  return TABLES;
}

export class LogMelStream {
  /** @param {{mean:number[], std:number[]}} cmvn */
  constructor(cmvn) {
    this.mean = Float32Array.from(cmvn.mean);
    this.istd = Float32Array.from(cmvn.std, (s) => 1.0 / Math.max(s, 1e-5));
    this.power = new Float32Array(N_FREQ);
    this.reset();
  }

  reset() {
    this.buf = new Float32Array(N_FFT - HOP); // 240 samples of zero history
  }

  /** Push 16 kHz mono samples; returns normalized log-mel rows as Float32Array(nFrames * 80). */
  push(samples) {
    const joined = new Float32Array(this.buf.length + samples.length);
    joined.set(this.buf, 0);
    joined.set(samples, this.buf.length);
    const nFrames = joined.length >= N_FFT ? Math.floor((joined.length - N_FFT) / HOP) + 1 : 0;
    const out = new Float32Array(nFrames * N_MELS);
    const { cos, sin, fb } = tables();
    const p = this.power;
    for (let f = 0; f < nFrames; f++) {
      const off = f * HOP;
      for (let k = 0; k < N_FREQ; k++) {
        let re = 0, im = 0;
        const base = k * N_FFT;
        for (let n = 0; n < N_FFT; n++) {
          const s = joined[off + n];
          re += s * cos[base + n];
          im += s * sin[base + n];
        }
        p[k] = re * re + im * im;
      }
      for (let m = 0; m < N_MELS; m++) {
        const { idx, w } = fb[m];
        let acc = 0;
        for (let j = 0; j < idx.length; j++) acc += p[idx[j]] * w[j];
        out[f * N_MELS + m] = (Math.log(Math.max(acc, 1e-10)) - this.mean[m]) * this.istd[m];
      }
    }
    this.buf = joined.slice(nFrames * HOP);
    return out;
  }
}
