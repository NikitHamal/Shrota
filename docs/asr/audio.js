// Browser audio I/O: decode any audio file to 16 kHz mono, and capture the microphone as a
// 16 kHz PCM stream via AudioWorklet (ScriptProcessor fallback).
export const TARGET_SR = 16000;

export async function decodeToMono16k(arrayBuffer) {
  const AC = window.AudioContext || window.webkitAudioContext;
  const ctx = new AC();
  let buf;
  try {
    buf = await new Promise((resolve, reject) => {
      const p = ctx.decodeAudioData(arrayBuffer.slice(0), resolve, reject);
      if (p && p.catch) p.catch(reject);
    });
  } finally {
    ctx.close?.();
  }
  const OAC = window.OfflineAudioContext || window.webkitOfflineAudioContext;
  const off = new OAC(1, Math.max(1, Math.ceil(buf.duration * TARGET_SR)), TARGET_SR);
  const src = off.createBufferSource();
  src.buffer = buf;
  src.connect(off.destination);
  src.start();
  const rendered = await off.startRendering();
  return rendered.getChannelData(0).slice();
}

/** Streaming area-average resampler (anti-aliased for the usual 44.1/48 kHz -> 16 kHz case). */
export class StreamResampler {
  constructor(inRate, outRate = TARGET_SR) {
    this.ratio = inRate / outRate;
    this.carry = new Float32Array(0);
    this.pos = 0; // fractional read position into carry
  }

  push(input) {
    if (Math.abs(this.ratio - 1) < 1e-6) return input;
    const data = new Float32Array(this.carry.length + input.length);
    data.set(this.carry, 0);
    data.set(input, this.carry.length);
    const out = [];
    let pos = this.pos;
    while (pos + this.ratio <= data.length) {
      const a = pos, b = pos + this.ratio;
      let i = Math.floor(a);
      let acc = 0;
      while (i < b && i < data.length) {
        const lo = Math.max(a, i), hi = Math.min(b, i + 1);
        acc += data[i] * (hi - lo);
        i++;
      }
      out.push(acc / this.ratio);
      pos = b;
    }
    const keep = Math.floor(pos);
    this.carry = data.slice(keep);
    this.pos = pos - keep;
    return Float32Array.from(out);
  }
}

export class MicCapture {
  /** @param {(pcm16k: Float32Array) => void} onPcm */
  constructor(onPcm, onLevel) {
    this.onPcm = onPcm;
    this.onLevel = onLevel;
    this.ctx = null;
    this.stream = null;
    this.node = null;
    this.source = null;
  }

  async start() {
    if (!navigator.mediaDevices?.getUserMedia) throw new Error("Microphone is not available in this browser or context (HTTPS required).");
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
    const AC = window.AudioContext || window.webkitAudioContext;
    this.ctx = new AC();
    await this.ctx.resume();
    this.resampler = new StreamResampler(this.ctx.sampleRate);
    this.source = this.ctx.createMediaStreamSource(this.stream);
    const handle = (block) => {
      if (this.onLevel) {
        let peak = 0;
        for (let i = 0; i < block.length; i++) peak = Math.max(peak, Math.abs(block[i]));
        this.onLevel(peak);
      }
      const pcm = this.resampler.push(block);
      if (pcm.length) this.onPcm(pcm);
    };
    if (this.ctx.audioWorklet) {
      await this.ctx.audioWorklet.addModule(new URL("./mic-worklet.js", import.meta.url));
      this.node = new AudioWorkletNode(this.ctx, "shrota-mic", { numberOfInputs: 1, numberOfOutputs: 0 });
      this.node.port.onmessage = (e) => handle(e.data);
      this.source.connect(this.node);
    } else {
      this.node = this.ctx.createScriptProcessor(2048, 1, 1);
      this.node.onaudioprocess = (e) => handle(new Float32Array(e.inputBuffer.getChannelData(0)));
      this.source.connect(this.node);
      this.node.connect(this.ctx.destination);
    }
  }

  async stop() {
    try { this.node?.disconnect(); this.source?.disconnect(); } catch (_) { /* already disconnected */ }
    this.stream?.getTracks().forEach((t) => t.stop());
    await this.ctx?.close?.();
    this.ctx = this.stream = this.node = this.source = null;
  }
}
