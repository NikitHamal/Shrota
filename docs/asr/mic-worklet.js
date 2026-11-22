// AudioWorklet processor: batches 128-sample render quanta into ~64 ms blocks for the main thread.
class ShrotaMic extends AudioWorkletProcessor {
  constructor() {
    super();
    this.block = new Float32Array(1024);
    this.n = 0;
  }

  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    let i = 0;
    while (i < ch.length) {
      const take = Math.min(ch.length - i, this.block.length - this.n);
      this.block.set(ch.subarray(i, i + take), this.n);
      this.n += take;
      i += take;
      if (this.n === this.block.length) {
        this.port.postMessage(this.block.slice());
        this.n = 0;
      }
    }
    return true;
  }
}
registerProcessor("shrota-mic", ShrotaMic);
