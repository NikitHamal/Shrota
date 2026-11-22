// Live microphone session: 16 kHz mic PCM -> 160 ms streaming chunks -> partial text, clause commits on pauses.
import { MicCapture } from "./asr/audio.js";

const COMMIT_SILENCE_SEC = 1.0;
const LONG_CLAUSE_TOKENS = 60;
const LONG_CLAUSE_SILENCE_SEC = 0.4;

export class LiveSession {
  /**
   * @param {object} o
   * @param {import("./asr/engine.js").ShrotaEngine} o.engine
   * @param {string} o.modelKey
   * @param {{onPartial:Function,onCommit:Function,onStats:Function,onLevel:Function,onError:Function}} o.ui
   */
  constructor({ engine, modelKey, ui }) {
    this.stream = engine.createStream(modelKey, { lang: "ne", chunkFrames: 16 });
    this.ui = ui;
    this.queue = Promise.resolve();
    this.backlogSec = 0;
    this.mic = new MicCapture((pcm) => this.#enqueue(pcm), (lv) => ui.onLevel(lv));
    this.running = false;
  }

  async start() {
    await this.mic.start();
    this.running = true;
  }

  #enqueue(pcm) {
    if (!this.running) return;
    const sec = pcm.length / 16000;
    this.backlogSec += sec;
    this.queue = this.queue
      .then(async () => {
        await this.stream.feed(pcm);
        this.backlogSec = Math.max(0, this.backlogSec - sec);
        this.#afterFeed();
      })
      .catch((e) => this.ui.onError(e));
  }

  #afterFeed() {
    const s = this.stream;
    this.ui.onPartial(s.rawText());
    const n = s.tokens.length;
    if (n && (s.silenceSec() >= COMMIT_SILENCE_SEC || (n > LONG_CLAUSE_TOKENS && s.silenceSec() >= LONG_CLAUSE_SILENCE_SEC))) {
      this.#commit();
    }
    this.ui.onStats({
      chunkMs: s.lastChunkMs,
      rtf: s.audioSec ? s.computeMs / 1000 / s.audioSec : 0,
      audioSec: s.audioSec,
      backlogSec: this.backlogSec,
    });
  }

  #commit() {
    const seg = this.stream.takeSegment();
    if (seg.raw) this.ui.onCommit(seg.raw);
    this.ui.onPartial("");
  }

  async stop() {
    this.running = false;
    await this.mic.stop();
    await this.queue;
    await this.stream.flush();
    if (this.stream.tokens.length) this.#commit();
    this.ui.onPartial("");
  }
}
