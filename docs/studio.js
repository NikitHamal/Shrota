// Studio tab controller: model/Shuddha controls, 18 example clips with decoded previews, file upload, live microphone.
import { applyClientShuddha } from "./shuddha_client.js";
import { getEngine } from "./asr/runtime.js";
import { decodeToMono16k } from "./asr/audio.js";
import { cleanSubwordStutter } from "./asr/text.js";
import { scoreAgainst } from "./asr/metrics.js";
import { LiveSession } from "./live.js";
import { $, esc, diffHtml, pct } from "./view.js";

const st = {
  model: "base",
  shuddha: true,
  samples: [],
  item: null, // current clip: {title, pcm, audioUrl, reference, words, raw, done, busy, progress, score}
  runId: 0,
  live: null,
  committed: [], // live mode: raw clauses
  partial: "",
  stats: { chunkMs: null, rtf: null, audioSec: null },
  manifest: null,
};

const MODEL_LABEL = { base: "Shrota-Base v2", tiny: "Shrota-Tiny v2" };

function setStatus(text, kind = "idle") {
  const chip = $("status-chip");
  chip.textContent = text;
  chip.className = `m3-chip status-${kind}`;
}

function setNote(text) {
  $("status-text").textContent = text || "";
}

// ---------- rendering ----------

function refine(raw, final, lang = "ne") {
  // Shuddha's lexicon and postposition rules are Nepali-only; Sanskrit passes through untouched.
  return st.shuddha && lang !== "sa" ? applyClientShuddha(raw, final) : raw;
}

function renderStats() {
  const cfg = st.manifest?.models?.[st.model];
  const s = st.stats;
  const cells = [
    [cfg?.size_mb ? `${cfg.size_mb} MB` : "-", "INT8 model size"],
    [s.chunkMs != null ? `${s.chunkMs.toFixed(0)} ms` : "-", "Compute per chunk"],
    [s.rtf ? `${(1 / s.rtf).toFixed(0)}x` : "-", s.rtf ? `Real time (RTF ${s.rtf.toFixed(3)})` : "Real time"],
    [s.audioSec != null ? `${s.audioSec.toFixed(1)} s` : "-", "Audio duration"],
  ];
  $("stat-row").innerHTML = cells
    .map(([v, l]) => `<div class="stat"><div class="stat-value">${v}</div><div class="stat-label">${l}</div></div>`)
    .join("");
}

function renderModelSelector() {
  const models = st.manifest?.models || { base: {}, tiny: {} };
  $("model-selector").innerHTML = Object.keys(models)
    .map((k) => `<button class="seg-btn${st.model === k ? " active" : ""}" data-model="${k}">${MODEL_LABEL[k] || k}</button>`)
    .join("");
  $("model-selector").querySelectorAll(".seg-btn").forEach((b) =>
    b.addEventListener("click", () => switchModel(b.dataset.model))
  );
}

function audioPlaying() {
  const a = $("studio-audio");
  return !a.hidden && !a.paused && !a.ended;
}

function renderClip() {
  const it = st.item;
  if (!it) return;
  const playing = audioPlaying();
  const t = $("studio-audio").currentTime;
  const words = it.words || [];
  const visible = playing ? words.filter((w) => w.t0 <= t) : words;
  const rawVisible = playing || !it.done ? cleanSubwordStutter(visible.map((w) => w.word).join(" "), it.lang) : it.raw;
  const final = !playing && it.done;
  const refined = refine(rawVisible, final, it.lang);
  $("out-text").innerHTML = rawVisible ? (st.shuddha ? diffHtml(rawVisible, refined) : esc(rawVisible)) : `<span class="placeholder">${it.busy ? "Transcribing..." : ""}</span>`;
  $("raw-text").textContent = rawVisible;
  $("ref-block").hidden = !it.reference;
  $("ref-text").textContent = it.reference || "";
  const sc = $("score-chips");
  if (it.done && it.reference && it.score) {
    sc.hidden = false;
    sc.innerHTML = [
      `<span class="m3-chip">Raw WER ${pct(it.score.raw.wer)} · CER ${pct(it.score.raw.cer)}</span>`,
      `<span class="m3-chip secondary">Shuddha WER ${pct(it.score.shuddha.wer)} · CER ${pct(it.score.shuddha.cer)}</span>`,
    ].join("");
  } else sc.hidden = true;
  $("words-block").hidden = words.length === 0;
  $("word-timestamps").innerHTML = words
    .map((w, i) => `<span class="word-pill${playing && t >= w.t0 && t < w.t1 + 0.25 ? " active" : ""}" data-i="${i}">${esc(w.word)} <small>${w.t0.toFixed(2)}s</small></span>`)
    .join("");
  $("progress").hidden = !it.busy;
  $("progress-bar").style.width = `${Math.round((it.progress || 0) * 100)}%`;
}

function renderLive() {
  const lines = st.committed.map((raw) => `<span class="commit">${esc(refine(raw, true))}</span>`);
  if (st.partial) lines.push(`<span class="partial">${esc(st.partial)}</span>`);
  $("out-text").innerHTML = lines.length ? lines.join(" ") : `<span class="placeholder">${st.live ? "Listening... speak Nepali" : ""}</span>`;
  $("raw-text").textContent = [...st.committed, st.partial].filter(Boolean).join(" ");
  $("ref-block").hidden = true;
  $("score-chips").hidden = true;
  $("words-block").hidden = true;
  $("progress").hidden = true;
}

function render() {
  renderStats();
  if (st.live) renderLive();
  else if (st.item) renderClip();
}

function tick() {
  if (audioPlaying() && st.item && !st.live) renderClip();
  requestAnimationFrame(tick);
}

// ---------- model + transcription ----------

async function ensureModel(key) {
  const engine = await getEngine();
  if (!st.manifest) { st.manifest = engine.manifest; renderModelSelector(); }
  if (!engine.sessions[key]) {
    setStatus("Loading model", "busy");
    setNote(`Downloading ${MODEL_LABEL[key]} (${engine.manifest.models[key].size_mb} MB) and starting the WebAssembly runtime...`);
    await engine.loadModel(key);
    setNote("");
  }
  return engine;
}

async function transcribeItem(item) {
  const run = ++st.runId;
  item.words = []; item.raw = ""; item.done = false; item.busy = true; item.progress = 0; item.score = null;
  render();
  try {
    const engine = await ensureModel(st.model);
    if (run !== st.runId) return;
    setStatus("Transcribing", "busy");
    const stream = engine.createStream(st.model, { lang: item.lang || "ne", chunkFrames: 16 });
    const step = 16000;
    for (let i = 0; i < item.pcm.length; i += step) {
      await stream.feed(item.pcm.subarray(i, Math.min(item.pcm.length, i + step)));
      if (run !== st.runId) return;
      item.words = stream.words();
      item.progress = Math.min(1, (i + step) / item.pcm.length);
      st.stats = { chunkMs: stream.lastChunkMs, rtf: stream.audioSec ? stream.computeMs / 1000 / stream.audioSec : 0, audioSec: stream.audioSec };
      render();
    }
    await stream.flush();
    if (run !== st.runId) return;
    const res = stream.result();
    item.words = res.words; item.raw = res.raw; item.done = true; item.busy = false;
    st.stats = { chunkMs: stream.lastChunkMs, rtf: res.rtf, audioSec: res.audioSec };
    if (item.reference) {
      item.score = { raw: scoreAgainst(item.reference, res.raw), shuddha: scoreAgainst(item.reference, item.lang === "sa" ? res.raw : applyClientShuddha(res.raw, true)) };
    }
    setStatus("Ready", "ok");
    render();
  } catch (err) {
    item.busy = false;
    console.error(err);
    setStatus("Error", "error");
    setNote(String(err.message || err));
    render();
  }
}

async function switchModel(key) {
  if (key === st.model) return;
  if (st.live) await stopLive();
  st.model = key;
  renderModelSelector();
  renderSamples();
  if (st.item?.pcm) transcribeItem(st.item);
  else ensureModel(key).then(() => setStatus("Ready", "ok")).catch(showError);
}

function showError(err) {
  console.error(err);
  setStatus("Error", "error");
  setNote(String(err.message || err));
}

// ---------- inputs: samples, files, microphone ----------

function attachAudio(url) {
  const a = $("studio-audio");
  a.hidden = false;
  if (a.dataset.src !== url) { a.src = url; a.dataset.src = url; }
  return a;
}

async function loadSample(sample, play = true) {
  if (st.live) await stopLive();
  document.querySelectorAll(".sample-item").forEach((el) => el.classList.toggle("active", el.dataset.id === sample.id));
  const audio = attachAudio(sample.audio_url);
  audio.currentTime = 0;
  if (play) audio.play().catch(() => {});
  const preRaw = st.model === "tiny" ? (sample.tiny_shuddha || sample.base_raw || "") : (sample.base_raw || sample.base_shuddha || "");
  const item = {
    title: sample.title,
    lang: sample.lang || "ne",
    reference: sample.reference,
    pcm: null,
    words: sample.words || [],
    raw: preRaw,
    busy: true,
    done: Boolean(preRaw),
    progress: preRaw ? 1 : 0,
    score: preRaw && sample.reference ? { raw: scoreAgainst(sample.reference, preRaw), shuddha: scoreAgainst(sample.reference, (sample.lang === "sa" ? preRaw : applyClientShuddha(preRaw, true))) } : null,
  };
  st.item = item;
  render();
  try {
    const buf = await (await fetch(new URL(sample.audio_url, document.baseURI))).arrayBuffer();
    item.pcm = await decodeToMono16k(buf);
    if (st.item !== item) return;
    await transcribeItem(item);
  } catch (err) { showError(err); }
}

async function loadFile(file) {
  if (!file) return;
  if (st.live) await stopLive();
  document.querySelectorAll(".sample-item").forEach((el) => el.classList.remove("active"));
  const audio = attachAudio(URL.createObjectURL(file));
  audio.currentTime = 0;
  const item = { title: file.name, lang: "ne", reference: "", pcm: null, words: [], busy: true, done: false, progress: 0 };
  st.item = item;
  setStatus("Decoding audio", "busy");
  render();
  try {
    item.pcm = await decodeToMono16k(await file.arrayBuffer());
    await transcribeItem(item);
    audio.play().catch(() => {});
  } catch (err) {
    item.busy = false;
    showError(new Error(`Could not decode "${file.name}". Try WAV, MP3, M4A or OGG. (${err.message || err})`));
  }
}

async function startLive() {
  st.runId++;
  $("studio-audio").pause();
  st.item = null;
  st.committed = []; st.partial = "";
  st.stats = { chunkMs: null, rtf: null, audioSec: null };
  try {
    const engine = await ensureModel(st.model);
    const live = new LiveSession({
      engine, modelKey: st.model,
      ui: {
        onPartial: (t) => { st.partial = t; renderLive(); },
        onCommit: (raw) => { st.committed.push(raw); renderLive(); },
        onStats: (s) => { st.stats = s; renderStats(); if (s.backlogSec > 2) setNote("Running behind real time. Try Shrota-Tiny."); },
        onLevel: (lv) => { $("mic-level").style.width = `${Math.min(100, Math.round(Math.sqrt(lv) * 130))}%`; },
        onError: (e) => showError(e),
      },
    });
    st.live = live;
    $("btn-mic").classList.add("recording");
    $("mic-label").textContent = "Stop";
    setStatus("Listening", "live");
    renderLive();
    await live.start();
  } catch (err) {
    st.live = null;
    $("btn-mic").classList.remove("recording");
    $("mic-label").textContent = "Start speaking";
    const denied = err && (err.name === "NotAllowedError" || err.name === "SecurityError");
    showError(denied ? new Error("Microphone permission was denied. Allow microphone access for this site and try again.") : err);
  }
}

async function stopLive() {
  const live = st.live;
  if (!live) return;
  st.live = null;
  $("btn-mic").classList.remove("recording");
  $("mic-label").textContent = "Start speaking";
  $("mic-level").style.width = "0%";
  setStatus("Finishing", "busy");
  try { await live.stop(); } catch (e) { console.error(e); }
  setStatus("Ready", "ok");
  renderLive();
}

// ---------- init ----------

function renderSamples() {
  const list = $("sample-deck-list");
  if (!st.samples.length) { list.innerHTML = `<div class="empty-state">No example clips found</div>`; return; }
  list.innerHTML = st.samples
    .map((s) => {
      const dec = st.model === "tiny" ? (s.tiny_shuddha || s.base_shuddha || s.base_raw || "") : (s.base_shuddha || s.base_raw || "");
      return `<div class="sample-item" data-id="${esc(s.id)}" tabindex="0" role="button">
        <div class="sample-head"><span>${esc(s.title)}</span><span class="m3-chip secondary">${s.lang === "sa" ? "Sanskrit" : "Nepali"} · ${s.duration_s}s</span></div>
        <div class="sample-preview deva">${esc(dec || s.reference || "")}</div>
        ${dec && s.reference && dec !== s.reference ? `<div class="sample-preview deva" style="font-size:12px;color:#6E5E55;margin-top:2px">Ref: ${esc(s.reference)}</div>` : ""}
        <div class="sample-play"><span class="play-icon" aria-hidden="true"></span>Play and transcribe</div>
      </div>`;
    })
    .join("");
  list.querySelectorAll(".sample-item").forEach((el) => {
    const go = () => loadSample(st.samples.find((s) => s.id === el.dataset.id));
    el.addEventListener("click", go);
    el.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); go(); } });
  });
}

function initInputs() {
  $("btn-mic").addEventListener("click", () => (st.live ? stopLive() : startLive()));
  const input = $("file-input");
  input.addEventListener("change", () => { loadFile(input.files[0]); input.value = ""; });
  const dz = $("dropzone");
  ["dragenter", "dragover"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
  dz.addEventListener("drop", (e) => loadFile(e.dataTransfer.files[0]));
  document.querySelectorAll("#shuddha-selector .seg-btn").forEach((b) =>
    b.addEventListener("click", () => {
      document.querySelectorAll("#shuddha-selector .seg-btn").forEach((x) => x.classList.toggle("active", x === b));
      st.shuddha = b.dataset.shuddha === "on";
      render();
    })
  );
  const audio = $("studio-audio");
  ["play", "pause", "ended", "seeked"].forEach((ev) => audio.addEventListener(ev, () => st.item && !st.live && renderClip()));
}

export async function initStudio() {
  initInputs();
  renderModelSelector();
  renderStats();
  setStatus("Idle", "idle");
  const grab = async (path) => {
    try { return (await (await fetch(new URL(path, document.baseURI), { cache: "no-cache" })).json()).samples || []; } catch (_) { return []; }
  };
  const [curated, extra] = await Promise.all([grab("metrics.json"), grab("samples/manifest.json")]);
  const seen = new Set();
  st.samples = [...curated, ...extra]
    .filter((x) => x.audio_url && !seen.has(x.audio_url) && seen.add(x.audio_url))
    .map((x) => ({
      id: x.id,
      title: x.title,
      lang: x.lang || "ne",
      domain: x.domain,
      audio_url: x.audio_url,
      duration_s: x.duration_s,
      reference: x.reference,
      base_raw: x.base_raw || "",
      base_shuddha: x.base_shuddha || "",
      tiny_shuddha: x.tiny_shuddha || "",
      words: x.words || [],
    }));
  renderSamples();
  requestAnimationFrame(tick);
  // Warm the runtime and the default model in the background so the first click is fast.
  ensureModel(st.model).then(() => { setStatus("Ready", "ok"); setNote(""); }).catch(showError);
}
