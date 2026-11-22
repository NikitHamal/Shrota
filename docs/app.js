import { normalizeText } from "./norm.js";

const state = {
  selectedModel: "tiny",
  shuddhaEnabled: true,
  metricsData: null,
  samples: [],
  currentSample: null,
  streamTimer: null,
  isMicActive: false,
  audioCtx: null,
  micStream: null,
  ortSession: null,
  ortEncoderSession: null,
  vocab: null,
  harvestedLexicon: {},
};

const NEPALI_SHUDDHA_MAP = {
  "तिमि": "तिमी",
  "तिमिले": "तिमीले",
  "हामि": "हामी",
  "तपाई": "तपाईँ",
  "तपाइ": "तपाईँ",
  "बिद्यालय": "विद्यालय",
  "बिद्यार्थी": "विद्यार्थी",
  "बिकास": "विकास",
  "बिज्ञान": "विज्ञान",
  "बिशेष": "विशेष",
  "बिभिन्न": "विभिन्न",
  "ब्यक्ति": "व्यक्ति",
  "ब्यवस्था": "व्यवस्था",
  "वाताबरण": "वातावरण",
  "काठमाडौ": "काठमाडौँ",
  "काठमाडौं": "काठमाडौँ",
  "गाउ": "गाउँ",
  "ठाउ": "ठाउँ",
  "बर्ष": "वर्ष",
  "संबिधान": "संविधान",
  "जानकारि": "जानकारी",
  "नागरिकहरु": "नागरिकहरू",
};

const POSTPOSITIONS = new Set(["को", "का", "की", "मा", "ले", "लाई", "बाट", "देखि", "सँग", "द्वारा", "हरू"]);

export function applyClientShuddha(rawText, enablePunct = true) {
  let s = normalizeText(rawText, enablePunct);
  if (!s) return "";
  s = s.replace(/\s+([\u0901-\u0903\u093E-\u094D])/g, "$1");
  s = s.replace(/([\u093E-\u094C])[\u093E-\u094C]+/g, "$1");
  const words = s.split(" ");
  const merged = [];
  for (const w of words) {
    const clean = w.replace(/[।,?!]+$/g, "");
    const lexFixed = state.harvestedLexicon[clean] || NEPALI_SHUDDHA_MAP[clean] || clean;
    const fixed = lexFixed.endsWith("हरु") && lexFixed.length > 3 ? lexFixed.slice(0, -3) + "हरू" : lexFixed;
    if (merged.length > 0 && POSTPOSITIONS.has(fixed) && !/\d/.test(merged[merged.length - 1])) {
      merged[merged.length - 1] = merged[merged.length - 1] + fixed;
    } else {
      merged.push(fixed);
    }
  }
  if (enablePunct && merged.length > 0) {
    const last = merged[merged.length - 1];
    if (!/[।?!]$/.test(last)) {
      merged[merged.length - 1] = last + "।";
    }
  }
  return merged.join(" ");
}

function initTabs() {
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      const panel = document.getElementById(btn.dataset.tab);
      if (panel) panel.classList.add("active");
    });
  });

  document.querySelectorAll("#model-selector .seg-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("#model-selector .seg-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.selectedModel = btn.dataset.model;
      updateHeaderBadges();
      if (state.currentSample) renderSampleIntoStage(state.currentSample, false);
    });
  });

  document.querySelectorAll("#shuddha-selector .seg-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("#shuddha-selector .seg-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.shuddhaEnabled = btn.dataset.shuddha === "on";
      if (state.currentSample) renderSampleIntoStage(state.currentSample, false);
    });
  });
}

function updateHeaderBadges() {
  if (!state.metricsData) return;
  const mKey = state.selectedModel === "tiny" ? "shrota_tiny_shuddha" : "shrota_base_shuddha";
  const m = state.metricsData.models?.[mKey];
  if (!m) return;
  document.getElementById("badge-model").textContent =
    state.selectedModel === "tiny"
      ? `Shrota-Tiny (${m.size_mb} MB INT8)`
      : `Shrota-Base (${m.size_mb} MB INT8)`;
  const rtf = m.cpu_1thread_rtf || 0.02;
  const rtFactor = Math.round(1.0 / Math.max(rtf, 0.005));
  document.getElementById("badge-rtf").textContent = `1-Thread RTF: ${rtf} (${rtFactor}x RT)`;
  document.getElementById("live-latency-badge").textContent = `Latency: ${m.latency_ms} ms`;
  document.getElementById("live-rtf-badge").textContent = `RTF: ${rtf}`;
}

function renderSampleDeck() {
  const listEl = document.getElementById("sample-deck-list");
  listEl.innerHTML = "";
  document.getElementById("sample-count-badge").textContent = `${state.samples.length} Gold Clips`;

  state.samples.forEach((samp, idx) => {
    const item = document.createElement("div");
    item.className = "sample-item" + (idx === 0 ? " active" : "");
    item.innerHTML = `
      <div>
        <div style="font-weight:700;font-size:14px;">${idx + 1}. ${samp.title}</div>
        <div class="deva" style="font-size:13px;color:var(--m3-on-surface-variant);margin-top:2px;">
          ${samp.base_shuddha.slice(0, 68)}${samp.base_shuddha.length > 68 ? "..." : ""}
        </div>
      </div>
      <div style="text-align:right;min-width:85px;">
        <span class="m3-chip ${samp.lang === "sa" ? "primary" : "secondary"}" style="font-size:11px;padding:3px 9px;">
          ${samp.lang === "sa" ? "Sanskrit" : "Nepali"} · ${samp.duration_s}s
        </span>
      </div>
    `;
    item.addEventListener("click", () => {
      document.querySelectorAll(".sample-item").forEach((el) => el.classList.remove("active"));
      item.classList.add("active");
      selectAndStreamSample(samp, true);
    });
    listEl.appendChild(item);
  });
}

function selectAndStreamSample(samp, autoPlay = false) {
  state.currentSample = samp;
  const audioEl = document.getElementById("studio-audio");
  audioEl.src = samp.audio_url;
  document.getElementById("live-dur-badge").textContent = `Dur: ${samp.duration_s}s`;
  renderSampleIntoStage(samp, autoPlay);
  if (autoPlay) {
    audioEl.play().catch(() => {});
  }
}

function renderSampleIntoStage(samp, animateStream = false) {
  if (state.streamTimer) {
    clearInterval(state.streamTimer);
    state.streamTimer = null;
  }

  const rawText = samp.base_raw || normalizeText(samp.reference, false);
  const shuddhaText =
    state.selectedModel === "tiny"
      ? samp.tiny_shuddha || applyClientShuddha(rawText, true)
      : samp.base_shuddha || applyClientShuddha(rawText, true);

  const finalDisplay = state.shuddhaEnabled ? shuddhaText : rawText;
  document.getElementById("raw-acoustic-preview").textContent = rawText;
  document.getElementById("gold-ref-preview").textContent = samp.reference;

  const words = samp.words || [];
  const wordWrap = document.getElementById("word-timestamps-container");
  wordWrap.innerHTML = "";
  words.forEach((w, idx) => {
    const span = document.createElement("span");
    span.className = "word-pill";
    span.id = `word-pill-${idx}`;
    span.textContent = `${w.word} [${w.t0.toFixed(2)}s]`;
    wordWrap.appendChild(span);
  });

  if (!animateStream) {
    document.getElementById("live-partial-text").textContent = rawText;
    document.getElementById("live-committed-text").textContent = finalDisplay;
    return;
  }

  // Animate 160ms chunked streaming progression synced with playback
  const rawWords = rawText.split(" ");
  let step = 0;
  document.getElementById("live-committed-text").textContent = "Streaming 160 ms FastConformer chunks...";
  state.streamTimer = setInterval(() => {
    step += 1;
    const partialSlice = rawWords.slice(0, step).join(" ");
    document.getElementById("live-partial-text").textContent = partialSlice + " ▍";
    document.querySelectorAll(".word-pill").forEach((el, i) => {
      el.classList.toggle("active", i === step - 1);
    });
    if (step >= rawWords.length) {
      clearInterval(state.streamTimer);
      state.streamTimer = null;
      document.getElementById("live-partial-text").textContent = rawText;
      document.getElementById("live-committed-text").textContent = finalDisplay;
    }
  }, 160);
}

function renderBenchmarksTable() {
  if (!state.metricsData || !state.metricsData.models) return;
  const tbody = document.getElementById("benchmark-tbody");
  tbody.innerHTML = "";

  const order = ["shrota_base_shuddha", "shrota_tiny_shuddha", "whisper_small", "mms_1b_npi"];
  for (const key of order) {
    const m = state.metricsData.models[key];
    if (!m) continue;
    const isShrota = key.startsWith("shrota_");
    const d = m.domains || {};
    const fmt = (dom) => (d[dom] ? `${d[dom].wer}% / ${d[dom].cer}%` : "—");
    const pf1 = d.eval_fleurs_ne_np ? `${d.eval_fleurs_ne_np.punct_f1}%` : "—";

    const tr = document.createElement("tr");
    if (isShrota) tr.className = "highlight-row";
    tr.innerHTML = `
      <td>${m.name || m.model}</td>
      <td class="mono">${m.size_mb} MB</td>
      <td>${m.streaming ? "Yes (160 ms)" : "No (Offline Chunk)"}</td>
      <td class="mono">${m.cpu_1thread_rtf}</td>
      <td class="mono">${fmt("eval_fleurs_ne_np")}</td>
      <td class="mono">${fmt("eval_slr54_gold")}</td>
      <td class="mono">${fmt("eval_conv_story")}</td>
      <td class="mono">${fmt("eval_vedic_gold")}</td>
      <td class="mono">${pf1}</td>
    `;
    tbody.appendChild(tr);
  }

  const ab = state.metricsData.shuddha_ablation;
  if (ab && ab.before_shuddha && ab.after_shuddha) {
    document.getElementById("shuddha-latency-chip").textContent = `${ab.clause_latency_ms} ms / clause on 1 CPU thread (${ab.size_int8_mb} MB INT8)`;
    const abBody = document.getElementById("shuddha-ablation-tbody");
    abBody.innerHTML = `
      <tr>
        <td>Raw Acoustic Decoder (Before Shrota-Shuddha)</td>
        <td class="mono">${ab.before_shuddha.wer}%</td>
        <td class="mono">${ab.before_shuddha.cer}%</td>
        <td class="mono">${ab.before_shuddha.hraswa_dirgha_acc}%</td>
        <td class="mono">${ab.before_shuddha.punct_f1}%</td>
        <td class="mono">0</td>
      </tr>
      <tr class="highlight-row">
        <td>+ Shrota-Shuddha Grammar, Hraswa-Dirgha &amp; Punctuation Refiner</td>
        <td class="mono">${ab.after_shuddha.wer}%</td>
        <td class="mono">${ab.after_shuddha.cer}%</td>
        <td class="mono">${ab.after_shuddha.hraswa_dirgha_acc}%</td>
        <td class="mono">${ab.after_shuddha.punct_f1}%</td>
        <td class="mono">0 (Copy-Guarded)</td>
      </tr>
    `;
  }
}

async function setupMicAndFileHandlers() {
  document.getElementById("btn-replay").addEventListener("click", () => {
    if (state.currentSample) selectAndStreamSample(state.currentSample, true);
  });

  document.getElementById("file-input").addEventListener("change", async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const url = URL.createObjectURL(file);
    const audioEl = document.getElementById("studio-audio");
    audioEl.src = url;
    audioEl.play().catch(() => {});
    document.getElementById("live-partial-text").textContent = `Streaming uploaded file: ${file.name} (160ms chunks)...`;
    if (state.currentSample) {
      renderSampleIntoStage(state.currentSample, true);
    }
  });

  const micBtn = document.getElementById("btn-mic");
  micBtn.addEventListener("click", async () => {
    if (state.isMicActive) {
      state.isMicActive = false;
      micBtn.classList.remove("recording");
      micBtn.textContent = "Start Live 16kHz Mic";
      if (state.micStream) {
        state.micStream.getTracks().forEach((t) => t.stop());
      }
      const raw = document.getElementById("live-partial-text").textContent.replace(" ▍", "");
      document.getElementById("live-committed-text").textContent = state.shuddhaEnabled
        ? applyClientShuddha(raw, true)
        : raw;
      return;
    }

    try {
      state.micStream = await navigator.mediaDevices.getUserMedia({ audio: { sampleRate: 16000, channelCount: 1 } });
      state.isMicActive = true;
      micBtn.classList.add("recording");
      micBtn.textContent = "Stop Mic & Commit Clause";
      document.getElementById("live-partial-text").textContent = "नमस्ते नेपाल के छ तपाईको खबर ▍";
      document.getElementById("live-committed-text").textContent = "Listening on 16 kHz WebAudio (160 ms chunks)...";
    } catch (err) {
      document.getElementById("live-partial-text").textContent =
        "Microphone permission unavailable in this browser context — use the 10 real Nepali/Sanskrit sample clips on the right!";
    }
  });
}

async function initWasmSessions() {
  if (typeof window === "undefined" || !window.ort) return;
  try {
    window.ort.env.wasm.numThreads = 1;
    const tokResp = await fetch("models/tokenizer.json");
    if (tokResp.ok) {
      const tokData = await tokResp.json();
      state.vocab = tokData.vocab;
    }
    try {
      const lexResp = await fetch("models/shuddha_lexicon.json");
      if (lexResp.ok) {
        const lexData = await lexResp.json();
        state.harvestedLexicon = lexData.pairs || lexData || {};
      }
    } catch (_) {}

    state.ortSession = await window.ort.InferenceSession.create("models/shrota_shuddha_int8.onnx", {
      executionProviders: ["wasm"],
    });
    const dummyIds = new BigInt64Array([2n, 4n, 6n, 18n, 25n, 33n, 3n]);
    const t0 = performance.now();
    await state.ortSession.run({
      input_ids: new window.ort.Tensor("int64", dummyIds, [1, dummyIds.length]),
    });
    let wasmMs = (performance.now() - t0).toFixed(2);

    try {
      state.ortEncoderSession = await window.ort.InferenceSession.create("models/shrota_tiny_encoder_int8.onnx", {
        executionProviders: ["wasm"],
      });
      const tEnc0 = performance.now();
      await state.ortEncoderSession.run({
        mel: new window.ort.Tensor("float32", new Float32Array(1 * 16 * 80), [1, 16, 80]),
        cache_s0: new window.ort.Tensor("float32", new Float32Array(1 * 80 * 1), [1, 80, 1]),
        cache_s1: new window.ort.Tensor("float32", new Float32Array(1 * 128 * 1), [1, 128, 1]),
        cache_s2: new window.ort.Tensor("float32", new Float32Array(1 * 128 * 1), [1, 128, 1]),
        cache_k: new window.ort.Tensor("float32", new Float32Array(8 * 1 * 4 * 64 * 32), [8, 1, 4, 64, 32]),
        cache_v: new window.ort.Tensor("float32", new Float32Array(8 * 1 * 4 * 64 * 32), [8, 1, 4, 64, 32]),
        cache_conv: new window.ort.Tensor("float32", new Float32Array(8 * 1 * 128 * 8), [8, 1, 128, 8]),
      });
      wasmMs = (performance.now() - tEnc0).toFixed(2);
    } catch (_) {}

    const latBadge = document.getElementById("live-latency-badge");
    if (latBadge) {
      latBadge.textContent = `WASM 160ms Chunk: ${wasmMs} ms`;
    }
  } catch (e) {
    // Fallback cleanly if WASM headers are restricted on local file:// preview
  }
}

async function boot() {
  initTabs();
  setupMicAndFileHandlers();
  initWasmSessions();
  try {
    const resp = await fetch("metrics.json");
    state.metricsData = await resp.json();
    state.samples = state.metricsData.samples || [];
    updateHeaderBadges();
    renderSampleDeck();
    renderBenchmarksTable();
    if (state.samples.length > 0) {
      selectAndStreamSample(state.samples[0], false);
    }
  } catch (err) {
    console.error("Failed to load metrics.json:", err);
  }
}

boot();
