// Studio tab: model/Shuddha controls, example list, transcript stage.
import { applyClientShuddha } from "./shuddha_client.js";

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));

const MODEL_KEYS = {
  base: ["shrota_base_shuddha", "shrota_base"],
  tiny: ["shrota_tiny_shuddha", "shrota_tiny"],
};

export const studio = { metrics: null, items: [], current: null, model: "base", shuddha: true, timer: null };

function modelEntry(kind) {
  const models = studio.metrics?.models || {};
  for (const k of MODEL_KEYS[kind]) if (models[k]) return models[k];
  return null;
}

function itemHasModel(item, kind) {
  if (!item) return false;
  return kind === "base" ? !!(item.base_raw || item.base_shuddha) : !!(item.tiny_raw || item.tiny_shuddha);
}

function outputsFor(item, kind) {
  const raw = (kind === "base" ? item.base_raw : item.tiny_raw) || item.base_raw || "";
  const refined = (kind === "base" ? item.base_shuddha : item.tiny_shuddha) || applyClientShuddha(raw, true);
  return { raw, refined };
}

// Word-level LCS diff: returns html for `b` with words not present in `a` highlighted.
function diffHtml(a, b) {
  const x = a.split(/\s+/).filter(Boolean);
  const y = b.split(/\s+/).filter(Boolean);
  const dp = Array.from({ length: x.length + 1 }, () => new Array(y.length + 1).fill(0));
  for (let i = x.length - 1; i >= 0; i--)
    for (let j = y.length - 1; j >= 0; j--)
      dp[i][j] = x[i] === y[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
  const out = [];
  let i = 0, j = 0;
  while (j < y.length) {
    if (i < x.length && x[i] === y[j]) { out.push(esc(y[j])); i++; j++; }
    else if (i < x.length && dp[i + 1][j] >= dp[i][j + 1]) { i++; }
    else { out.push(`<span class="diff-add">${esc(y[j])}</span>`); j++; }
  }
  return out.join(" ");
}

function renderStats() {
  const m = modelEntry(studio.model);
  const row = $("stat-row");
  if (!m) { row.innerHTML = ""; return; }
  const rtf = m.cpu_1thread_rtf;
  const stats = [
    [`${Number(m.size_mb).toFixed(1)} MB`, "INT8 size"],
    [rtf ? `${Math.round(1 / rtf)}x` : "-", `Real time, RTF ${rtf ?? "-"}`],
    [m.latency_ms != null ? `${m.latency_ms} ms` : "-", "Chunk latency"],
    [`${m.chunk_ms || 160} ms`, "Chunk size"],
  ];
  row.innerHTML = stats.map(([v, l]) => `<div class="stat"><div class="stat-value">${v}</div><div class="stat-label">${l}</div></div>`).join("");
}

function renderModelSelector() {
  const wrap = $("model-selector");
  const avail = ["base", "tiny"].filter((k) => modelEntry(k));
  wrap.innerHTML = ["base", "tiny"]
    .map((k) => {
      const m = modelEntry(k);
      const label = k === "base" ? "Shrota-Base" : "Shrota-Tiny";
      const disabled = !m || !itemHasModel(studio.current, k);
      return `<button class="seg-btn${studio.model === k ? " active" : ""}" data-model="${k}" ${disabled ? "disabled" : ""}>${label}</button>`;
    })
    .join("");
  if (!avail.includes(studio.model) && avail.length) studio.model = avail[0];
  wrap.querySelectorAll(".seg-btn").forEach((b) =>
    b.addEventListener("click", () => {
      studio.model = b.dataset.model;
      renderModelSelector();
      renderStats();
      renderStage();
    })
  );
}

export function renderStage() {
  clearInterval(studio.timer);
  const item = studio.current;
  if (!item) return;
  const { raw, refined } = outputsFor(item, studio.model);
  const shown = studio.shuddha ? refined : raw;
  $("out-text").innerHTML = studio.shuddha ? diffHtml(raw, shown) : esc(raw);
  $("raw-text").textContent = raw;
  $("ref-text").textContent = item.reference || "";

  const words = item.words || [];
  $("words-block").hidden = words.length === 0;
  $("word-timestamps").innerHTML = words.map((w, i) => `<span class="word-pill" id="wp-${i}">${esc(w.word)} ${w.t0.toFixed(2)}s</span>`).join("");

  const audio = $("studio-audio");
  audio.hidden = !item.audio_url;
  if (item.audio_url && !audio.src.endsWith(item.audio_url)) audio.src = item.audio_url;
}

function syncWords() {
  const audio = $("studio-audio");
  const words = studio.current?.words || [];
  audio.ontimeupdate = () => {
    const t = audio.currentTime;
    words.forEach((w, i) => $(`wp-${i}`)?.classList.toggle("active", t >= w.t0 && t < w.t1));
  };
}

export function selectItem(item) {
  studio.current = item;
  if (!itemHasModel(item, studio.model)) studio.model = "base";
  document.querySelectorAll(".sample-item").forEach((el) => el.classList.toggle("active", el.dataset.id === item.id));
  renderModelSelector();
  renderStats();
  renderStage();
  syncWords();
}

function renderList() {
  const list = $("sample-deck-list");
  if (!studio.items.length) {
    list.innerHTML = `<div class="empty-state">No examples yet</div>`;
    return;
  }
  list.innerHTML = studio.items
    .map((s) => {
      const meta = [s.lang === "sa" ? "Sanskrit" : "Nepali", s.duration_s ? `${s.duration_s}s` : null].filter(Boolean).join(" · ");
      return `<div class="sample-item" data-id="${esc(s.id)}" tabindex="0" role="button">
        <div class="sample-head"><span>${esc(s.title)}</span><span class="m3-chip ${s.lang === "sa" ? "tertiary" : "secondary"}">${meta}</span></div>
        <div class="sample-preview deva">${esc(s.reference || "")}</div>
      </div>`;
    })
    .join("");
  list.querySelectorAll(".sample-item").forEach((el) => {
    const go = () => {
      const item = studio.items.find((s) => s.id === el.dataset.id);
      selectItem(item);
      if (item.audio_url) $("studio-audio").play().catch(() => {});
    };
    el.addEventListener("click", go);
    el.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); go(); } });
  });
}

export function initStudio(metrics) {
  studio.metrics = metrics;
  studio.items = metrics?.samples?.length ? metrics.samples : metrics?.examples || [];
  document.querySelectorAll("#shuddha-selector .seg-btn").forEach((b) =>
    b.addEventListener("click", () => {
      document.querySelectorAll("#shuddha-selector .seg-btn").forEach((x) => x.classList.toggle("active", x === b));
      studio.shuddha = b.dataset.shuddha === "on";
      renderStage();
    })
  );
  renderList();
  if (studio.items.length) selectItem(studio.items[0]);
  else { renderModelSelector(); renderStats(); }
}
