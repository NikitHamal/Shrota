// Shrota Studio entry: tabs, data loading, clipboard.
import { loadLexicon } from "./shuddha_client.js";
import { initStudio } from "./studio.js";
import { renderBenchmarks, renderShuddha, renderDownloads, initBenchmarkControls } from "./benchmarks.js";

function initTabs() {
  const tabs = document.querySelectorAll(".tab-btn");
  const show = (id) => {
    tabs.forEach((b) => b.classList.toggle("active", b.dataset.tab === id));
    document.querySelectorAll(".tab-panel").forEach((p) => p.classList.toggle("active", p.id === id));
    history.replaceState(null, "", `#${id.replace("tab-", "")}`);
  };
  tabs.forEach((b) => b.addEventListener("click", () => show(b.dataset.tab)));
  const hash = location.hash.slice(1);
  if (hash && document.getElementById(`tab-${hash}`)) show(`tab-${hash}`);
}

function initCopy() {
  document.querySelectorAll(".copy-btn").forEach((btn) =>
    btn.addEventListener("click", async () => {
      const text = document.getElementById(btn.dataset.copy).textContent;
      try {
        await navigator.clipboard.writeText(text);
        btn.textContent = "Copied";
      } catch (_) {
        btn.textContent = "Failed";
      }
      setTimeout(() => (btn.textContent = "Copy"), 1500);
    })
  );
}

async function loadMetrics() {
  try {
    const r = await fetch(new URL("metrics.json", document.baseURI), { cache: "no-cache" });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return await r.json();
  } catch (err) {
    console.warn("metrics.json unavailable:", err.message);
    return null;
  }
}

async function boot() {
  initTabs();
  initCopy();
  const [metrics] = await Promise.all([loadMetrics(), loadLexicon()]);
  initStudio();
  renderBenchmarks(metrics, "wer");
  initBenchmarkControls(metrics);
  renderShuddha(metrics);
  renderDownloads(metrics);
}

boot();
