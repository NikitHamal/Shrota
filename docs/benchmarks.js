// Benchmarks tab + Integrate tab data rendering.
const $ = (id) => document.getElementById(id);

const DOMAINS = [
  ["eval_fleurs_ne_np", "FLEURS"],
  ["eval_slr54_gold", "SLR54"],
  ["eval_conv_story", "Conversation"],
  ["eval_vedic_gold", "Vedic"],
];
const ORDER = ["shrota_base_shuddha", "shrota_base", "shrota_tiny_shuddha", "shrota_tiny", "whisper_small", "mms_1b_npi"];
const REPO = "https://github.com/NikitHamal/Shrota/releases/download/v1-ckpt";

const sizeLabel = (mb) => (mb >= 100 ? `${Math.round(mb)} MB` : `${Number(mb).toFixed(1)} MB`);

export function renderBenchmarks(metrics, metric = "wer") {
  const models = metrics?.models || {};
  const keys = ORDER.filter((k) => models[k]);
  $("benchmark-thead").innerHTML = `<tr><th>Model</th><th>Size</th><th>RTF</th>${DOMAINS.map(([, l]) => `<th>${l}</th>`).join("")}</tr>`;

  const best = {};
  for (const [d] of DOMAINS) {
    const vals = keys.map((k) => models[k].domains?.[d]?.[metric]).filter((v) => v != null);
    best[d] = vals.length ? Math.min(...vals) : null;
  }
  const other = metric === "wer" ? "cer" : "wer";
  $("benchmark-tbody").innerHTML = keys
    .map((k) => {
      const m = models[k];
      const ours = k.startsWith("shrota_");
      const cells = DOMAINS.map(([d]) => {
        const v = m.domains?.[d];
        if (!v) return `<td class="pending">-</td>`;
        const isBest = v[metric] === best[d];
        return `<td class="num${isBest ? " best" : ""}">${v[metric]}%<span class="sub">${other.toUpperCase()} ${v[other]}%</span></td>`;
      }).join("");
      return `<tr class="${ours ? "ours" : ""}"><td>${m.name || m.model}</td><td class="num">${sizeLabel(m.size_mb)}</td><td class="num">${m.cpu_1thread_rtf ?? "-"}</td>${cells}</tr>`;
    })
    .join("");
}

export function renderShuddha(metrics) {
  const ab = metrics?.shuddha || metrics?.shuddha_ablation;
  const body = $("shuddha-ablation-tbody");
  if (!ab?.before_shuddha || !ab?.after_shuddha) {
    body.innerHTML = `<tr><td class="pending" colspan="5">No data</td></tr>`;
    $("shuddha-chip").hidden = true;
    return;
  }
  $("shuddha-chip").textContent = `${ab.clause_latency_ms} ms per clause · ${ab.size_int8_mb} MB`;
  const row = (label, r, cls = "") =>
    `<tr class="${cls}"><td>${label}</td><td class="num">${r.wer}%</td><td class="num">${r.cer}%</td><td class="num">${r.hraswa_dirgha_acc}%</td><td class="num">${r.punct_f1}%</td></tr>`;
  body.innerHTML = row("Corrupted FLEURS text", ab.before_shuddha) + row("After Shuddha", ab.after_shuddha, "ours");
}

export function renderDownloads(metrics) {
  const models = metrics?.models || {};
  const sizeOf = (...ks) => {
    for (const k of ks) if (models[k]) return models[k].size_mb;
    return null;
  };
  const shud = metrics?.shuddha || metrics?.shuddha_ablation;
  const files = [
    ["shrota_shuddha_int8.onnx", shud?.size_int8_mb, "models/shrota_shuddha_int8.onnx"],
    ["tokenizer.json", null, "models/tokenizer.json"],
    ["shrota_base_encoder_int8.onnx", sizeOf("shrota_base", "shrota_base_shuddha"), `${REPO}/shrota_base_encoder_int8.onnx`],
    ["shrota_tiny_encoder_int8.onnx", sizeOf("shrota_tiny", "shrota_tiny_shuddha"), `${REPO}/shrota_tiny_encoder_int8.onnx`],
  ];
  $("download-list").innerHTML = files
    .map(([n, mb, href]) => `<a class="download-item" href="${href}" ${href.startsWith("http") ? 'target="_blank" rel="noopener"' : "download"}><span class="download-name mono">${n}</span>${mb ? `<span class="m3-chip">${mb} MB</span>` : ""}</a>`)
    .join("");
}

export function initBenchmarkControls(metrics) {
  document.querySelectorAll("#metric-selector .seg-btn").forEach((b) =>
    b.addEventListener("click", () => {
      document.querySelectorAll("#metric-selector .seg-btn").forEach((x) => x.classList.toggle("active", x === b));
      renderBenchmarks(metrics, b.dataset.metric);
    })
  );
}
