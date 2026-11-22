// Loads onnxruntime-web (WASM, single thread: GitHub Pages cannot send COOP/COEP headers) and the Shrota engine.
import { ShrotaEngine } from "./engine.js";

const ORT_VERSION = "1.20.1";
const ORT_BASE = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;

const url = (p) => new URL(p, document.baseURI).toString();

async function fetchOk(path) {
  const r = await fetch(url(path));
  if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
  return r;
}

let enginePromise = null;

export function getEngine() {
  if (!enginePromise) {
    enginePromise = (async () => {
      const ort = await import(/* @vite-ignore */ `${ORT_BASE}ort.wasm.min.mjs`);
      ort.env.wasm.wasmPaths = ORT_BASE;
      ort.env.wasm.numThreads = 1;
      ort.env.wasm.proxy = false;
      const manifest = await (await fetchOk("models/manifest.json")).json();
      return new ShrotaEngine({
        ort,
        manifest,
        loadBytes: async (p) => (await fetchOk(p)).arrayBuffer(),
        loadJson: async (p) => (await fetchOk(p)).json(),
      });
    })().catch((e) => {
      enginePromise = null;
      throw e;
    });
  }
  return enginePromise;
}
