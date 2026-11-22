// Client-side Shuddha fallback: lexicon + postposition rules (mirrors shrota/streaming.py rules).
import { normalizeText } from "./norm.js";

export const lexicon = { pairs: {} };

export async function loadLexicon(url = "models/shuddha_lexicon.json") {
  try {
    const r = await fetch(url);
    if (r.ok) {
      const d = await r.json();
      lexicon.pairs = d.pairs || d || {};
    }
  } catch (_) { /* optional asset */ }
}

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
    const lexFixed = lexicon.pairs[clean] || NEPALI_SHUDDHA_MAP[clean] || clean;
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
