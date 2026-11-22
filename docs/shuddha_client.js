// Client-side Shuddha v2 refiner: lexicon + homophone resolution + guarded postposition & punctuation rules.
import { normalizeText } from "./norm.js";

export const lexicon = { pairs: {}, vocabFreq: {} };

export async function loadLexicon(url = "models/shuddha_lexicon.json") {
  try {
    const r = await fetch(url);
    if (r.ok) {
      const d = await r.json();
      lexicon.pairs = d.pairs || d || {};
      lexicon.vocabFreq = d.vocab_freq || {};
    }
  } catch (_) { /* optional asset */ }
}

const NEPALI_SHUDDHA_MAP = {
  "तिमि": "तिमी",
  "तिमिले": "तिमीले",
  "तिमिलाई": "तिमीलाई",
  "हामि": "हामी",
  "हामिले": "हामीले",
  "हामिलाई": "हामीलाई",
  "तपाई": "तपाईँ",
  "तपाइ": "तपाईँ",
  "तपाइँ": "तपाईँ",
  "बिद्यालय": "विद्यालय",
  "बिद्यार्थी": "विद्यार्थी",
  "बिश्वबिद्यालय": "विश्वविद्यालय",
  "बिकास": "विकास",
  "बिज्ञान": "विज्ञान",
  "बिशेष": "विशेष",
  "बिभिन्न": "विभिन्न",
  "ब्यक्ति": "व्यक्ति",
  "ब्यवस्था": "व्यवस्था",
  "ब्यवसाय": "व्यवसाय",
  "बिचार": "विचार",
  "बिषय": "विषय",
  "बिभाग": "विभाग",
  "बिरुद्ध": "विरुद्ध",
  "बिदेश": "विदेश",
  "बित्तीय": "वित्तीय",
  "वाताबरण": "वातावरण",
  "बाताबरण": "वातावरण",
  "काठमाडौ": "काठमाडौँ",
  "काठमाडौं": "काठमाडौँ",
  "गाउ": "गाउँ",
  "गाउमा": "गाउँमा",
  "ठाउ": "ठाउँ",
  "ठाउमा": "ठाउँमा",
  "बर्ष": "वर्ष",
  "बार्षिक": "वार्षिक",
  "संबिधान": "संविधान",
  "संबैधानिक": "संवैधानिक",
  "अधिकारि": "अधिकारी",
  "कर्मचारि": "कर्मचारी",
  "जानकारि": "जानकारी",
  "नेपालि": "नेपाली",
  "नागरिकहरु": "नागरिकहरू",
  "मानिसहरु": "मानिसहरू",
  "साथीहरु": "साथीहरू",
  "कुराहरु": "कुराहरू",
  "समस्याहरु": "समस्याहरू",
  "सिक्षा": "शिक्षा",
  "सान्ति": "शान्ति",
  "सासन": "शासन",
  "प्रसासन": "प्रशासन",
};

const HOMOPHONE_SWAPS = [
  ["ि", "ी"], ["ी", "ि"], ["ु", "ू"], ["ू", "ु"],
  ["ब", "व"], ["व", "ब"], ["स", "श"], ["स", "ष"],
  ["श", "स"], ["ष", "स"], ["न", "ण"], ["ण", "न"],
  ["ं", "ँ"], ["ँ", "ं"], ["इ", "ई"], ["ई", "इ"],
  ["उ", "ऊ"], ["ऊ", "उ"],
];

const POSTPOSITIONS = new Set(["को", "का", "की", "मा", "ले", "लाई", "बाट", "हरू"]);
const QUESTION_WORDS = new Set(["के", "कति", "कहाँ", "कहिले", "किन", "कसरी", "को", "कुन", "कसले", "कसलाई", "कस्तो", "होला"]);
const CLAUSE_END_VERBS = new Set([
  "छ", "छन्", "थियो", "थिए", "हो", "हुन्", "हुन्छ", "भयो", "भए", "गर्छन्", "गर्छ",
  "गरे", "गर्यो", "बताए", "भने", "पर्छ", "सकिन्छ", "छैन", "छैनन्", "थिएन", "गर्नुभयो",
  "बताउनुभयो", "भन्नुभयो", "गरेको", "रहेको", "गर्दै", "हुँदै",
]);

function resolveHomophone(w) {
  if (NEPALI_SHUDDHA_MAP[w]) return NEPALI_SHUDDHA_MAP[w];
  const vf = lexicon.vocabFreq || {};
  if (lexicon.pairs[w] && (vf[w] || 0) === 0) return lexicon.pairs[w];
  if (!Object.keys(vf).length || (vf[w] || 0) >= 1) return w;
  let best = w;
  let bestFreq = 25;
  for (let i = 0; i < w.length; i++) {
    const ch = w[i];
    for (const [src, dst] of HOMOPHONE_SWAPS) {
      if (ch === src) {
        const cand = w.slice(0, i) + dst + w.slice(i + 1);
        const cf = vf[cand] || 0;
        if (cf > bestFreq) {
          bestFreq = cf;
          best = cand;
        }
      }
    }
  }
  return best;
}

export function applyClientShuddha(rawText, enablePunct = true) {
  let s = normalizeText(rawText, enablePunct);
  if (!s) return "";
  const hadQuestion = s.includes("?");
  s = s.replace(/\s+([\u0901-\u0903\u093E-\u094D])/g, "$1");
  s = s.replace(/([\u093E-\u094C])[\u093E-\u094C]+/g, "$1");
  const words = s.split(" ").filter(Boolean);
  const merged = [];
  const vf = lexicon.vocabFreq || {};

  for (const w of words) {
    const clean = w.replace(/[।,?!]+$/g, "");
    const trail = w.slice(clean.length);
    const lexFixed = resolveHomophone(clean);
    const fixed = lexFixed.endsWith("हरु") && lexFixed.length > 3 ? lexFixed.slice(0, -3) + "हरू" : lexFixed;
    const prev = merged.length > 0 ? merged[merged.length - 1] : "";
    if (
      enablePunct &&
      prev &&
      POSTPOSITIONS.has(fixed) &&
      !/[।,?!]$/.test(prev) &&
      !/[0-9\u0966-\u096F]/.test(prev) &&
      (fixed === "हरू" || (vf[prev + fixed] || 0) >= Math.max(5, (vf[prev] || 0) * 2))
    ) {
      merged[merged.length - 1] = prev + fixed + trail;
    } else {
      merged.push(fixed + trail);
    }
  }
  if (enablePunct && merged.length > 0) {
    const last = merged[merged.length - 1];
    if (!/[।?!]$/.test(last)) {
      const lastClean = last.replace(/[।,?!]+$/g, "");
      const hasQ = hadQuestion || merged.some((w) => QUESTION_WORDS.has(w.replace(/[।,?!]+$/g, "")));
      if (hasQ && (CLAUSE_END_VERBS.has(lastClean) || QUESTION_WORDS.has(lastClean))) {
        merged[merged.length - 1] = last + "?";
      } else if (merged.length >= 2 || CLAUSE_END_VERBS.has(lastClean)) {
        merged[merged.length - 1] = last + "।";
      }
    }
  }
  return merged.join(" ");
}
