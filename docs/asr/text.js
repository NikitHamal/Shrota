// Token -> text decoding, Devanagari v2 cleanup, and 4-gram Nepali Prefix Beam Search + Lexicon Snapping.
// 1:1 port of shrota/tokenizer.py, shrota/streaming.py, and shrota/lm_beam.py.
export const WORD_BOUNDARY = "\u2581";
const SPECIAL = new Set(["<blk>", "<unk>", "<bos>", "<eos>", "<ne>", "<sa>", "<punct>", "<nopunct>"]);
const KEEP_SINGLE = new Set([
  "र", "न", "छ", "हो", "की", "कि", "यो", "ती", "सो", "के", "को", "मा", "ले", "वा", "ऊ", "यी", "त", "नि", "पो", "ल", "खै", "नै", "तै", "उ", "म",
]);
const REDUPLICATIVE = new Set([
  "तथा", "सानो", "ठूलो", "धेरै", "थोरै", "आफ्नो", "घर", "गाउँ", "ठाउँ", "दिन", "रात", "एक", "दुई", "तीन", "चार", "पाँच", "सय", "हजार", "लाख", "करोड", "अलग", "फरक", "राम्रो", "नयाँ", "छिटो", "ढिलो", "बिस्तारै", "साथ", "पछि", "अघि", "माथि", "तल", "भित्र", "बाहिर", "वरिपरि", "कहिले", "कहाँ", "के", "को", "कुन", "कसरी", "किन",
]);
const BOUND_SUFFIXES = new Set([
  "हरू", "हरु", "हरूको", "हरूलाई", "हरूमा", "हरूले", "हरूबाट", "हरूमध्ये", "हरूसँग",
  "को", "का", "की", "मा", "ले", "लाई", "बाट", "मध्ये",
]);
const PLURAL_VERBS = new Set([
  "छन", "हुन", "गर्छन", "गर्दछन", "सक्छन", "सक्दछन", "हुन्छन", "थिएनन", "छैनन", "पार्छन",
  "हिँड्छन", "जान्छन", "आउँछन", "भन्छन", "दिन्छन", "लिन्छन", "मान्छन", "देखिन्छन", "पाइन्छन",
  "गरिन्छन", "भनिन्छन", "बस्छन", "राख्छन", "खान्छन", "पढ्छन", "लेख्छन",
]);

const LOG_ZERO = -1e9;

export function isSpecial(tok) {
  return !tok || SPECIAL.has(tok) || tok.startsWith("<extra_");
}

export function decodeTokens(ids, vocab) {
  let raw = "";
  for (const id of ids) {
    const t = vocab[id];
    if (isSpecial(t)) continue;
    raw += t;
  }
  return raw.split(WORD_BOUNDARY).join(" ").trim().replace(/\s+([।,?!])/g, "$1").replace(/\s+/g, " ").trim();
}

function logAdd(a, b) {
  if (a <= LOG_ZERO) return b;
  if (b <= LOG_ZERO) return a;
  return a > b ? a + Math.log1p(Math.exp(b - a)) : b + Math.log1p(Math.exp(a - b));
}

function scoreLmToken(lm, p3, p2, p1, c) {
  if (!lm) return 0;
  const k4 = `${p3},${p2},${p1},${c}`;
  if (lm.fourgrams && lm.fourgrams[k4] !== undefined) return lm.fourgrams[k4];
  const k3 = `${p2},${p1},${c}`;
  if (lm.trigrams && lm.trigrams[k3] !== undefined) return lm.trigrams[k3] - 0.15;
  const k2 = `${p1},${c}`;
  if (lm.bigrams && lm.bigrams[k2] !== undefined) return lm.bigrams[k2] - 0.35;
  const u = lm.unigrams && c >= 0 && c < lm.unigrams.length ? lm.unigrams[c] : -9.5;
  return u - 0.65;
}

/** Fast CTC Prefix Beam Search with 4-gram Nepali Subword Shallow Fusion (< 1 ms in JS). */
export function ctcPrefixBeamSearch(frameLogProbs, lm, { blankId = 0, bosId = 2, beamSize = 6, topK = 6, alpha = 0.25, beta = 0.08 } = {}) {
  if (!frameLogProbs || !frameLogProbs.length) return [];
  let beams = new Map([["", { ids: [], pb: 0.0, pnb: LOG_ZERO, lmScore: 0.0, frame: -1, frames: [] }]]);

  for (let t = 0; t < frameLogProbs.length; t++) {
    const lp = frameLogProbs[t];
    const V = lp.length;
    const lpBlank = lp[blankId];

    if (lpBlank > -0.04) {
      for (const b of beams.values()) {
        b.pb = logAdd(b.pb, b.pnb) + lpBlank;
        b.pnb = LOG_ZERO;
      }
      continue;
    }

    // Select topK non-special token indices for frame t
    const topIdx = [];
    for (let v = 8; v < V; v++) {
      const val = lp[v];
      if (val < -8.0) continue;
      if (topIdx.length < topK) {
        topIdx.push(v);
        topIdx.sort((a, b) => lp[b] - lp[a]);
      } else if (val > lp[topIdx[topK - 1]]) {
        topIdx[topK - 1] = v;
        topIdx.sort((a, b) => lp[b] - lp[a]);
      }
    }

    const nextBeams = new Map();
    const addNext = (key, ids, frames, addPb, addPnb, lmScore) => {
      const ex = nextBeams.get(key);
      if (ex) {
        ex.pb = logAdd(ex.pb, addPb);
        ex.pnb = logAdd(ex.pnb, addPnb);
      } else {
        nextBeams.set(key, { ids, frames, pb: addPb, pnb: addPnb, lmScore });
      }
    };

    for (const [key, b] of beams.entries()) {
      const pTot = logAdd(b.pb, b.pnb);
      addNext(key, b.ids, b.frames, pTot + lpBlank, LOG_ZERO, b.lmScore);

      const len = b.ids.length;
      const lastTok = len > 0 ? b.ids[len - 1] : -1;
      const p1 = len >= 1 ? b.ids[len - 1] : bosId;
      const p2 = len >= 2 ? b.ids[len - 2] : bosId;
      const p3 = len >= 3 ? b.ids[len - 3] : bosId;

      for (let k = 0; k < topIdx.length; k++) {
        const c = topIdx[k];
        const lpC = lp[c];
        if (c === lastTok) {
          if (b.pnb > LOG_ZERO) addNext(key, b.ids, b.frames, LOG_ZERO, b.pnb + lpC, b.lmScore);
          if (b.pb > LOG_ZERO) {
            const newIds = b.ids.concat(c);
            const newFrames = b.frames.concat(t);
            const newKey = key ? `${key},${c}` : `${c}`;
            const dLm = lm ? alpha * (scoreLmToken(lm, p3, p2, p1, c) + 4.0) + beta : 0.0;
            addNext(newKey, newIds, newFrames, LOG_ZERO, b.pb + lpC, b.lmScore + dLm);
          }
        } else {
          const newIds = b.ids.concat(c);
          const newFrames = b.frames.concat(t);
          const newKey = key ? `${key},${c}` : `${c}`;
          const dLm = lm ? alpha * (scoreLmToken(lm, p3, p2, p1, c) + 4.0) + beta : 0.0;
          addNext(newKey, newIds, newFrames, LOG_ZERO, pTot + lpC, b.lmScore + dLm);
        }
      }
    }

    if (nextBeams.size > beamSize) {
      const sorted = Array.from(nextBeams.entries()).sort(
        (a, b) => (logAdd(b[1].pb, b[1].pnb) + b[1].lmScore) - (logAdd(a[1].pb, a[1].pnb) + a[1].lmScore)
      );
      beams = new Map(sorted.slice(0, beamSize));
    } else {
      beams = nextBeams;
    }
  }

  let best = null;
  let bestScore = -Infinity;
  for (const b of beams.values()) {
    const sc = logAdd(b.pb, b.pnb) + b.lmScore;
    if (sc > bestScore) {
      bestScore = sc;
      best = b;
    }
  }
  if (!best) return [];
  return best.ids.map((id, idx) => ({ id, frame: best.frames[idx] }));
}

function charEditDist(a, b, maxDist = 2) {
  if (a === b) return 0;
  const la = a.length, lb = b.length;
  if (Math.abs(la - lb) > maxDist) return maxDist + 1;
  const dp = Array.from({ length: lb + 1 }, (_, i) => i);
  for (let i = 1; i <= la; i++) {
    let prev = dp[0];
    dp[0] = i;
    let rowMin = dp[0];
    const ai = a[i - 1];
    for (let j = 1; j <= lb; j++) {
      const tmp = dp[j];
      const cost = ai === b[j - 1] ? 0 : 1;
      const val = Math.min(dp[j] + 1, dp[j - 1] + 1, prev + cost);
      dp[j] = val;
      if (val < rowMin) rowMin = val;
      prev = tmp;
    }
    if (rowMin > maxDist) return maxDist + 1;
  }
  return dp[lb];
}

export function refineWordsWithLm(text, lm, lang = "ne") {
  if (!text || !lm || !lm.word_freq || lang === "sa") return text;
  const wf = lm.word_freq;
  const wn = lm.word_next || {};
  const wp = lm.word_prev || {};

  const rawTokens = text.split(/\s+/).filter(Boolean);
  const words = [];
  const puncts = [];
  for (const rt of rawTokens) {
    const stem = rt.replace(/[।,?!-]+$/g, "");
    const p = rt.slice(stem.length);
    if (!stem) {
      if (words.length) puncts[puncts.length - 1] += p;
      continue;
    }
    if (stem === "वर्षपछि") {
      words.push("वर्ष", "पछि");
      puncts.push("", p);
      continue;
    }
    words.push(stem);
    puncts.push(p);
  }

  // Merge detached bound suffixes & compound words present in corpus lexicon
  const mergedW = [];
  const mergedP = [];
  let i = 0;
  while (i < words.length) {
    if (i + 1 < words.length && !puncts[i]) {
      const w1 = words[i], w2 = words[i + 1];
      const joined = w1 + (w2.startsWith("हरु") ? "हरू" + w2.slice(3) : w2);
      const fj = wf[joined] || 0;
      const f1 = wf[w1] || 0;
      const f2 = wf[w2] || 0;
      const hasDigit = /[0-9\u0966-\u096F]/.test(w1);
      if (!hasDigit) {
        if (BOUND_SUFFIXES.has(w2) && w1.length >= 2 && (fj >= 1 || w2.startsWith("हरू") || w2.startsWith("हरु") || ["मा", "ले", "लाई", "बाट", "को", "का", "की"].includes(w2))) {
          if (!["र", "न", "छ", "हो", "की", "कि", "वा", "तर", "अनि", "पनि"].includes(w1)) {
            mergedW.push(joined);
            mergedP.push(puncts[i + 1]);
            i += 2;
            continue;
          }
        } else if (fj >= 4 && (f1 === 0 || f2 === 0 || fj > Math.max(f1, f2))) {
          mergedW.push(joined);
          mergedP.push(puncts[i + 1]);
          i += 2;
          continue;
        }
      }
    }
    mergedW.push(words[i]);
    mergedP.push(puncts[i]);
    i++;
  }

  // Contextual word-bigram snap for OOV words
  for (let idx = 0; idx < mergedW.length; idx++) {
    const w = mergedW[idx];
    if (w.length <= 2 || /[0-9\u0966-\u096F]/.test(w) || (wf[w] || 0) >= 2) continue;
    const prevW = idx > 0 ? mergedW[idx - 1] : "";
    const nextW = idx + 1 < mergedW.length ? mergedW[idx + 1] : "";
    const cands = [...(wn[prevW] || []), ...(wp[nextW] || [])];
    let bestCand = null, bestSc = -1;
    for (const cand of cands) {
      if (cand === w || Math.abs(cand.length - w.length) > 1) continue;
      const maxD = w.length <= 5 ? 1 : 2;
      const d = charEditDist(w, cand, maxD);
      if (d <= maxD && (d === 1 || (cand[0] === w[0] && cand[cand.length - 1] === w[w.length - 1]))) {
        const cf = wf[cand] || 0;
        if (cf >= 6) {
          const sc = cf / (d * d);
          if (sc > bestSc) { bestSc = sc; bestCand = cand; }
        }
      }
    }
    if (bestCand) mergedW[idx] = bestCand;
  }

  return mergedW.map((w, idx) => {
    const fixed = lang === "ne" && PLURAL_VERBS.has(w) ? w + "्" : w;
    return fixed + mergedP[idx];
  }).join(" ");
}

export function cleanSubwordStutter(input, lang = "ne", lm = null) {
  let s = input;
  if (!s) return "";
  s = s.replace(/\s+([\u0901-\u0903\u093E-\u094D])/g, "$1");
  s = s.replace(/(^|\s)[\u0901-\u0903\u093E-\u094D]+/g, "$1");
  s = s.replace(/([\u093E-\u094C])[\u093E-\u094C]+/g, "$1");
  s = s.replace(/\u094D\u094D+/g, "\u094D");
  s = s.replace(/([\u0900-\u097F])[a-zA-Z]+|[a-zA-Z]+(?=[\u0900-\u097F])/g, "$1");
  s = s.replace(/\u0930\u094D\u0930\u094D\u0930/g, "\u0930\u094D\u0930");
  s = s.replace(/([\u0915-\u092F\u0931-\u0939])\1\u094D/g, "$1\u094D");
  s = s.replace(/(^|\s)(?:जेरो|जिरो|जे)\s*(?=०)/g, "$1०");
  s = s.replace(/([\u0904-\u0939][\u0900-\u0963]+)([\u0966-\u096F])/g, "$1 $2");
  s = s.replace(/(^|\s)[\u0915-\u0939][\u093E-\u094C]?(?=[\u0966-\u096F])/g, "$1");
  s = s.replace(/(^|\s)(०)\s+(?=[\u0966-\u096F])/g, "$1$2");
  s = s.replace(/([\u0966-\u096F])\1\1+/g, "$1$1");
  if (lang === "ne") {
    s = s.replace(/(?<![मदतसन])\u094D(?=\s|[।,?!]|$)/g, "");
  }
  const words = s.split(/\s+/).filter(Boolean);
  const out = [];
  for (let i = 0; i < words.length; i++) {
    const w = words[i];
    const wc = w.replace(/[।,?!-]+$/g, "");
    if (!wc) continue;
    if (i + 1 < words.length) {
      const nxt = words[i + 1].replace(/[।,?!-]+$/g, "");
      if (wc.length === 1 && !["र", "न", "छ"].includes(wc) && nxt && /^[\u0966-\u096F]/.test(nxt)) continue;
      if (wc.length >= 1 && wc.length <= 2 && nxt.length >= wc.length + 2 && !KEEP_SINGLE.has(wc) && (nxt.startsWith(wc) || wc[0] === nxt[0])) continue;
      if (wc === nxt && !REDUPLICATIVE.has(wc) && wc.length >= 2) continue;
    }
    if (out.length && wc.length === 1 && wc === out[out.length - 1].replace(/[।,?!-]+$/g, "") && !["र", "न", "छ"].includes(wc)) continue;
    const fixed = lang === "ne" && PLURAL_VERBS.has(wc) ? wc + "्" + w.slice(wc.length) : w;
    out.push(fixed);
  }
  const joined = out.join(" ");
  return lm ? refineWordsWithLm(joined, lm, lang) : joined;
}

/** Group emitted tokens ({id, frame}) into words with timestamps (seconds, 40 ms frames for 4x subsampling). */
export function tokensToWords(tokens, vocab, frameSec = 0.04) {
  const words = [];
  let cur = null;
  for (const { id, frame } of tokens) {
    const t = vocab[id];
    if (isSpecial(t)) continue;
    const startsWord = t.startsWith(WORD_BOUNDARY);
    const piece = t.split(WORD_BOUNDARY).join("");
    const isPunctOnly = /^[।,?!]+$/.test(piece);
    if (!cur || (startsWord && !isPunctOnly)) {
      if (cur) words.push(cur);
      cur = { word: "", t0: Math.max(0, frame * frameSec), t1: (frame + 1) * frameSec };
    }
    cur.word += piece;
    cur.t1 = (frame + 1) * frameSec;
  }
  if (cur) words.push(cur);
  return words.filter((w) => w.word.length);
}
