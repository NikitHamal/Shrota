// Token -> text decoding and Devanagari cleanup. Port of ShrotaTokenizer.decode and
// clean_subword_stutter_and_matras from shrota/tokenizer.py and shrota/streaming.py.
export const WORD_BOUNDARY = "\u2581";
const SPECIAL = new Set(["<blk>", "<unk>", "<bos>", "<eos>", "<ne>", "<sa>", "<punct>", "<nopunct>"]);
const KEEP_SINGLE = new Set(["र", "न", "छ", "हो", "की", "कि", "यो", "ती", "सो", "के", "को"]);

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

export function cleanSubwordStutter(input, lang = "ne") {
  let s = input;
  if (!s) return "";
  s = s.replace(/\s+([\u0901-\u0903\u093E-\u094D])/g, "$1");
  s = s.replace(/(^|\s)[\u0901-\u0903\u093E-\u094D]+/g, "$1");
  s = s.replace(/([\u093E-\u094C])[\u093E-\u094C]+/g, "$1");
  s = s.replace(/\u094D\u094D+/g, "\u094D");
  s = s.replace(/(^|\s)[0-9\u0966-\u096F](?=[\u0915-\u0939]\u094D)/g, "$1यद");
  if (lang === "ne") s = s.replace(/(?<![मदतस])\u094D(?=\s|[।,?!]|$)/g, "");
  const words = s.split(/\s+/).filter(Boolean);
  const out = [];
  for (let i = 0; i < words.length; i++) {
    const w = words[i];
    const wc = w.replace(/[।,?!]+$/g, "");
    if (i + 1 < words.length) {
      const nxt = words[i + 1];
      if (wc.length >= 1 && wc.length <= 2 && nxt.length >= 2 && wc[0] === nxt[0] && !KEEP_SINGLE.has(wc)) continue;
    }
    out.push(w);
  }
  return out.join(" ");
}

/** Group emitted tokens ({id, frame}) into words with timestamps (seconds, 80 ms frames). */
export function tokensToWords(tokens, vocab, frameSec = 0.08) {
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
