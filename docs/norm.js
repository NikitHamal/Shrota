// Canonical NFC Devanagari & Sanskrit normalization + punctuation hygiene
// Exact 1:1 parity with shrota/norm.py

const ALLOWED_PUNCT = new Set(["।", ",", "?", "!", "-"]);

const CHAR_MAP = {
  "\u200b": "",
  "\u200c": "",
  "\u200d": "",
  "\ufeff": "",
  "\u00a0": " ",
  "\u0965": "।",
  "|": "।",
  "॥": "।",
  "–": "-",
  "—": "-",
  "’": "'",
  "‘": "'",
  "“": '"',
  "”": '"',
};

const VEDIC_ACCENT_RE = /[\u0951-\u0954\u1CD0-\u1CFF\uA8E0-\uA8FF]/g;
const LETTER_MARK_NUM_RE = /^[\p{L}\p{M}\p{N}]$/u;

export function normalizeText(text, keepPunct = true, stripAccents = true) {
  if (!text) return "";
  let s = String(text).normalize("NFC");
  for (const [k, v] of Object.entries(CHAR_MAP)) {
    if (s.includes(k)) s = s.split(k).join(v);
  }
  if (stripAccents) {
    s = s.replace(VEDIC_ACCENT_RE, "");
  }
  // Convert trailing period after Devanagari character to danda
  s = s.replace(/([\u0900-\u097F])\s*\.(?=\s|$)/g, "$1।");

  const outChars = [];
  for (const ch of s) {
    if (ALLOWED_PUNCT.has(ch)) {
      outChars.push(keepPunct ? ch : " ");
    } else if (LETTER_MARK_NUM_RE.test(ch)) {
      outChars.push(ch);
    } else if (/\s/.test(ch)) {
      outChars.push(" ");
    } else {
      outChars.push(" ");
    }
  }
  s = outChars.join("");
  if (keepPunct) {
    s = s.replace(/\s+([।,?!])/g, "$1");
    s = s.replace(/([।,?!\-])\s*\1+/g, "$1");
    s = s.replace(/([।,?!])([^\s।,?!])/g, "$1 $2");
  }
  s = s.replace(/\s+/g, " ").trim();
  if (!keepPunct) {
    s = s.replace(/-/g, " ").replace(/\s+/g, " ").trim();
  }
  return s;
}

export function devanagariRatio(text) {
  const s = normalizeText(text, false);
  const letters = [...s].filter((c) => !/\s/.test(c));
  if (letters.length === 0) return 0.0;
  const dev = letters.filter((c) => c >= "\u0900" && c <= "\u097f").length;
  return dev / letters.length;
}

export function hasPunctuation(text) {
  const s = String(text || "");
  return s.includes("।") || s.includes("?") || s.includes("!") || s.includes(",");
}

export function tagUtterance(text, lang = "ne", forceMode = null) {
  const rawHasPunct = hasPunctuation(text);
  const mode = forceMode || (rawHasPunct ? "punct" : "nopunct");
  const normed = normalizeText(text, mode === "punct");
  const langTok = lang === "sa" ? "<sa>" : "<ne>";
  const punctTok = mode === "punct" ? "<punct>" : "<nopunct>";
  return `${langTok} ${punctTok} ${normed}`.trim();
}
