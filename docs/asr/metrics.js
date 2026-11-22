// Word / character error rate for on-page sample scoring.
import { normalizeText } from "../norm.js";

function lev(a, b) {
  let prev = Array.from({ length: b.length + 1 }, (_, j) => j);
  for (let i = 1; i <= a.length; i++) {
    const cur = [i];
    for (let j = 1; j <= b.length; j++) {
      cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
    }
    prev = cur;
  }
  return prev[b.length];
}

export function scoreAgainst(reference, hypothesis) {
  const ref = normalizeText(reference, false);
  const hyp = normalizeText(hypothesis, false);
  if (!ref) return null;
  const rw = ref.split(" ").filter(Boolean);
  const hw = hyp.split(" ").filter(Boolean);
  return {
    wer: (100 * lev(rw, hw)) / Math.max(rw.length, 1),
    cer: (100 * lev([...ref], [...hyp])) / Math.max([...ref].length, 1),
  };
}
