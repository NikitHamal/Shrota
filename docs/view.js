// Small DOM/text helpers shared by the studio modules.
export const $ = (id) => document.getElementById(id);
export const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

/** Word-level LCS diff: html for `b` with words that differ from `a` highlighted. */
export function diffHtml(a, b) {
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

export const pct = (v) => `${v.toFixed(0)}%`;
