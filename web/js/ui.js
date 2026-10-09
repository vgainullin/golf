// Small DOM helpers shared by the views.

import { RANK_SCORES, rankOf, suitOf, RANK_LABELS, SUIT_SYMBOLS } from "./engine.js";

export const $ = (sel, root = document) => root.querySelector(sel);

export function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

export async function loadJSON(path) {
  const res = await fetch(path, { cache: "no-cache" });
  if (!res.ok) throw new Error(`${path}: HTTP ${res.status}`);
  return res.json();
}

/** Base URL of the repository checkout relative to web/ (overridable for static builds). */
export const REPO_BASE = document.querySelector('meta[name="repo-base"]')?.content || "../";

export const fmt = (x, d = 2) => (x == null || Number.isNaN(x) ? "—" : x.toFixed(d));
export const pct = (x, d = 1) => (x == null ? "—" : `${(x * 100).toFixed(d)}%`);
export const money = (x, cur = "USD") =>
  new Intl.NumberFormat("en-US", { style: "currency", currency: cur, maximumFractionDigits: 0 }).format(x);

/** HTML for one playing card. `faceUp=false` renders the back. */
export function cardHTML(card, { faceUp = true, small = false, cls = "", attrs = "" } = {}) {
  const size = small ? " small" : "";
  if (card == null || card < 0) return `<div class="pc empty${size} ${cls}" ${attrs}></div>`;
  if (!faceUp) return `<div class="pc back${size} ${cls}" ${attrs} aria-label="face-down card"></div>`;
  const red = suitOf(card) === 1 || suitOf(card) === 2 ? " red" : "";
  const label = RANK_LABELS[rankOf(card)] + SUIT_SYMBOLS[suitOf(card)];
  const sc = RANK_SCORES[rankOf(card)];
  return `<div class="pc${red}${size} ${cls}" ${attrs} aria-label="${label}, ${sc} points">${label}<span class="sc">${sc}</span></div>`;
}

/** Slots whose column is a revealed rank match (both cards score 0). */
export function matchedSlots(cards, revealed) {
  const out = new Set();
  for (let c = 0; c < 3; c++) {
    if (revealed[c] && revealed[c + 3] && rankOf(cards[c]) === rankOf(cards[c + 3])) {
      out.add(c); out.add(c + 3);
    }
  }
  return out;
}

/** Mean and 95% CI half-width of a numeric array. */
export function meanCI(xs) {
  const n = xs.length;
  if (!n) return [NaN, NaN];
  const m = xs.reduce((a, b) => a + b, 0) / n;
  if (n < 2) return [m, NaN];
  const v = xs.reduce((a, b) => a + (b - m) ** 2, 0) / (n - 1);
  return [m, 1.96 * Math.sqrt(v / n)];
}
