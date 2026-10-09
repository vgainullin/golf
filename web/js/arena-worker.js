// Runs arena simulations off the main thread.
// Message in:  { roster: string[], gamesPerSeating: number, holes: number, cycle: boolean, seed: number }
// Messages out: { type: "progress", done, total } and { type: "result", labels, seatings, matches, ms }

import { playMatch, winShares, makeRng } from "./engine.js";
import { makeAgent } from "./agents.js";

function permutations(xs) {
  if (xs.length <= 1) return [xs.slice()];
  const out = [], seen = new Set();
  xs.forEach((x, i) => {
    for (const rest of permutations([...xs.slice(0, i), ...xs.slice(i + 1)])) {
      const p = [x, ...rest], key = p.join(",");
      if (!seen.has(key)) { seen.add(key); out.push(p); }
    }
  });
  return out;
}

self.onmessage = (ev) => {
  const { roster, gamesPerSeating, holes, cycle, seed } = ev.data;
  const rng = makeRng(seed);
  const seatings = cycle ? permutations(roster) : [roster];
  const total = seatings.length * gamesPerSeating;
  const perMatch = {}; // label -> per-match avg score per hole
  const wins = {};     // label -> win shares
  roster.forEach((k) => { perMatch[k] = []; wins[k] = []; });

  const t0 = performance.now();
  let done = 0, lastPost = 0;
  for (const seating of seatings) {
    for (let g = 0; g < gamesPerSeating; g++) {
      const agents = seating.map((k) => makeAgent(k, rng));
      const totals = playMatch(agents, holes, rng);
      const shares = winShares(totals);
      seating.forEach((k, s) => { perMatch[k].push(totals[s] / holes); wins[k].push(shares[s]); });
      done++;
      const now = performance.now();
      if (now - lastPost > 80) { self.postMessage({ type: "progress", done, total }); lastPost = now; }
    }
  }
  const labels = Object.keys(perMatch).map((k) => {
    const xs = perMatch[k], n = xs.length;
    const mean = xs.reduce((a, b) => a + b, 0) / n;
    const sd = Math.sqrt(xs.reduce((a, b) => a + (b - mean) ** 2, 0) / Math.max(n - 1, 1));
    const win = wins[k].reduce((a, b) => a + b, 0) / n; // per-seat win rate
    return { key: k, mean, ci: 1.96 * sd / Math.sqrt(n), win, samples: n };
  });
  labels.sort((a, b) => a.mean - b.mean);
  self.postMessage({ type: "result", labels, seatings: seatings.length, matches: total, ms: performance.now() - t0 });
};
