// Browser ports of the hand-coded and belief-based players.
//
// Each port mirrors its Python counterpart so arena results line up with the
// repo's own evaluations:
//   random      -> random_stage0 / random_stage1          (src/vectorized_golf.py)
//   simple      -> simple_stage0 / simple_stage1
//   heuristic   -> heuristic_stage0 / heuristic_stage1     ("base heuristic", H in scripts/seat_cycling.py)
//   improved    -> heuristic_stage0 / improved_stage1      ("improved heuristic", I)
//   lookahead   -> lookahead_stage0 / lookahead_stage1     (src/bayes_optimal.py, L)

import {
  NUM_CARDS, NUM_RANKS, RANK_SCORES, TAKE_FACE, DRAW_DECK,
  computeScore, rankOf, cardScore, placeAction, flipAction,
} from "./engine.js";

const RANK_CUTOFF = 4;

const firstHidden = (revealed) => revealed.findIndex((r) => !r);

// ---------------------------------------------------------------------------
// Random / simple
// ---------------------------------------------------------------------------

function randomAgent(rng = Math.random) {
  return {
    stage0: () => (rng() < 0.5 ? DRAW_DECK : TAKE_FACE),
    stage1: (hole) => {
      const valid = hole.validActions();
      return valid[Math.floor(rng() * valid.length)];
    },
  };
}

function simpleStage0(hole) {
  return cardScore(hole.discardTop) < RANK_CUTOFF ? TAKE_FACE : DRAW_DECK;
}

function simpleAgent(rng = Math.random) {
  return {
    stage0: simpleStage0,
    stage1: (hole, p) => {
      const rev = hole.revealed[p];
      const hidden = [0, 1, 2, 3, 4, 5].filter((i) => !rev[i]);
      const pool = hidden.length ? hidden : [0, 1, 2, 3, 4, 5];
      return placeAction(pool[Math.floor(rng() * pool.length)]);
    },
  };
}

// ---------------------------------------------------------------------------
// Heuristics
// ---------------------------------------------------------------------------

export function heuristicStage0(hole, p) {
  const face = hole.discardTop;
  if (cardScore(face) < RANK_CUTOFF) return TAKE_FACE;
  const r = rankOf(face);
  const cards = hole.cards[p], rev = hole.revealed[p];
  for (let i = 0; i < 6; i++) if (rev[i] && rankOf(cards[i]) === r) return TAKE_FACE;
  return DRAW_DECK;
}

function trialScore(cards, revealed, pos, held) {
  const c = cards.slice(); c[pos] = held;
  const r = revealed.slice(); r[pos] = true;
  return computeScore(c, r);
}

function cutoffDecision(current, bestScore, bestPos, rev) {
  const first = firstHidden(rev);
  if (bestScore <= current - RANK_CUTOFF) return placeAction(bestPos);
  if (first >= 0 && bestScore - current < RANK_CUTOFF) return placeAction(first);
  if (first >= 0) return flipAction(first);
  return placeAction(bestPos);
}

/** Base heuristic: only considers placing at face-down slots. */
export function heuristicStage1(hole, p) {
  const cards = hole.cards[p], rev = hole.revealed[p], held = hole.holding[p];
  const current = computeScore(cards, rev);
  let bestScore = 1e6, bestPos = 0;
  for (let pos = 0; pos < 6; pos++) {
    if (rev[pos]) continue;
    const s = trialScore(cards, rev, pos, held);
    if (s < bestScore) { bestScore = s; bestPos = pos; }
  }
  if (firstHidden(rev) < 0) {
    for (let pos = 0; pos < 6; pos++) {
      const c = cards.slice(); c[pos] = held;
      const s = computeScore(c, rev);
      if (s < bestScore) { bestScore = s; bestPos = pos; }
    }
    return placeAction(bestPos);
  }
  return cutoffDecision(current, bestScore, bestPos, rev);
}

/** Improved heuristic: also considers replacing face-up cards. */
export function improvedStage1(hole, p) {
  const cards = hole.cards[p], rev = hole.revealed[p], held = hole.holding[p];
  const current = computeScore(cards, rev);
  let bestScore = 1e6, bestPos = 0;
  for (let pos = 0; pos < 6; pos++) {
    const s = trialScore(cards, rev, pos, held);
    if (s < bestScore) { bestScore = s; bestPos = pos; }
  }
  return cutoffDecision(current, bestScore, bestPos, rev);
}

// ---------------------------------------------------------------------------
// Bayes 1-step lookahead
// ---------------------------------------------------------------------------

/**
 * Expected final score of a layout when hidden slots are drawn without
 * replacement from the unobserved multiset (port of expected_score()).
 */
export function expectedScore(cards, revealed, multiset, total) {
  const totalF = Math.max(total, 1);
  const totalM1 = Math.max(total - 1, 1);
  let sumNScore = 0, pairCorr = 0;
  for (let r = 0; r < NUM_RANKS; r++) {
    sumNScore += multiset[r] * RANK_SCORES[r];
    pairCorr += ((multiset[r] * (multiset[r] - 1)) / (totalF * totalM1)) * RANK_SCORES[r];
  }
  const eUnknown = sumNScore / totalF;
  const eBothHidden = 2 * eUnknown - 2 * pairCorr;

  const oneRevealed = (rank) => {
    const s = RANK_SCORES[rank];
    const n = multiset[rank];
    const pMatch = n / totalF;
    const eOther = (sumNScore - n * s) / Math.max(totalF - n, 1);
    return (1 - pMatch) * (s + eOther);
  };

  let out = 0;
  for (let col = 0; col < 3; col++) {
    const a = col, b = col + 3;
    const ra = rankOf(cards[a]), rb = rankOf(cards[b]);
    if (revealed[a] && revealed[b]) out += ra === rb ? 0 : RANK_SCORES[ra] + RANK_SCORES[rb];
    else if (revealed[a]) out += oneRevealed(ra);
    else if (revealed[b]) out += oneRevealed(rb);
    else out += eBothHidden;
  }
  return out;
}

function bestPlacement(cards, revealed, held, multiset, total) {
  let bestScore = 1e6, bestPos = 0;
  for (let pos = 0; pos < 6; pos++) {
    const c = cards.slice(); c[pos] = held;
    const r = revealed.slice(); r[pos] = true;
    const s = expectedScore(c, r, multiset, total);
    if (s < bestScore) { bestScore = s; bestPos = pos; }
  }
  return [bestScore, bestPos];
}

function lookaheadAgent() {
  const unobserved = new Array(NUM_CARDS).fill(true);
  const multiset = () => {
    const m = new Array(NUM_RANKS).fill(0);
    for (let c = 0; c < NUM_CARDS; c++) if (unobserved[c]) m[c % NUM_RANKS]++;
    return m;
  };
  const total = () => unobserved.reduce((s, u) => s + (u ? 1 : 0), 0);

  return {
    resetHole: () => unobserved.fill(true),
    // Remove every card visible from seat p: face-up layouts, discard top, own holding.
    observe: (hole, p) => {
      for (let q = 0; q < hole.n; q++)
        for (let i = 0; i < 6; i++) if (hole.revealed[q][i]) unobserved[hole.cards[q][i]] = false;
      if (hole.pile.length) unobserved[hole.discardTop] = false;
      if (hole.holding[p] >= 0) unobserved[hole.holding[p]] = false;
    },
    stage0: (hole, p) => {
      const cards = hole.cards[p], rev = hole.revealed[p];
      const ms = multiset(), tot = total();
      const currentE = expectedScore(cards, rev, ms, tot);
      const eTake = Math.min(bestPlacement(cards, rev, hole.discardTop, ms, tot)[0], currentE);
      let eDraw = 0;
      for (let r = 0; r < NUM_RANKS; r++) {
        if (ms[r] === 0) continue;
        const pr = ms[r] / Math.max(tot, 1);
        const dms = ms.slice(); dms[r] = Math.max(dms[r] - 1, 0);
        const dt = Math.max(tot - 1, 1);
        const best = bestPlacement(cards, rev, r, dms, dt)[0];
        eDraw += pr * Math.min(best, expectedScore(cards, rev, dms, dt));
      }
      return eTake <= eDraw ? TAKE_FACE : DRAW_DECK;
    },
    stage1: (hole, p) => {
      const cards = hole.cards[p], rev = hole.revealed[p];
      const ms = multiset(), tot = total();
      const currentE = expectedScore(cards, rev, ms, tot);
      const [best, pos] = bestPlacement(cards, rev, hole.holding[p], ms, tot);
      const first = firstHidden(rev);
      return best < currentE || first < 0 ? placeAction(pos) : flipAction(first);
    },
  };
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

export const AGENTS = {
  random: {
    name: "Random", code: "R", blurb: "Uniform over legal moves. The floor.",
    make: (rng) => randomAgent(rng),
  },
  simple: {
    name: "Greedy", code: "S", blurb: "Takes low discards, otherwise drops cards on random face-down slots.",
    make: (rng) => ({ ...simpleAgent(rng) }),
  },
  heuristic: {
    name: "Base heuristic", code: "H", blurb: "Column-aware, but never swaps out a face-up card.",
    make: () => ({ stage0: heuristicStage0, stage1: heuristicStage1 }),
  },
  improved: {
    name: "Improved heuristic", code: "I", blurb: "Base heuristic that also replaces face-up cards.",
    make: () => ({ stage0: heuristicStage0, stage1: improvedStage1 }),
  },
  lookahead: {
    name: "Bayes lookahead", code: "L", blurb: "Tracks every unseen card and picks the move with the lowest expected final score.",
    make: () => lookaheadAgent(),
  },
};

export function makeAgent(key, rng) {
  const spec = AGENTS[key];
  if (!spec) throw new Error(`unknown agent ${key}`);
  return { key, ...spec.make(rng) };
}
