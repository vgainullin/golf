// Single-game Golf engine for the browser.
//
// A line-by-line port of the rules in src/vectorized_golf.py and the turn loop
// in src/bayes_optimal.py::run_bayes_eval, so that scores produced here are
// comparable with the Python evaluations. Card encoding matches Python:
// card = suit * 13 + rank, rank 0 = "2", ..., 8 = "10", 9 = J, 10 = Q, 11 = K, 12 = A.

export const NUM_RANKS = 13;
export const NUM_CARDS = 52;
export const RANK_SCORES = [-2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 0, 1];
export const RANK_LABELS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"];
export const SUIT_SYMBOLS = ["♠", "♥", "♦", "♣"];
export const MAX_ROUNDS = 40; // same safety cap as the Python eval loop

// Stage-0 actions
export const TAKE_FACE = 0;
export const DRAW_DECK = 1;
// Stage-1 actions: 2..7 = place at slot 0..5, 9..14 = discard held + flip slot 0..5
export const placeAction = (pos) => 2 + pos;
export const flipAction = (pos) => 9 + pos;
export const isPlace = (a) => a >= 2 && a <= 7;
export const isFlip = (a) => a >= 9 && a <= 14;
export const actionPos = (a) => (isPlace(a) ? a - 2 : a - 9);

export const rankOf = (card) => card % NUM_RANKS;
export const suitOf = (card) => Math.floor(card / NUM_RANKS);
export const cardScore = (card) => RANK_SCORES[rankOf(card)];
export const cardLabel = (card) => RANK_LABELS[rankOf(card)] + SUIT_SYMBOLS[suitOf(card)];

/** Seedable PRNG (mulberry32). Returns floats in [0, 1). */
export function makeRng(seed = Date.now()) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function shuffle(arr, rng) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

/**
 * Visible score of a 2x3 layout (slots 0,1,2 = top row; 3,4,5 = bottom row).
 * Hidden cards count 0; a revealed column whose two ranks match scores 0.
 */
export function computeScore(cards, revealed) {
  let total = 0;
  for (let col = 0; col < 3; col++) {
    const a = col, b = col + 3;
    const sa = revealed[a] ? cardScore(cards[a]) : 0;
    const sb = revealed[b] ? cardScore(cards[b]) : 0;
    if (revealed[a] && revealed[b] && rankOf(cards[a]) === rankOf(cards[b])) continue;
    total += sa + sb;
  }
  return total;
}

export const ALL_REVEALED = [true, true, true, true, true, true];
export const finalScore = (cards) => computeScore(cards, ALL_REVEALED);

/** One hole of Golf. All six cards start face-down, as in reset_games(). */
export class Hole {
  constructor(nPlayers = 4, rng = Math.random) {
    this.n = nPlayers;
    this.rng = rng;
    const deck = shuffle([...Array(NUM_CARDS).keys()], rng);
    this.cards = [];
    this.revealed = [];
    for (let p = 0; p < nPlayers; p++) {
      this.cards.push(deck.slice(p * 6, p * 6 + 6));
      this.revealed.push([false, false, false, false, false, false]);
    }
    this.pile = [deck[nPlayers * 6]]; // discard pile, last element is the face-up top
    this.deck = deck.slice(nPlayers * 6 + 1);
    this.holding = new Array(nPlayers).fill(-1);
    this.heldFrom = null; // "pile" | "deck" for the card currently held
    this.current = 0;
    this.stage = 0;
    this.round = 0;
    this.lastTurn = false;
    this.endGamePlayer = -1;
    this.done = false;
    this.log = [];
  }

  get discardTop() {
    return this.pile[this.pile.length - 1];
  }

  get deckRemaining() {
    return this.deck.length;
  }

  visibleScore(p) {
    return computeScore(this.cards[p], this.revealed[p]);
  }

  finalScores() {
    return this.cards.map((c) => finalScore(c));
  }

  validActions() {
    if (this.done) return [];
    if (this.stage === 0) return [TAKE_FACE, DRAW_DECK];
    const out = [];
    for (let pos = 0; pos < 6; pos++) out.push(placeAction(pos));
    for (let pos = 0; pos < 6; pos++) if (!this.revealed[this.current][pos]) out.push(flipAction(pos));
    return out;
  }

  _reshuffleIfEmpty() {
    if (this.deck.length > 0) return;
    const top = this.pile.pop();
    this.deck = shuffle(this.pile, this.rng);
    this.pile = [top];
  }

  /** Stage 0: take the face-up discard (0) or draw from the deck (1). */
  step0(action) {
    if (this.done || this.stage !== 0) throw new Error("not in stage 0");
    const p = this.current;
    if (action === DRAW_DECK) {
      this._reshuffleIfEmpty();
      if (this.deck.length === 0) action = TAKE_FACE; // nothing left to draw
    }
    if (action === TAKE_FACE) {
      this.holding[p] = this.pile.pop();
      this.heldFrom = "pile";
    } else {
      this.holding[p] = this.deck.pop();
      this.heldFrom = "deck";
    }
    this.log.push({ p, stage: 0, action, card: this.holding[p], round: this.round });
    this.stage = 1;
  }

  /** Stage 1: place held card at a slot (2..7) or discard it and flip a hidden slot (9..14). */
  step1(action) {
    if (this.done || this.stage !== 1) throw new Error("not in stage 1");
    const p = this.current;
    const pos = actionPos(action);
    const held = this.holding[p];
    if (isFlip(action) && this.revealed[p][pos]) throw new Error("slot already face-up");
    let discarded;
    if (isPlace(action)) {
      discarded = this.cards[p][pos];
      this.cards[p][pos] = held;
    } else {
      discarded = held;
    }
    this.revealed[p][pos] = true;
    this.pile.push(discarded);
    this.holding[p] = -1;
    this.heldFrom = null;
    this.log.push({ p, stage: 1, action, card: held, discarded, round: this.round });
    this.stage = 0;

    if (this.revealed[p].every(Boolean) && !this.lastTurn) {
      this.lastTurn = true;
      this.endGamePlayer = p;
    }
    this._advance();
  }

  _advance() {
    this.current = (this.current + 1) % this.n;
    if (this.current === 0) this.round += 1;
    if (this.round >= MAX_ROUNDS || (this.lastTurn && this.endGamePlayer === this.current)) {
      this.done = true;
      for (let p = 0; p < this.n; p++) this.revealed[p] = [...ALL_REVEALED];
    }
  }
}

/**
 * Play one hole with agents only. `agents[p]` exposes stage0(hole, p),
 * stage1(hole, p) and optionally observe(hole, p) / resetHole().
 */
export function playHole(agents, rng) {
  const hole = new Hole(agents.length, rng);
  const observeAll = () => agents.forEach((a, p) => a.observe && a.observe(hole, p));
  agents.forEach((a) => a.resetHole && a.resetHole());
  observeAll();
  while (!hole.done) {
    const p = hole.current;
    hole.step0(agents[p].stage0(hole, p));
    observeAll();
    hole.step1(agents[p].stage1(hole, p));
    if (!hole.done) observeAll();
  }
  return hole.finalScores();
}

/** Play a full match (default 9 holes). Returns per-player cumulative scores. */
export function playMatch(agents, holes = 9, rng = Math.random) {
  const totals = new Array(agents.length).fill(0);
  for (let h = 0; h < holes; h++) {
    const s = playHole(agents, rng);
    for (let p = 0; p < agents.length; p++) totals[p] += s[p];
  }
  return totals;
}

/** Win shares: lowest total wins, ties split the win (same as scripts/seat_cycling.py). */
export function winShares(totals) {
  const best = Math.min(...totals);
  const k = totals.filter((t) => t === best).length;
  return totals.map((t) => (t === best ? 1 / k : 0));
}
