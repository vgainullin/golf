// Run with: node --test web/tests/
import test from "node:test";
import assert from "node:assert/strict";
import { computeScore, finalScore, Hole, playMatch, makeRng, winShares, TAKE_FACE, placeAction, flipAction } from "../js/engine.js";
import { makeAgent } from "../js/agents.js";

const card = (rank, suit = 0) => suit * 13 + rank; // rank 0 = "2", 11 = K, 12 = A

test("scoring matches src/vectorized_golf.py", () => {
  // 2, K, A on top; 9, 10, Q below -> -2 + 0 + 1 + 9 + 10 + 10
  assert.equal(finalScore([card(0), card(11), card(12), card(7), card(8), card(10)]), 28);
  // column match zeroes both cards, even 2s
  assert.equal(finalScore([card(7), card(0), card(3), card(7, 1), card(0, 2), card(4)]), 5 + 6);
  // hidden cards count 0 and never match
  assert.equal(computeScore([card(7), card(1), card(1), card(7, 1), card(1), card(1)], [true, false, false, false, false, false]), 9);
});

test("hole ends one full round after a player reveals all six cards", () => {
  const hole = new Hole(4, makeRng(1));
  for (let i = 0; i < 6; i++) hole.revealed[0][i] = i !== 5;
  hole.step0(TAKE_FACE);
  hole.step1(flipAction(5));
  assert.equal(hole.lastTurn, true);
  for (let p = 1; p < 4; p++) { hole.step0(TAKE_FACE); hole.step1(placeAction(0)); }
  assert.equal(hole.done, true);
});

test("win shares split ties", () => {
  assert.deepEqual(winShares([10, 10, 20, 30]), [0.5, 0.5, 0, 0]);
});

// Reference values from `python -m src.bayes_optimal --player lookahead --games 3000 --holes 9 --eval-config <cfg>`
// (6.869 for R,H,R and 7.844 for H,R,H) and the documented heuristic baseline (~14.1/hole).
test("browser agents reproduce the Python players' scores", () => {
  const rng = makeRng(11);
  const seat0 = (roster, n) => {
    let tot = 0;
    for (let g = 0; g < n; g++) tot += playMatch(roster.map((k) => makeAgent(k, rng)), 9, rng)[0] / 9;
    return tot / n;
  };
  const l = seat0(["lookahead", "random", "improved", "random"], 600);
  assert.ok(Math.abs(l - 6.869) < 0.35, `lookahead R,H,R = ${l}`);
  const h = seat0(["heuristic", "random", "heuristic", "random"], 1500);
  assert.ok(Math.abs(h - 14.1) < 0.4, `heuristic = ${h}`);
});
