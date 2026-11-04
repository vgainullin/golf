# Weight-space programming of the Exp14 DQN champion (run opus_max_b25)

**Result:** `final.pt` = `cand_03.pt`, **8.955** avg score/hole on the practice benchmark vs **9.523**
for `base.pt` (paired Δ −0.568, SE 0.094 over 24 seatings; win rate 37.4% → 42.4%). Two edits:
- a hand-built circuit that removes most of the network's learned premium for revealing cards;
- a one-number fix to the take/draw head, making it consistent with the network's own stage-1 values.

Five of the 25 passes were used.

## What the network computes

Architecture recap: card tokens (rank emb 14×32 ⊕ suit emb 5×32) + position emb → two pre-norm
transformer layers (4 heads, FFN 256) → mean-pool over positions 0..6 (own six slots + held card) →
LayerNorm → concat deck_remaining/27 → MLP 65-256-256-16.

Training reward is the hindsight true-score change, so Q is in points:
Q(s,a) − Q(s,b) ≈ the difference in expected final score.

1. **Per-slot "keeper" units (second MLP layer, h2).** These are families of units that fire when slot p
   holds a *low* revealed card. Activation order: 2 > K > A > 3 > 4 > 5 > …; for 9/T/J/Q and for a
   face-down card it sits at baseline.
   - slot 0: 153, 29, 139, 198
   - slot 1: 14, 190
   - slot 2: 128, 98, 96, 37
   - slot 3: 42, 120, 233
   - slot 4: 212, 176, 234
   - slot 5: 143, 13

   Each writes ≈ −0.5..−1.0 into Q[place at p] only, i.e. "don't replace a good card at p".
2. **Where a high card sits is nearly invisible.**
   - In the pooled vector, "a 2 at slot p" is a slot-specific direction (cosine ≈ 0.35 between slots),
     but "a T at slot p" is mostly a common direction (cosine ≈ 0.9).
   - Moving slot p from face-down to a revealed T changes Q[place p] − Q[flip] by +0.6. The 1-step
     truth is +4.5.
   - h2 units 115, 126, 221 and h1 units 136, 165, 172 encode "some high card is showing", with no
     location.
   - At the layer-1 FFN input, revealed J/T at slots 1, 4, 5 are partly indistinguishable from
     face-down cards. Layer 0 still separates them perfectly.
3. **A ~2–3 point premium for revealing.** Measured against flipping, on traced states:
   - Placing on a face-down slot tracks the 1-step expected gain of the Bayes lookahead's model
     (slope 0.9, offset −0.4).
   - Placing on an already revealed slot is valued 2.4–3.9 points below its 1-step gain, for every
     rank 2..Q.
   - On final turns (an opponent is already out) the premium is only 0.4. On normal turns it is ~2.0
     (decision-relevant region) and up to ~3 overall.

   Effect: the base net reveals fast and ends the hole itself. At seat 0 it went out first in 61/72
   traced holes. That is bad for its own score, because every hole-ending cuts its own improvement
   turns. Seat-cycling shows this: the seat that acts first scores ~1 point better for every player.
4. **The stage-0 head is biased toward taking.** Q0(take) − Q0(draw) exceeds the network's own
   one-step lookahead by ≈ +1.0 at every face rank. The lookahead is: take = max over valid stage-1
   actions of Q1 with the face card in hand; draw = belief-weighted mean over unseen ranks of max Q1.
   The offset is ~0.4–0.7 with one face-down card left and ~1.0–1.2 mid-hole. The base takes 4s/5s
   81%/61% of the time; its own lookahead says 32%/21% (the lookahead player takes 43%/30%).
5. **Terminal blindness.** Actions that end C's hole (acting on its last face-down card, or anything
   on its final turn) get Q ≈ 14, where the truth is ≈ the immediate reward. The net does not
   represent terminality. It compares mostly immediate rewards, so it under-values continuing.
   Place-vs-flip on the last card is still well calibrated (10 wrong of 248, ~0.06/hole).
6. **Spare capacity.**
   - 106/256 h2 units and 17/256 h1 units are never active on real states (83 and 9 on augmented
     states).
   - Many layer-1 FFN units can be ablated with max |ΔQ| < 0.3 and no decision change.
   - Layer-1 head 3 is the least important head: ablating it changes 4.6% of traced decisions, with
     regret 0.005 by the net's own Q.
7. **Progress units.** h1 unit 30 ≈ the opponents' total face-down count (+0.36 per card,
   R² 0.83; almost no dependence on C's own count). h1 unit 233 ≈ overall hole progress.

   The network does *not* encode "an opponent is fully revealed" (= C's final turn). A linear readout
   that looks 94% accurate on traces drops to 73% once synthetic "opponent just went out" states are
   added; it was reading general lateness.

## Edits in final.pt

### E1 — reveal-premium correction circuit (`build_reveal.py`, δ = 1.5, fit on trace_001+002)
Intended function: Q[place p] −= δ·H_p and Q[flip *] −= δ, where H_p = "own slot p is face-down".
Every card-revealing stage-1 action costs δ relative to replacing an already revealed card.

- **Detector (transformer layer-1 FFN, units 75 113 192 55 24 9 187 135 79 222 32 201; each had
  max |ΔQ| ≤ 0.26 when ablated).** Six clip pairs, ReLU(a) − ReLU(a−1).
  - Input weights are a closed-form Fisher direction for "token is slot p and face-down" on the
    layer-1 FFN input (trace states plus slot-permuted variants).
  - Thresholds sit at the 99th/1st class percentiles.
  - Output goes along a "channel": a least-variance pooled direction inside the subspace orthogonal
    to 1/γ_poolnorm, so LayerNorm centring leaves the six channels orthonormal.
  - This is the last layer, so only the mean-pool sees it.
- **Isolation.** Every live h1 row has the 6 channel directions projected out, with its bias
  compensated at the data mean.
- **Readers.** Dead h1 units 4, 6, 170, 181, 190, 192 each read one channel. Dead h2 units
  0, 5, 6, 19, 21, 22, 23, 31, 43, 45, 54, 58 clip the readings into H_p ∈ [0,1]. Their output
  weights reach only Q[place p] (∓δ). Their input weights read only their own reader.
- **Flip biases:** b4[9:15] −= δ.
- **Known limits.**
  - False "face-down" on revealed J/T at slots 1/4/5 in ~1.5–3% of states. The bonus is simply
    missing there.
  - False "revealed" ≈ 0.
  - The injected channels raise the pooled std by ~3%, shifting all Q by ~0.5 roughly uniformly.
    That changed 0/471 stage-0 decisions on traces.

### E2 — stage-0 consistency fix (`edit_s0.py`)
b4[0] −= 1.0 (the take bias; action 0 is valid only in stage 0). After the fix the take head agrees
with its own lookahead (disagreements 44/471 → 8/507). The take rate fell from 0.41 to 0.30.

### Weights changed (final.pt vs base.pt): 22,876 of 189,552
| tensor | changed |
|---|---|
| transformer.layers.1.linear1.weight / bias | 768 / 12 (12 FFN units) |
| transformer.layers.1.linear2.weight | 768 (their output columns) |
| mlp.0.weight / bias | 16,326 / 255 (channel projection touches every live row; 6 reader rows) |
| mlp.2.weight / bias | 4,536 / 12 (12 new h2 rows; new h1 columns zeroed into old units) |
| mlp.4.weight / bias | 192 / 7 (12 new columns; b4[0], b4[9:15]) |

## Passes (practice: 100 games × 24 seatings, seed 1)
| pass | candidate | change | C avg/hole | Δ vs base (SE) | L avg |
|---|---|---|---|---|---|
| 1 | base | — | 9.523 | — | 9.043 |
| 2 | cand_02 | E1 δ=1.5 (fit on trace 1) | 9.234 | −0.289 (0.103) | 8.788 |
| 3 | cand_03 = **final** | E1 δ=1.5 (fit on traces 1+2) + E2 | **8.955** | **−0.568 (0.094)** | 8.769 |
| 4 | cand_04 | E1 δ=3.0 + E2 | 9.322 | −0.201 (0.156) | 8.460 |
| 5 | cand_05 | E1 δ=2.0 + E2 (only W4/b4 differ from cand_03) | 8.971 | −0.552 (0.126) | 8.650 |

Pairwise SE between close candidates is ~0.08–0.11. The δ optimum is flat over 1.5–2.0.

Per seat (C vs base), δ = 1.5 / 2.0 / 3.0:

| C's seat | δ=1.5 | δ=2.0 | δ=3.0 |
|---|---|---|---|
| 0 | −0.83 | −1.05 | −0.88 |
| 1 | −0.72 | −0.67 | −0.23 |
| 2 | −0.30 | −0.30 | +0.13 |
| 3 | −0.42 | −0.19 | +0.18 |

Not revealing pays when C is the one who would end the hole (early seats), and costs when it isn't.
L and I also score better against the edited C, because holes get longer.

## What did not work / was abandoned
- **δ = 3.0.** It produces "stalls" that lose points on C's final turn, where there is no future,
  e.g. replacing a revealed 4 with an 8 to avoid flipping. Final-turn decisions cost 0.44/hole on the
  seat-0 trace (δ=1.5: 0.08/hole).
- **Two-stage detector (`build_reveal2.py`).**
  - Plan: a clean layer-0 AND detector ("face-down" ∧ "slot p"; separable with margin there) writes a
    small flag, and the layer-1 FFN amplifies it. Aim: remove the J/T false positives.
  - Failure: the flag readout at the layer-1 FFN input is too noisy after LayerNorm (natural range
    ±0.35–1.0 along the quietest directions). Only a flag of norm ≥5 plus a second position gate
    would work.
  - Not built: the payoff (~2% of states) was too small.
- **Final-turn gate and go-out penalty** (both analysed, not built). Both need a reliable "an
  opponent is fully revealed" signal, which the net lacks (see finding 7). An ungated go-out penalty
  would distort the ~28 per 72 final-turn decisions that have one face-down card left, the same
  failure as δ=3.
- **Rejected after analysis:**
  - "Which face-down slot" disagreements with the lookahead (~0.5 pt each). The offsets are
    systematic (e.g. mid card next to a revealed low card valued −1.1) and plausibly reflect
    "reserve that slot", which 1-step cannot see.
  - Place-vs-flip on the last card: calibrated.
  - Position biases: ±0.2.
  - Take-then-flip: never happens.

## What I would try next
1. **Final-turn detector by attention surgery.** Repurpose layer-0 head 3 (ablation regret 0.017):
   - Queries of the opponents' holding tokens (14/21/28, always face-down) attend to their own
     opponent's slots, keyed "group member + c·face-down" (layer-0 readouts are clean, ranges
     ≤ 1.3×).
   - Value = "revealed", so the output is ≈1 only when that opponent is fully revealed.
   - Repurpose layer-1 head 3 (regret 0.005) to broadcast that flag to all pooled tokens. Seven
     tokens carrying it gives full amplitude in the pool with negligible LayerNorm disturbance.

   Then gate δ off on the final turn and add a go-out penalty on normal turns.
2. **Race-aware δ.** Make δ larger when C leads the reveal race and smaller when an opponent is
   close to going out. h1 unit 30 (opponents' face-down total) is a ready-made linear gate. Needs
   clipped H_p at h1 (12 units) plus AND units in h2.
3. **Rank-dependent bonus.** After the uniform δ, replacing revealed T/J/Q is still under-valued by
   ~2 points vs ~1 for mid cards.
4. **Phase-dependent take bias.** The −1.0 over-corrects by ~0.4–0.6 when C has one face-down card
   left.

## Files
| file | purpose |
|---|---|
| `wsp_tools.py` | model I/O, feature taps, obs-only lookahead values |
| `trace_util.py` | drops trace rows logged for finished games |
| `net_internals.py` | manual forward pass exposing transformer internals |
| `build_reveal.py` | E1 |
| `edit_s0.py` | E2 |
| `verify_reveal.py` | intended-vs-actual Q change check |
| `s0_consistency.py` | take-head vs own-lookahead check |
| `build_reveal2.py` | the abandoned two-stage detector |
| `ft_data.py` | synthetic final-turn states |
| `circuit_meta_c03.json` | unit indices used in final.pt |

No gradient descent and no game simulation were used. The Fisher directions (class-mean differences
whitened by the pooled within-class covariance), least-variance directions (eigenvectors) and
projections are closed-form linear algebra on network activations from the pass traces.
