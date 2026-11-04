# Weight-space programming, round 3 (run opus_max_r3): stop going out early, reveal fast, hold two

**Result:** `final.pt` = `cand_21.pt` (sha256 ff3454f0…5d4a). Confirmed at **8.1720** per hole vs **8.8366**
for `base.pt`: paired Δ **−0.665** (SE 0.026) over 24,000 games on the confirm seed.

| | base.pt | cand_12 | cand_18 | **cand_21 = final** |
|---|---|---|---|---|
| practice (seed 1, 7,200 games) | 8.8141 | 8.2067 (−0.607) | 8.1854 (−0.629) | **8.1147 (−0.699, SE 0.031)** |
| confirm (seed 2, 24,000 games) | 8.8366 | 8.2454 (−0.591) | 8.2321 (−0.605) | **8.1720 (−0.665, SE 0.026)** |
| L in the same confirm games | 8.755 | 8.218 | 8.168 | 8.157 |
| C win rate (confirm) | 43.3% | 44.4% | 43.9% | 44.6% |

- On the confirm seed, paired: cand_21 − cand_18 = −0.060 (SE 0.007) and cand_18 − cand_12 = −0.013 (SE 0.005).
- Per seat on confirm, cand_21 vs base: −0.76 / −0.71 / −0.61 / −0.58 for C in seats 0–3.
- Only the MLP head was edited: 17,671 of 189,552 weights. The transformer is untouched.
- Used 25 of 25 practice passes (the last for an ablation, after the confirms were spent) and 3 of 3 confirms.
- L and I also score better against the edited C, because holes get longer. Only C's score was targeted.

## The story in three steps

1. **Going out is the error.** Only C's own score counts.
   - When C reveals its last face-down card on a normal turn, it ends its own improvement; the others get one
     more turn and C gets none.
   - If C instead places the held card on a revealed slot ("stays"), it is guaranteed at least one more turn:
     a normal turn, or its final turn if an opponent goes out first.
   - The base net went out 58% of the time in that spot.
   - A go-out penalty (GO) alone, with round 1's reveal penalty still in place, was worth −0.39 (passes 2–5).
2. **Round 1's general reveal penalty was a proxy for that.** Once going out has its own penalty, the general
   penalty hurts. Revealing early has information value: you learn which cards to replace.
   - Penalty → 0 (passes 6–8): −0.15.
   - A small reveal bonus (pass 10): another −0.03.
3. **But not at two face-down cards.** With exactly two face-down cards, one reveal puts C into the
   one-card regime, where every non-ending move is a replacement. So the bonus was split by count:
   - Bonus 0.75 → 0 at two face-down: −0.044 ± 0.006 (pass 19).
   - Bonus 0.75 → 1.5 at three or more: −0.027 ± 0.011 (pass 21). 2.5 changed nothing (saturated).
   - A penalty at two face-down was flat (pass 20).

Resulting play (practice trace 21 vs base trace 1):
- Normal-turn reveal rate by own face-down count: 6–3 cards: 1.00 / 1.00 / 0.99 / 0.96 (base 1.00 / 0.93 /
  0.89 / 0.87); 2 cards: 0.77 (base 0.74); 1 card: 0.19 (base 0.58).
- C goes out in 17% of holes (base 35%) and has a final turn in 83% (65%).

**Ablations of the final config.**
- Pass 23: without GO, 8.966 (+0.85 vs final, +0.15 vs base). The reveal bonus only works with the go-out
  penalty in place. Without GO, C goes out early: it wins more often (45.7%) but scores much worse.
- Pass 25: without the three final-turn fixes (T, FH, FM), 8.197 (+0.083 ± 0.007). Final turns now happen
  in 83% of holes, so final-turn play is worth more than in earlier rounds.

## What the network computes (findings this round)

1. **Final-turn take bias.**
   - On final turns (an opponent is out) the 1-step lookahead with the exact belief is optimal.
   - There, Q0(take) − Q0(draw) = 0.96·(lookahead margin) + 1.45.
   - Round 2's final-turn take bonus was fitted under the biased obs-only belief.
2. **E1's slot readers had broken.**
   - E1's h1 readers (4, 6, 170, 181, 190, 192) read revealed 6/7/8 at slot 1 as face-down in most such
     states, including 36 of 40 sevens. They did the same for 6s at slot 4 (42 of 108).
   - "Slot p face-down" is not linearly separable at the layer-1 FFN input (Fisher gap −1 to −5 sd), where
     E1's detectors sit.
   - It is separable in f (pool-norm output): 0.1–1% errors, on normal and final turns alike if fitted on both.
3. **Own face-down count is not readable from f or h1.** "One left" vs "more" has cross-validated Fisher
   d′ ≈ 3.3, with overlap. Count gates are built from per-slot readers instead.
4. **The 2-step reference undervalues staying.**
   - The reference: going out = exact score; staying = E(after) − G, with G the expected gain of one more draw.
   - The realized benefit of a stay over going out was 6.9 points (trace 1) and 4.8 (trace 5), against 5.7
     and 3.1 predicted.
   - Its implied best go-out penalty was 1.5–2; the games put it at 6.5.
5. **Final-turn stage-1 valuation is noisy.**
   - The net values a hidden placement +1.9 and a flip +1.4 (score units) over replacing a revealed card.
   - Its place-hidden-vs-flip margin moves 0.32 per point of the hidden card's expected value (EU); the ideal
     is 1.2. EU is barely readable from f (R² 0.34): it depends on history.
   - A fixed-average-belief lookahead would have 0.055/hole regret; the net has ~0.15.
   - One clean sub-pattern: discarding held 3/4/5s on the final turn (all such flips in three traces were wrong).
6. **2-on-2.**
   - The net places a held 2 next to a revealed 2 (cancelling −4) in ~12% of opportunities.
   - "Held is a 2" is perfectly separable in f (hardest negatives: Ks).
   - "Slot q shows a 2" is separable only within held-2 states.
7. **Which face-down slot.**
   - The net spreads low cards over columns; the lookahead pairs them (0.39/hole at 1 step, 0.25 at 2 steps).
   - A pairing bonus did nothing in play (pass 17). The net's preference carries future value the references
     miss, as rounds 1–2 suspected.
8. **Spare units.**
   - No h1 unit is dead. 11 change no decision when mean-ablated (130, 26, 50, 160, 127, 147, 39, 184, 175,
     215, 14); about 15 more change 2–9 of ~18k.
   - 53 h2 units were dead; 50 are now used.
9. **F** (round 2's final-turn flag, h1 73) is clean: ≥ 1.24 on final turns, exactly 0 otherwise. All gates
   use it.
10. **E3 is sharply tuned.**
    - E3 gives +1.0 for replacing a revealed 9–Q. Its coverage is only 37–63% at slots 0/3/4/5, and
      f-space refits did not help.
    - Both E3 = 0 and E3 = 2 cost 0.055 (passes 14–15).

## Edits in final.pt (cand_21), all in the MLP head

| edit | what | units |
|---|---|---|
| R | E1's six readers re-fitted as f-space Fisher readers of "slot p face-down", on normal and final states of traces 1–4 plus slot-permuted / opponent-rotated variants. Threshold minimises misses + false flags; scale puts the 5th percentile of face-down scores at 2. E1 pairs' F gate goes from −8 to −20. | h1 4, 6, 170, 181, 190, 192 |
| GO | Go-out penalty. Per slot, conj_p = clip(2·(x_p − 8·Σ_{j≠p} x_j) − 0.5, 0, 1), F gate −40, writing −6.5 to Q[place p] and Q[flip p]. It fires fully on 95–97% of one-face-down normal states, 0.2% of others, never on final turns. | h2 91, 95, 100, 102, 103, 106, 108, 109, 111, 114, 117, 124 |
| δ | Round 1's general reveal penalty removed. E1 pairs now only carry FH's normal-turn give-back (W4 ±1.0), and the flip weight W4[9:15, 63] = 0. | E1 pairs, h2 63 |
| OD3 | Reveal bonus +1.5 to Q[place p] and Q[flip p] when slot p is face-down and ≥ 3 own cards are face-down, on normal turns only. Clip companions of the readers give clipped H_p. Gate per slot: clip(2·Σ_{j≠p} H_j − 3 + 12·(H_p − 1)), F gate −40. It is on in 99.6% of true cases and wrong in 0.3%. | h1 39, 114, 203, 20, 34, 100 (mean-ablated); h2 220, 222, 226, 229, 230, 235, 236, 238, 239, 245, 247, 249 |
| FH | Final-turn hidden-placement penalty 1.0. Ungated clip pairs on the readers write −1.0; E1's normal-turn pairs give it back, so only final turns see it. | h2 135, 137, 138, 142, 147, 149, 150, 158, 162, 168, 169, 172 |
| T | Final-turn take fix: b4[0] −= 1.0 and W4[0, 63] += 1.0 (round 2's final take bonus removed). | b4[0], W4[0, 63] |
| TT | 2-on-2: −2.5 to Q[place p] when the held card is a 2 and p's partner shows a 2. Clipped held-2 reader (h1 130/26), per-slot shows-2 readers fitted on held-2 states (h1 50, 160, 147, 184, 215, 14), mean-ablated old functions. Gate clip(T_q + M·c_h − M − 1), M = 85. | h2 175, 181, 182, 184, 186, 187, 196, 197, 200, 202, 204, 206 |
| FM | Final turns: −2.0 to every flip when the held card is a 3/4/5. Clipped held-3/4/5 reader (final states up-weighted 4×) fires on 76–81% of such final states, falsely on 1–2 held 6s per trace; gated on by F1. | h1 127/161, h2 209/210 |
| E6 off | Round 2's E6 (+1 to flips holding 8–Q) removed. | W4 cols 92, 93 |

Weights changed vs `base.pt`:

| tensor | changed |
|---|---|
| mlp.0.weight | 1,430 |
| mlp.0.bias | 22 |
| mlp.2.weight | 15,158 |
| mlp.2.bias | 229 |
| mlp.4.weight | 831 |
| mlp.4.bias | 1 |
| **total** | **17,671** |

Most of mlp.2.weight is zeroed rows of the 50 newly used dead h2 units and zeroed columns of 16 mean-ablated
h1 units. Kept unchanged from earlier rounds: F, E3 (+1.0), E2 (normal-turn take −1.0).

## Passes (practice seed 1; Δ vs base and pairwise, SE over 24 seatings)

| pass | cand | change | C avg | Δ vs base | pairwise |
|---|---|---|---|---|---|
| 1 | 01 | base copy (for a trace) | 8.8141 | 0 | |
| 2 | 02 | R + GO 1.5 + T | 8.5576 | −0.257 (0.021) | |
| 3 | 03 | GO 2.5 | 8.4777 | −0.336 (0.030) | −0.080 (0.009) vs 2 |
| 4 | 04 | GO 4.0 | 8.4267 | −0.387 (0.037) | −0.051 (0.009) vs 3 |
| 5 | 05 | GO 6.0 | 8.4244 | −0.390 (0.039) | −0.002 (0.007) vs 4 |
| 6 | 06 | GO 5 + joint readers + sharper gate + FH | 8.4121 | −0.402 (0.042) | −0.012 (0.010) vs 5 |
| 7 | 07 | δ 0.75 | 8.3005 | −0.514 (0.030) | −0.112 (0.019) vs 6 |
| 8 | 08 | δ 0 | 8.2609 | −0.553 (0.024) | −0.040 (0.017) vs 7 |
| 9 | 09 | δ 0, GO 6.5 | 8.2440 | −0.570 (0.027) | −0.017 (0.009) vs 8 |
| 10 | 10 | δ −0.75 (bonus), GO net 6.5 | 8.2159 | −0.598 (0.029) | −0.028 (0.012) vs 9 |
| 11 | 11 | δ −1.5, GO net 6.5 | 8.2549 | −0.559 (0.029) | +0.039 (0.013) vs 10 |
| 12 | 12 | + TT 2.5 | 8.2067 | −0.607 (0.028) | −0.009 (0.004) vs 10 |
| 13 | 13 | GO net 8 | 8.2222 | −0.592 (0.034) | +0.016 (0.011) vs 12 |
| 14 | 14 | E3 × 2 | 8.2610 | −0.553 (0.028) | +0.054 (0.010) vs 12 |
| 15 | 15 | E3 off | 8.2620 | −0.552 (0.029) | +0.055 (0.016) vs 12 |
| 16 | 16 | E6 off | 8.1989 | −0.615 (0.031) | −0.008 (0.006) vs 12 |
| 17 | 17 | + PL 1.0 (pair low cards) | 8.2002 | −0.614 (0.029) | −0.007 (0.013) vs 12 |
| 18 | 18 | cand_12 + FM 2 + E6 off | 8.1854 | −0.629 (0.031) | −0.021 (0.006) vs 12 |
| 19 | 19 | bonus 0 at 2 face-down, 0.75 at ≥ 3 | 8.1416 | −0.673 (0.029) | −0.044 (0.006) vs 18 |
| 20 | 20 | penalty 0.75 at 2 face-down | 8.1443 | −0.670 (0.030) | +0.003 (0.008) vs 19 |
| 21 | **21** | bonus 1.5 at ≥ 3 | **8.1147** | **−0.699 (0.031)** | −0.027 (0.011) vs 19 |
| 22 | 22 | bonus 2.5 at ≥ 3 | 8.1160 | −0.698 (0.030) | +0.001 (0.003) vs 21 |
| 23 | 23 | ablation: final without GO | 8.9661 | +0.152 (0.069) | +0.851 (0.083) vs 21 |
| 24 | 24 | final with GO 5.0 | 8.1156 | −0.698 (0.026) | +0.001 (0.012) vs 21 (GO 5–6.5 is a plateau) |
| 25 | 25 | ablation: final without T, FH, FM (final-turn fixes) | 8.1974 | −0.617 (0.031) | +0.083 (0.007) vs 21 |

In passes 10–13, "GO net" = GO weight minus the reveal bonus at one face-down card.

Confirms (seed 2): cand_12 8.2454, cand_18 8.2321, cand_21 8.1720. All three practice gains held up on the
confirm seed within one practice SE.

## Did not work, or was dropped

- **The 2-step reference as a tuner** for reveal and go-out decisions. It undervalues staying, so those
  parameters were set by passes.
- **PL, a pair-low-cards bonus** (pass 17).
  - It used 14 repurposed h1 units.
  - Offline it looked like −0.11/hole (1-step and 2-step agreed); in play it was −0.007 ± 0.013.
- **Larger E3, no E3** (passes 14–15): both cost 0.055.
- **Uniform final-turn offsets** beyond FH = 1.0, and final take tweaks: flat offline, not built.
- **Race-aware go-out** (not built). Stays when an opponent still has ≥ 2 face-down cards realized less
  benefit, but the separable part is ~0.01/hole.
- **E3 repair via f-space readers** (not built). It did not raise the 37–63% coverage.

## Caveats

- **References.** The exact 1-step lookahead is optimal only on final turns. Every normal-turn parameter (GO,
  reveal bonus by count, E3) was set by passes, not by an offline reference.
- **Selection.** Sweeping the bonus and GO over passes 3–22 picks the best of several close candidates, so
  final's practice number is optimistic by about one pairwise SE (0.01–0.02). The confirm (−0.665) is the
  honest number; its practice-to-confirm drop (0.034) is in line with that.
- **Fits.** Readers were fitted on practice-seed traces (all seats) plus permuted variants. The confirm seed
  reproduced every confirmed gain.
- **Q scale.** Q is no longer calibrated in points: GO (−6.5) and the bonuses are decision devices. Resuming
  TD training from this net would need them removed or re-expressed.

## What I would try next

1. **A count flag built in attention.** One attention head on own tokens (F's sink trick: sink weight
   1/(1+k)) would replace the reader conjunctions and free ~12 h1 units. With it:
   - a GO penalty that depends on stay cost and opponent progress;
   - bonus and penalty per count (only ≥ 3 vs 2 vs 1 was tried).
2. **A real final-turn evaluator** (~0.1/hole; C now has a final turn in 83% of holes). It needs per-slot
   card-value channels the pool lacks, probably a layer-0 head copying each slot's partner and the held card.
3. **Which revealed card to give up when staying or holding** (~0.08/hole of 2-step regret, diffuse).
4. **E3 coverage at slots 0/3/4/5.**

## Files

| file | purpose |
|---|---|
| `t.py` | toolkit: model I/O, forward with internals, light forward, trace parser, exact-belief lookahead, G |
| `build.py` | all edits: `build` (R, GO, δ, FH, T), `twotwo_hook` (TT), `finalmid_hook` (FM), `od3_hook` (OD3), `pairlow_hook` (PL, dropped), `scale_e3`, `scale_e6`, `chain` |
| `evalc.py`, `an*.py` | offline comparisons and the analyses quoted above |
| `seats.py`, `pair.py` | per-seat and pairwise comparisons of passes and confirms |
| `cand_*_meta.json` | build parameters and unit indices of each candidate |
| `x_*.pt` | unscored drafts |
| `scratch_log.md` | running notes |

No gradient descent, optimizer, training code, engine, simulator, opponent or seat-cycling code was run.
- The only project import is `make_model` from `src.tournament`, used to construct the network.
- The lookahead formula was re-implemented in numpy after reading `src/bayes_optimal.py`.
- The network was evaluated only on trace or hand-built inputs, and played only through `scripts.wsp_pass`.
