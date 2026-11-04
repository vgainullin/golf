# WSP opus_b25: hand-editing the Exp 14 champion

**Result:** `final.pt` = `cand_05.pt` = `base.pt` with **one weight changed**: `mlp.4.bias[1]` (the Q of
action 1, "draw from deck") raised by **+1.5**. On the practice benchmark the score went from 9.523 to
9.199 per hole, **−0.324 ± 0.063** (paired SE over 24 seatings). Win rate went from 0.374 to 0.426.
I used 9 of the 25 passes.

## What the network computes (as far as it matters here)

- Architecture: rank+suit+position embeddings → 2-layer pre-norm transformer (30 tokens) → mean-pool
  of tokens 0–6 (own grid + held card) → LayerNorm → MLP 65→256→256→16. The deck-remaining
  scalar enters `mlp.0` directly (column 64).
- **Stage 0 (take/draw).** Swept on hand-built boards: Q0−Q1 is strongly positive for a discarded
  2/K/A (+2.5 to +6) and for column matches with a revealed card (+7 to +9). It is marginally positive
  for 3/4 (+0.1 to +1.3) and slightly negative for 5 and above (−0.2 to −1.8).
- **Stage 1, place rows Q2..Q7.** Q(place at k) tracks the value of the revealed card at slot k with a
  slope of about 1.0 on its own slot and about 0.15 cross-talk to other slots (measured with
  contrastive variants of trace states). Placing on a hidden slot carries a large learned offset over
  replacing a revealed card. In the trace it put held 2/3/4/K/A on hidden slots about 90% of the time.
- **Last hidden layer (mlp.2 output):** 78–79 of 256 units are dead on everything I fed it (trace
  states plus synthetic boards). In `mlp.0`, only 4 units are dead (4, 170, 181, 190).
- **Where "slot k is revealed" can be read linearly:** best in the pooled 64-d vector (probe SNR
  about 4.7), worse in h1 (about 3) and in h2 (about 1.6).

## Key diagnostic: the stage-0 head is inconsistent with the net's own stage-1 head

I computed what the net's own stage-1 head implies for each stage-0 option (`bellman.py`):
- implied take = max over place actions, with the discard as the held card;
- implied draw = average over unseen cards of the max over valid stage-1 actions.

On 377 trace states the stage-0 head overvalues take by 0.82 and undervalues draw by 1.07. The net
implied take−draw is 1.6–1.9 lower than the head's own take−draw (median 1.56). For the marginal
discards (3/4/5/K/A) the gap is 0.7–0.9; for a 2 it is 0.4. The gap is flat across deck size and
hidden-card count. So a single constant on the draw logit is the right shape of fix. It only ever
affects stage 0, because action 1 is invalid in stage 1.

## Edits and passes

| pass | candidate | edit | score/hole | vs base (paired) |
|---|---|---|---|---|
| 1 | base | none | 9.523 | — |
| 2 | cand_02 | stage-1 bonus `0.8·max(0, value(slot k)−2)` on revealed slots, via probe → dead units | 9.636 | +0.113 ± 0.065 |
| 3 | cand_03 | `mlp.4.bias[1]` += 0.5 | 9.291 | −0.232 ± 0.075 |
| 4 | cand_04 | `mlp.4.bias[1]` += 1.0 | 9.231 | −0.292 ± 0.081 |
| 5 | **cand_05** | `mlp.4.bias[1]` += 1.5 | **9.199** | **−0.324 ± 0.063** |
| 6 | cand_06 | `mlp.4.bias[1]` += 2.5 | 9.299 | −0.224 ± 0.075 |
| 7 | cand_07 | cand_05 + flip biases `mlp.4.bias[9..14]` += 0.5 | 9.222 | −0.301 ± 0.061 (vs pass 5: +0.023 ± 0.063) |
| 8 | cand_08 | cand_07 + revealed-slot penalty of about −2 on place rows, via probe → dead units | 10.095 | +0.573 ± 0.087 |
| 9 | cand_09 | `mlp.4.bias[1]` += 2.0 | 9.200 | −0.323 ± 0.071 |

A quadratic fit over draw biases {0, 0.5, 1, 1.5, 2, 2.5} has its vertex at +1.55 (predicted 9.18),
so I kept +1.5. Behaviour change at +1.5 (trace 5 vs trace 1): stage-0 take rate fell from 0.41 to
0.25. It now passes on most discarded 3/4/5 (3: 35/36 → 7/30, 5: 20/33 → 6/52) and still takes
2/K/A and column matches.

### How the probe edits (cand_02, cand_08) were built, and why they failed

- **Probe.** A closed-form ridge probe on the pooled vector, or on h1, read either "slot k revealed"
  (balanced across card ranks) or the value target. I used it only as a readout, never fitted to Q.
- **Routing.** The probe was written into a revived dead `mlp.0` unit, which feeds a dead `mlp.2`
  unit, which feeds only `mlp.4.weight[2+k]`. The revived units' other outgoing weights were zeroed,
  so stage 0 and the flip actions are unchanged (max change 0.0).
- **Size of the change.** cand_02 changed `mlp.0` rows 4/170/181/190, the matching bias entries, 6
  rows and 6 bias entries of `mlp.2`, 6 columns of `mlp.2`, 6 columns of `mlp.4`, and 6 entries of
  `mlp.4.bias`.
- **Motivation.** Cand_02 came from my own 1-step expected-score evaluator (`ev.py`). It said the net
  under-replaces revealed high cards (stage-1 regret about 0.33/decision). It ignores tempo: holes
  got longer, 16.2 → 17.6 decisions, and the score got worse.
- **Cand_08 went the opposite way.** It came from a stage-1 self-consistency check (`s1bell.py`,
  n = 55 states), which said revealed-slot placements are overvalued by about 2. That hurt badly.
  Many column matches are completed by replacing a revealed card (e.g. replacing an A to pair a 6),
  and the stage-0 head still takes those discards expecting the match.
- **Both directions hurt.** The probe readout carries ±1.25 of per-state noise on top of its mean,
  and that alone perturbs placement choices. Lesson: on this net, noisy feature edits to the
  placement rows cost more than they recover. Noise-free bias edits worked.

## Things I checked and did not act on

- **Stage-1 flip vs place on a hidden slot:** the stage-1 self-consistency check gave flip
  +0.53 ± 0.18. Pass 7 tested it: no measurable effect.
- **Slot-contrast amplification** (`w_{2+k}` += α(w_{2+k} − mean)): the base is already at the
  surrogate's optimum, so I did not spend a pass on it.
- **Going out with one hidden card:** about 0.1/hole of 1-step regret, partly offset by the opposite
  error. A fix would need a new feature.
- **Final-turn decisions:** fewer than one per hole and already fairly good.

## Caveats

- Paired differences between two candidates have SE 0.06–0.08, because decks drift once play
  differs. Effects under about 0.1 (e.g. 1.0 vs 1.5 vs 2.0 draw bias) are not resolved. I chose
  between them with the fitted curve.
- The opponents' scores move too (L went from 9.04 to 9.2–9.3 with the edit). The held-out games may
  shift the absolute numbers.
- Trace rows for finished games (all six cards revealed, held card 52) are still fed to the net by
  the harness. I filtered them out of every analysis.

## What I'd try next

- A noise-free, rank-aware stage-0 correction: the self-consistency gap is about 0.4 for a 2 and
  2–3.8 for high cards. It could be built from the discard token's rank embedding (token 7) through
  a dead unit, if a probe for "discard rank" turns out to be near-exact (rank is a direct input, so
  it may be).
- The stage-1 offset of hidden over revealed placement, but gated on "no column match completes
  here", so match completions stay intact. That needs a much cleaner feature than my linear probes,
  e.g. reading the slot-k token before pooling, which this architecture does not expose to the head.
- Final-turn-specific play, where 1-step EV is exact. It needs a "some opponent has all cards up"
  feature.

## Files

`wsp.py` (load/save/forward helpers), `ev.py` (1-step expected-score evaluator), `synth.py`
(hand-built random observations), `bellman.py` and `s1bell.py` (self-consistency checks), `dirs.py`,
`probe.py`, `feats.py` (direction and probe analysis), `edit_bias.py`, `edit_rev2.py`, `edit_rev3.py`
(edit scripts), `cmp.py` (paired comparison between passes), `dead.pt`, `dirs_base.pt` (analysis
artifacts).
