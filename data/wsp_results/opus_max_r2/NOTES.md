# Weight-space programming, round 2 (run opus_max_r2): final-turn flag, high-card bonuses

**Result:** `final.pt` = `cand_15.pt`, practice **8.818** avg score/hole vs **8.949** for `base.pt`. The paired
Δ is **−0.132** (SE 0.067 over 24 seatings) and the win rate rose from 41.7% to 43.4%. Practice seed 2,
12 of 25 passes used.

Practice cannot tell the four best candidates apart: cand_05, cand_08, cand_14 and cand_15 all score
8.814–8.818. Their pairwise SE is 0.055–0.07. Two ablations each move the score by about −0.08, and the
two effects add up:
- E3 alone: −0.079 ± 0.061.
- The final-turn circuit plus its take bonus: −0.075 ± 0.061.

`final.pt` adds three changes to `base.pt` (base.pt already holds the prior session's E1/E2):
- **F, a final-turn flag** ("some opponent has all six cards face-up", i.e. this is C's last turn). It is
  hand-built across both attention layers and the MLP. When it fires:
  - E1's reveal penalty is off (all of it);
  - E2's take bias is off (+1.0 to take).
- **E3:** +1.0 to Q[place at p] when own slot p shows a revealed 9/T/J/Q.
- **E6:** +1.0 to every flip action when the held card is 8/9/T/J/Q.

32,565 of 189,552 weights differ from `base.pt`. The tensor table is below.

## What the network computes (new findings this session)

### 1. The final turn is invisible, so E1 misfires there
- Nothing in the net encodes "an opponent is fully revealed". E1 therefore charges its δ = 1.5 on
  C's final turn too.
- On the final turn the 1-step lookahead is exact up to the belief, because there is no future.
- The original champion's final-turn reveal premium was small (+0.4, per the prior notes). With E1 it
  is negative: about −0.7 vs the lookahead in my probe (`probe_hr.py`). The net replaced a revealed
  card when placing the held card face-down was clearly better.
- Seat-0 trace, base: final-turn stage-1 lookahead regret was 11.6 over 29 decisions. Removing E1 on
  final turns brings it to 2.7.

### 2. Layer-0 tokens can be enumerated exactly, so position, group and face-down are cleanly readable there
- The layer-0 input is a pure function of (card, position): 1,382 possible tokens.
- Weighted least-squares readouts at the layer-0 norm1 output:
  - "card of opponent j": 0.87–1.16 in-group, ≤ 0.0006 everywhere else;
  - "position 8 / 15 / 22": ≤ 0.0002 elsewhere;
  - "face-down" and "stage token": exact.
- At the layer-1 input none of the position classes (own / opponent card / holding / discard) is
  linearly separable. Fisher overlap is 2–3σ. Any position logic therefore has to happen in layer 0.

### 3. Head costs (mean-ablation on 1,002 trace decisions)

| head | decisions changed | own-Q regret per decision | note |
|---|---|---|---|
| L0H3, all queries | 7.9% | 0.0135 | |
| L0H3, own-token outputs only | 6.9% | — | most of the cost |
| L0H3, opponent-card outputs only | 3.5% | 0.003 | |
| L1H3 | 3.4% | 0.004 | cheapest layer-1 head |
| L1H0–2 | 6–43% | — | |

For L0H3's own-token queries, the 3 weakest q/k dimensions (by SVD of the kept-query × key logit
structure) carry 0.05% of the logit mass. They can be freed at almost no cost.

### 4. "A high card at slot p" is a product feature
- It is not linearly readable even at the layer-0 input. This confirms the prior session's "where a
  high card sits is nearly invisible".
- At the layer-1 FFN input, a per-slot Fisher direction finds 9/T/J/Q at slot p:
  - slots 0–2: TPR 0.91–0.99;
  - slots 3–5: TPR 0.83–0.88;
  - both at a 0.05% false-positive rate.
- `probe_rank.py`, Q[place p] − Q[flip] by the rank at slot p, base minus lookahead:
  - mid cards 3–7: −0.4 to −0.85;
  - 9: −1.4;
  - T/J/Q: −1.75 to −1.9.
- After E3 the curve is flat from 3 to Q (−0.4 to −0.9). Replacing high cards is no longer
  undervalued relative to mid cards.

### 5. The held card
- `probe_held.py`, holding T/J/Q: the net values placing it face-down (vs flipping that slot) +1.74
  higher than the lookahead does. 2–8 are within ±0.6. After E6 the gap is +0.8.
- At the holding token (layer-1 FFN input):
  - "held is a 2" is cleanly separable;
  - "held 8..Q" is detected with TPR 0.94 at 0.02% false positives; the hardest negatives are held 7s.
- In the pooled LayerNorm features (f), held-rank classes overlap by about 6%.

### 6. Count readouts in f
- Own face-down count: RMSE 0.20 (linear).
- Minimum opponent face-down count: RMSE 0.33, R² 0.97.
- At stage 0, "discard is 6..Q" is almost separable (0.1–0.2% overlap).

### 7. The obs-only lookahead is biased, which matters for diagnosis
- The obs-only belief (the prior session's `wsp_tools.belief`, also used by my `oracle.py`) counts
  buried discards as unseen. Players discard high cards, so late in a hole it overestimates the mean
  unseen card.
- Bias against C's *own* observation history (`hist_la.py`): +0.46 at one own face-down card, +0.37
  at two, about 0 early. L's tracker sees every discard, roughly two more per round than C, so the
  real bias is larger.
- This bias pushes obs-only diagnostics toward revealing (placing on face-down slots) and toward taking.
- Several "errors" that look large against the obs-only lookahead disappear under the history belief,
  or under the history belief with 4–8 extra buried high cards removed:
  - the late-hole take preference (E4 below);
  - the last-card place-vs-flip bias.

### 8. δ, the reveal penalty, has a sharp local optimum at 1.5. The myopic lookahead cannot judge it.
- Uniform δ = 1.0: +0.131 ± 0.074, worse than δ = 1.5.
- δ = 1.0 while an opponent has ≤2 face-down cards and 2.0 otherwise: **+0.27 ± 0.085**, worse.
- δ = 2.0 near the end and 1.5 otherwise: −0.010 ± 0.055, neutral.
- The lookahead always prefers less penalty (more revealing), with either belief. Its regret said
  every one of these variants was an improvement.

## Edits in final.pt

### F: final-turn flag
Scripts: `build_ft.py` (`build_all(variant="uniform", amp=20, c=10, amp2=6, orth=True)`) and `edit_mlp.py`.

**Stage 1. Layer-0 head 3, dual use.**
- Only the queries of tokens 8, 15 and 22 (the first card of each opponent) get a new job. Every other
  query keeps the old head.
- Q/K change of basis:
  - QR plus an SVD of the kept-query × key logit structure.
  - This is an exact basis change (q′·k′ = q·k).
  - The 3 weakest dimensions are freed (0.05% of logit mass).
- New dimension j:
  - query = "I am token 8+7j" (readout leaks ≤ 0.0002 onto other tokens);
  - key = 32·[card of opponent j] + 36·[face-down] + 55·[stage token].
- The old logits of the three repurposed queries are suppressed by a weighted refit. Kept queries
  change by RMS 0.05 logits.
- Result for each repurposed query:
  - it lands on its opponent's face-down cards if any exist;
  - otherwise it lands on the stage token, which acts as a sink.
- Values: the value basis is PCA-rotated, and the least-used dimension (variance 0.0006) is replaced by
  "is the stage token".
- Token 8+7j therefore receives 20·d exactly when opponent j is fully revealed. d is the zero-mean
  direction with the best Fisher SNR at the layer-1 input.
- Verified: sink weight 1.0000 on every fully-revealed case and ≤ 0.02 otherwise.

**Stage 2. Layer-1 head 3, repurposed.**
- Constant query.
- Key = 10·(flag readout − 0.5). The flag readout is a Fisher discriminant: flagged tokens 0.78–1.2,
  all other tokens ≤ 0.19 (std 0.03).
- On normal turns, attention is about uniform. 15 value dimensions keep the old head's values
  (PCA-rotated), so own tokens get the uniform average of the old values.
- The 16th value dimension carries the flag readout and is written as 6·d2.
- d2 is Fisher-quiet in pooled-LayerNorm space and orthogonal to E1's six slot channels.

**Stage 3. MLP.**
- Dead h1 unit 73 reads F with a final-vs-normal Fisher direction, made blind to E1's channels.
- Every live h1 row has the F direction projected out, with bias compensation.
- Gating:
  - E1's 12 h2 clip pairs (0, 5, 6, 19, 21, 22, 23, 31, 43, 45, 54, 58) get −8·F1;
  - the flip penalty moves from b4[9:15] into h2 unit 63 = relu(1 − 8·F1), with W4[9:15, 63] = −1.5.
- Final-turn take bonus: W4[0, 63] = −1.0 and b4[0] += 1.0, which switches E2 off when F fires.

Checks:
- F1 ≥ 0.82 on every synthetic final state and exactly 0 on every normal state (three fresh seeds over
  12 traces).
- Final-turn regret against the history-aware lookahead, over 7 traces:
  - stage 1: 54.3 → 24.7;
  - stage 0: 65.8 → 13.0.
- Collateral: the whole of final.pt changes 405 of 6,935 decisions on 7 traces vs base. Most are
  intended (E3/E6/final turns); the rest are near-ties from the repurposed heads.
- Final-turn settings swept under three belief shifts:
  - take bonus 1.0 is best under every shift;
  - a 0.5 residual reveal penalty would be slightly better than 0 (see "Did not work").

### E3: localized high-card replacement (`build_high.py`, β = 1.0, gain 2)
- Detectors, per slot p:
  - two rarely used layer-1 FFN units (250, 49, 168, 90, 37, 21, 103, 22, 181, 26, 71, 130), chosen
    because zeroing each changed 0 trace decisions;
  - each pair is a clip pair on the slot-p Fisher direction for 9/T/J/Q;
  - thresholds: the 99.95th percentile of negatives and the 10th percentile of positives.
- Each detector writes **−2·o_p**, where o_p is E1's slot-p channel vector. E1 writes +o_p for
  "face-down", so the channel takes {+1, 0, −2} and E1's readers are unaffected.
- Readers: rarely used h1 units 24, 230, 49, 84, 202 and 174, mean-compensated, read the negative side.
  Dead h2 pairs (66, 68, 75, 76, 78, 79, 80, 83, 85, 86, 87, 88) clip to [0, 1] and add +1.0 to
  Q[place p].
- Read rates on fresh states:
  - slots 0–2: 85–98%;
  - slots 3–5: 57–76%;
  - false positives about 0.1%, mostly 8s at slot 0.
- Normal-turn stage-1 regret against the history-aware lookahead: 363 → 305 over 4 traces.

### E6: held-high flip boost (`build_held.py`, β = 1.0)
- Detector: layer-1 FFN units 110 and 25, a clip pair on "the holding token shows 8..Q". TPR 0.94,
  false-positive rate 0.02%.
- It writes a new pooled channel. The channel is quiet and orthogonal, in f, to E1's channels and to
  F's reader.
- Live h1 rows have the channel projected out (bias-compensated).
- Reader: h1 unit 54, then h2 pair 92/93, then +1.0 on actions 9–14.
- In held-high states the flips gain +0.73 relative to placements. The channel also lowers every action
  by about 0.17, through the LayerNorm.
- Stage-1 regret against the history-aware lookahead: 450 → 415 over 6 traces. The direction holds
  under every belief shift.

### Weights changed (final.pt vs base.pt): 32,565 of 189,552

| tensor | changed | what |
|---|---|---|
| transformer.layers.0.self_attn.in_proj_weight / bias | 3,072 / 48 | head-3 q, k, v rows |
| transformer.layers.0.self_attn.out_proj.weight / bias | 1,024 / 64 | head-3 columns; mean-compensation of the dropped dimension |
| transformer.layers.1.self_attn.in_proj_weight / bias | 3,072 / 48 | head 3 |
| transformer.layers.1.self_attn.out_proj.weight / bias | 1,024 / 64 | head 3 |
| transformer.layers.1.linear1.weight / bias | 896 / 14 | 14 FFN units (E3: 12, E6: 2) |
| transformer.layers.1.linear2.weight | 896 | their output columns |
| mlp.0.weight / bias | 16,008 / 250 | projections in every live row (F and E6 channels); 8 reader rows |
| mlp.2.weight / bias | 5,594 / 244 | gating weights, new clip pairs, mean-compensations |
| mlp.4.weight / bias | 240 / 7 | E3/E6/F outputs; b4[0] and b4[9:15] |

## Passes (practice seed 2; 100 games × 24 seatings)

| pass | candidate | content | C avg | Δ vs base (SE) | seat 0 / 1 / 2 / 3 Δ |
|---|---|---|---|---|---|
| 1 | base | — | 8.949 | — | |
| 2 | cand_02 | F (first version: mean-ablated L1H3, F not orthogonal to the channels) | 8.863 | −0.086 (0.075) | +0.04 / −0.24 / −0.20 / +0.06 |
| 3 | cand_03 | cand_02 + δ 2.0 + final take bonus | 8.909 | −0.040 (0.084) | −0.12 / −0.12 / −0.27 / +0.34 |
| 4 | cand_05 | F (orthogonal, mean L1H3) + E3 + final take bonus | 8.818 | −0.132 (0.072) | −0.33 / −0.14 / −0.06 / +0.01 |
| 5 | cand_06 | cand_05 + E4 (late take bonus) | 8.825 | −0.124 (0.091) | |
| 6 | cand_07 | cand_06 + δ 1.0 near the end / 2.0 otherwise | **9.098** | **+0.149** (0.085) | −0.06 / +0.14 / +0.04 / +0.47 |
| 7 | cand_08 | cand_06 + δ 2.0 near the end / 1.5 otherwise | 8.815 | −0.134 (0.087) | |
| 8 | cand_10 | cand_05 with δ 1.0 | 8.948 | −0.001 (0.065) | |
| 9 | cand_12 | cand_05 without E3 | 8.897 | −0.052 (0.070) | |
| 10 | cand_13 | base + E3 only | 8.892 | −0.057 (0.080) | |
| 11 | cand_14 | cand_05 + E6 | 8.814 | −0.136 (0.068) | |
| 12 | cand_15 = **final** | F with the uniform-attention L1H3 + E3 + final take bonus + E6 | **8.818** | **−0.132** (0.067) | −0.11 / −0.50 / −0.06 / +0.14 |

Pairwise comparisons (`pair.py`):

| comparison | Δ (SE) | reading |
|---|---|---|
| cand_05 vs cand_12 | −0.079 (0.061) | E3's effect |
| cand_05 vs cand_13 | −0.075 (0.061) | F + take bonus |
| cand_07 vs cand_06 | +0.272 (0.085) | race split hurts |
| cand_10 vs cand_05 | +0.131 (0.074) | δ 1.0 hurts |
| cand_14 vs cand_05 | −0.004 (0.055) | E6: no visible effect |
| cand_15 vs cand_14 | +0.004 (0.072) | the two L1H3 variants are equivalent |

I chose cand_15 over cand_14 for two reasons:
- it changes fewer base decisions on 7 traces (405 vs 507 of 6,935), so it stays closer to the base
  on normal turns;
- its total regret against the history-aware lookahead over those traces is lower (575 vs 596).

Their F readers are equally reliable: minimum F1 on fresh final states is 0.82–1.06 for cand_15 and
0.88–1.23 for cand_14, and 0 on normal states for both.

Seat 3 is never clearly helped: +0.01 to +0.34 in most candidates, with a per-seat SE of about 0.15.

## Did not work, or was dropped
- **E4, a late-hole take bonus.** +1.0 to take at ≤2 own face-down cards, vetoed when the discard is
  6..Q or on the final turn (`build_late.py`). It was built because the obs-only lookahead wanted to
  take discarded 3/4/5s late (regret 54 → 27).
  - Under the history belief its gain halves.
  - Once 4–8 buried high cards are removed from the belief, it becomes neutral, then harmful.
  - Practice: +0.008 vs cand_05. Dropped.
- **Race-aware δ (E5, `build_race.py`).** Uses a min-opponent-face-down readout from f. Built twice:
  - 1.0 near / 2.0 far: +0.27, the worst result of the session;
  - 2.0 near / 1.5 far: neutral.
  - Dropped. Uniform δ = 2.0 (pass 3) and δ = 1.0 (pass 8) were not better either.
- **A residual 0.5 reveal penalty on final turns (`build_resid.py`).** The offline sweep favoured it
  under all three belief shifts. In the network it changed none of the intended decisions:
  - on final turns the F broadcast reaches the layer-1 FFN;
  - E1's slot detectors then misfire (6.5% false "face-down" per slot vs 1.1% on normal turns);
  - for example a revealed 8 was read as face-down.
  - Isolating the detectors from the broadcast fixed the misfires but degraded normal turns (6% of
    states moved by more than 0.3 in Q). Not shipped. E1 is gated off on final turns, so the misfires
    are harmless in final.pt.
- **First F designs.**
  - A one-head design at layer 1 failed: groups are not separable there.
  - Dual-use L1H3 failed: the old logits outweighed the flag (attention on the flag was 0 in some
    states) and the key noise moved about 10% of attention mass on normal turns.
  - Uniform attention with a summed flag failed: the noise of 30 tokens adds up (finals vs normals
    overlapped).
  - A soft key with a large value was unreliable.
  - Isolating the MLP along the mean final-vs-normal shift added about 40 changed normal-turn
    decisions (80 in total). Isolating along the designed quiet direction added none.
  - After E3, the F reader had to be made orthogonal to E1's channels: E3's −2 writes had hidden F,
    which the first version read partly off those channels.
- **2-on-2 column fix (not built).** Placing a held 2 next to a revealed 2 cancels −4. The net picks
  such a slot in 12% of opportunities vs 5% for the lookahead. That is worth about 0.05/hole across 4
  traces, and it needs per-slot ANDs that f cannot provide.
- **Last-card place-vs-flip and stall fixes (not built).** The apparent bias (−0.9 to −1.3 for mid held
  cards at one face-down card) is about the size of the belief bias. Costly stalls happen 0–3 times per
  trace.
- **"Which face-down slot" (not built).** The lookahead closes a column next to a revealed low card; the
  net keeps two columns half-open. This is the largest lookahead-regret class (~0.3/hole), but the
  1-step difference is small and the future value of open columns is real. I left it, as the prior
  session did.
- **Refitting E3 on 10 traces instead of 2:** no better (slots 3–5 are limited by separability).

## Caveats
- **Diagnostics.**
  - The prior session's obs-only lookahead is biased toward revealing and taking, so it overstates
    those "errors".
  - Every edit in final.pt was rechecked against the history-aware lookahead and with extra high
    cards removed.
  - Only δ-type questions, the value of keeping cards face-down, were left to the passes, and the
    lookahead was wrong on them.
- **Fits.**
  - Detector directions, thresholds and readouts are closed-form (least squares, Fisher, SVD/PCA,
    eigenvectors). They were fitted on practice-trace states, which are seat 0 only, plus
    slot-permuted, opponent-rotated and synthetic "opponent fully revealed" variants.
  - The pass results cover all seats.
- **Noise.** Single passes resolve only about ±0.15. Each component was kept because offline analysis
  gave a robust direction, not because a pass proved it.

## What I would try next
1. **Make E1's slot detectors blind to the F broadcast.** Refit their Fisher directions inside the
   subspace orthogonal to the broadcast, using normal and final states together. Then add the 0.5
   final-turn residual on face-down placements (worth about −15 regret over 500 holes).
2. **A belief-mean ("card counting") channel.**
   - In the mean-ablated L1H3 variant, value dimensions 0–14 are free, and attention is uniform on
     normal turns.
   - Values like "visible card score" and "revealed" would give the MLP the obs-only unseen mean.
   - The net still has to learn how buried discards bias it, so this only pays with a correction
     on top.
3. **Extend E3 to 7/8** at β ≈ 0.4: their remaining deficit is about 0.4 more than mid cards.
4. **Build a non-myopic reference before any further δ work.** Every δ-related lookahead signal this
   session pointed the wrong way.
5. **Replace the obs-only belief in future diagnostics** with the history belief (`hist_la.py`) plus a
   high-card correction.

## Files
| file | purpose |
|---|---|
| `build_ft.py` | F circuit: layer-0 tokens, stages 1–3, `fisher_dir`, `build_all` |
| `edit_mlp.py` | δ and final-turn take-bonus edits in the MLP head |
| `build_high.py` | E3 |
| `build_held.py` | E6 |
| `build_late.py`, `build_race.py`, `build_resid.py` | dropped edits (E4, E5, final residual) |
| `heads.py` | manual forward pass with per-head masks and mean-ablation |
| `ftdata.py` | trace states plus slot-permuted, opponent-rotated and opponent-revealed variants |
| `an_trace.py`, `oracle.py`, `cmp.py` | trace audit and obs-only lookahead regret; offline what-if corrections |
| `hist_la.py`, `cmp_hist.py` | history-aware lookahead reference and candidate scoring |
| `probe_rank.py`, `probe_held.py`, `probe_hr.py`, `probe_22.py` | calibration probes (revealed rank, held rank, reveal vs replace, 2-on-2) |
| `seats.py`, `pair.py` | per-seat and pairwise pass comparisons |
| `iso.py`, `fix_freader.py`, `diag*.py`, `test_ft.py`, `sweep_f.py` | isolation experiments and build diagnostics |
| `cand_*_meta.json` | unit indices of each built candidate |
| `wsp_tools.py`, `trace_util.py`, `net_internals.py`, `s0_consistency.py` | copied from `prior/` |

No gradient descent, optimizer, training code, engine, simulator, opponent or seat-cycling code was
run. The network was evaluated only on hand-built or trace-derived inputs, and played only through
`scripts.wsp_pass`.
