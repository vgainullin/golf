# Experiments: Improving Beyond Heuristic

## Game: 6-Card Golf

4 players, 9 holes, French deck (no jokers). Each player has 6 cards in a 2x3 grid, initially face-down. Two cards are revealed at deal.

**Scoring:** 2=-2, 3-9=face value, 10/J/Q=10, K=0, A=1. Column match (same rank in both rows of a column) zeroes both cards. Lower is better.

**Turn structure (2 stages):**
- Stage 0: Take the face-up discard pile card, or draw from the deck.
- Stage 1: Place the held card at any grid position (replacing that card, which goes to discard), or discard it and flip one face-down card.

Game ends when any player has all 6 cards revealed, then each other player gets one final turn.

**Heuristic strategy:**
- S0: Take face card if score < 4 or rank matches a revealed card (column match opportunity). Else draw.
- S1: Only considers placing at unrevealed positions. Place if improvement >= 4, or if it doesn't make score worse (information gain). Else discard + flip.
- Weakness: never swaps revealed cards to create column matches (theoretical 9.6 vs actual 14.0).

## Goal

Beat the heuristic baseline (~14.0/hole) using RL on top of an imitation-learned DQN.

| Baseline | Score (avg/hole) |
|----------|-----------------|
| Heuristic | ~14.0 |
| Imitation DQN | ~14.0 (matches heuristic) |
| Tournament RL (from scratch) | ~16.3 (never beats heuristic) |
| Improved heuristic (theoretical) | 9.6 |

Eval config: [DQN, Random, Heuristic, Random], 9 holes.

## Why tournament RL failed

1. Epsilon-greedy can't discover strategic improvements
2. Q-values drift with no anchor (catastrophic forgetting)
3. Co-adaptation when evaluating DQN vs DQN

---

## Experiment 1: Residual DQN with alpha scaling (2026-03-06)

**Commit:** `src/dqn_offline.py` (ResidualDQN), `src/residual_dqn.py`

**Approach:** `Q_total = Q_base + alpha * Q_residual`. Q_base is frozen imitation model. Q_residual is a fresh GolfDQNv2Shallow. Alpha scheduled 0.1 -> 1.0. Both DQN loss and margin loss computed on Q_total. Boltzmann exploration. Self-play [DQN, H, H, H]. Demo buffer from heuristic rollouts.

**Bug:** Alpha inside forward scales gradients by alpha. At alpha=0.1, effective LR = 1e-5. The residual can't flip action rankings -- needs to overcome Q_base logit gaps divided by alpha.

**Result:** DQN loss 13.5 -> 4.3 but eval flat at ~14.0. Best: **13.949** at iter 40 (alpha=0.45). Worse at alpha=1.0 (14.08) because residual was trained under damped gradients then given full weight.

---

## Experiment 2: Alpha removed from forward (2026-03-06)

**Fix:** Forward changed to `Q_base + Q_residual` (no alpha). Residual gets full gradients.

**Remaining bug:** DQN loss and margin loss still computed on Q_combined. Q_base is cross-entropy logits, not Q-values. Residual spends capacity correcting logit scale rather than finding better actions. Margin loss trivially satisfied (~0.001) since Q_base already ranks expert actions highest.

**Result:** Best **13.893** at iter 100. DQN loss 12 -> 3.7. Marginal improvement.

---

## Experiment 3: Losses on Q_residual alone (2026-03-06)

**Fix:** DQN loss computed on Q_res alone (TD targets from Q_res_target only). Margin loss on Q_res alone. Q_base only used for action selection in Double DQN next-action choice.

**Observations:**
- Margin loss now provides actual gradient: 0.6 -> 0.15 -> 0.78
- Margin loss *increases* after iter 50: Q_res learns different action preferences
- But eval flat at ~14.0 because `evaluate_model` uses argmax(Q_base + Q_res) and Q_base dominates

**Verification** (Q_res alone vs combined at iter 50 checkpoint):

| Model | Score |
|-------|-------|
| Q_base alone | 14.0 (heuristic) |
| Q_base + Q_res | 13.9 |
| Q_res alone | 18.2 (bad) |

Q_res hasn't learned a useful standalone Q-function.

**Result:** Best **13.802** at iter 50. Still within noise of heuristic.

---

## Diagnosis

The residual architecture is fundamentally wrong for this setting.

**Root cause:** Q_base outputs cross-entropy logits, not Q-values. Residual RL assumes Q_base is a valid Q-function. The logits dominate argmax(Q_base + Q_res) regardless of what Q_res learns, preventing any policy change.

| Problem | Effect |
|---------|--------|
| Q_base is logits, not Q-values | Not Bellman-consistent, arbitrary scale |
| Q_base dominates argmax | argmax(Q_base + Q_res) ~ argmax(Q_base) |
| Margin loss vacuous on Q_combined | Q_base satisfies margin -> zero gradient |
| No exploration diversity | Boltzmann on Q_combined still follows heuristic |

---

## Next directions

### A. Pure DQfD (recommended next)

Drop the residual. Train a standalone DQN with DQfD:

1. Initialize from imitation checkpoint (already matches heuristic)
2. Demo buffer from heuristic rollouts
3. Self-play [DQN, H, H, H] with Boltzmann exploration directly on Q
4. Margin loss on Q prevents forgetting (replaces Q_base's anchor role)
5. DQN loss on Q learns actual Q-values grounded in rewards

Key difference: Q IS the Q-function. Exploration acts on it directly. No logits blocking action divergence.

### B. n-step returns

Add n-step TD targets (n=3 or 5) alongside 1-step. Propagates reward signal faster through sparse-reward game. DQfD paper uses both.

### C. Prioritized experience replay

Focus learning on surprising transitions (high TD error). DQfD paper uses this with a demo priority bonus.

### D. Opponent diversity

Mix in random opponents or past checkpoints instead of all-heuristic. Creates more diverse game states for learning.

### E. Curriculum on critical decisions

Weight specific suboptimal heuristic decisions more heavily:
- Column matching (place high card to zero out a column)
- -2 vs -4 (joker) choice
- End-game timing

**Recommended order:** A -> A+B -> A+B+C

---

## Experiment 4: Pure DQfD (2026-03-06)

**Script:** `src/dqfd.py`

**Approach:** Initialize GolfDQNv2Shallow from imitation checkpoint. Train all parameters with DQN loss + margin loss. Mixed demo + self-play batches. Boltzmann exploration directly on Q. No residual wrapper.

### Run 1: lr=1e-4, lambda_margin decaying 1.0 -> 0.1

**Result:** Catastrophic forgetting by iter 80. DQN score collapsed to 28.8 (worse than random). The DQN loss reshaped logits into Q-values, destroying the policy. Margin loss too weak (0.05) with decaying lambda.

### Run 2: lr=3e-5, lambda_margin constant at 1.0

**Result:** Slower forgetting but same trajectory. Stable at ~14.0 through iter 80, then drifts up: iter 120 = 15.0, iter 150 = 16.5. DQN loss plateaued at ~2.0 but continued to erode policy. Best: **13.895** at iter 1 (essentially the imitation model).

### Diagnosis

The DQN loss is fundamentally destructive when starting from imitation logits. The model's outputs are cross-entropy logits with good action ranking but no Bellman consistency. The TD loss converts these to Q-values, which inevitably destroys the action ranking before the Q-values become useful. The margin loss can slow this but not prevent it because:

1. The margin only constrains the *expert* action to be best-by-margin -- it says nothing about the Q-value *scale*
2. The TD loss reshapes the entire output distribution, not just action ordering
3. By the time Q-values are Bellman-consistent, the policy is already degraded

### Key insight (REVISED -- see Experiment 5)

The diagnosis above was wrong. The real issue was a bug in transition recording: `next_obs` was captured immediately after the player's action, before opponents acted. The agent never sees that state -- it sees the board after all opponents complete their turns. This broke the Markov property, making TD targets inconsistent and causing the Q-values to diverge.

---

## Experiment 5: DQfD with next_obs fix (2026-03-06)

**Fix:** Defer stage-1 transition recording until the player's next turn, when the correct post-opponent-actions observation is available. Applied to `collect_demo_transitions`, `collect_agent_transitions` (residual_dqn.py), `train_episode`, and `train_episodes_vectorized` (tournament.py).

**Commit:** `38194d3`

### Vanilla tournament DQN (from scratch)

Re-ran tournament training with identical config to the `tournament_v2s` baseline (20 gens, 12 agents, v2s, seed 42).

| Metric | Old (stale next_obs) | Fixed |
|--------|---------------------|-------|
| Best solo score | 18.2 | **16.0** |
| Champion solo gen 20 | 18.3-19.0 | **16.2** |

The fix cut the solo-vs-heuristic gap roughly in half (4.2 -> 2.0).

### DQfD from imitation checkpoint

Re-ran DQfD with lr=1e-4, lambda_margin=1.0 constant, 200 iterations. Same config as Experiment 4 Run 1 but with correct transitions.

| Iteration | Old DQfD (Run 2, lr=3e-5) | Fixed DQfD (lr=1e-4) |
|-----------|--------------------------|----------------------|
| 1 | 13.9 | 13.9 |
| 50 | ~14.0 | 14.3 (brief drift) |
| 100 | ~14.5 (degrading) | **13.6** (improving) |
| 150 | **16.5** (collapsed) | **13.6** (stable) |
| 200 | n/a | **13.5** |
| Best | 13.9 at iter 1 | **13.3** at iter 170 |

The DQN now **beats the heuristic baseline** (13.3 vs 14.0). No catastrophic forgetting through 200 iterations. The logit/Q-value transition that Experiment 4 diagnosed as "fundamentally destructive" is actually manageable when TD targets are computed from correct states.

### Why the fix matters

In a 4-player game, 3 opponents act between a player's turns. The stale `next_obs` reflected the board state immediately after the player's action, not the state they actually see next. This meant:

1. TD targets `r + gamma * Q(s')` used phantom states `s'` that never occur in actual play
2. The Q-function couldn't converge to correct Bellman values
3. The resulting gradient noise was destructive enough to overcome the margin loss anchor

With correct `next_obs`, the TD targets are consistent with the actual game dynamics, allowing the Q-values to converge properly while the margin loss preserves the policy.

### DQfD hyperparameter sweep (2026-03-07)

Tested whether higher LR and lower temperature would accelerate learning.

| Config | Best score | Iter at best | Plateau |
|--------|-----------|-------------|---------|
| lr=1e-4, temp 1.0->0.1 | **13.3** | 170 | ~13.3-13.6 |
| lr=1e-3, temp 0.3->0.1 | **13.3** | 195 | ~13.3-13.5 |

Both runs converge to the same plateau (~13.3). Higher LR makes the DQN loss converge faster (2.1 at iter 10 vs 3.3) but eval improvement is identical. The bottleneck is not gradient speed.

### Heuristic decomposition (2026-03-07)

Measured how much each strategic component contributes by testing variant heuristics in 4-player self-play (20k games, 9 holes, avg across all seats). Lower is better.

**Strategy isolation** (random draw/placement as controls):

| Variant | Avg/hole | vs random |
|---------|---------|-----------|
| Random | 30.84 | -- |
| Random draw + base placement | 21.61 | -9.2 |
| Random draw + improved placement | 19.28 | -11.6 |
| Smart draw + random placement | 22.41 | -8.4 |
| Base (smart draw + smart placement) | 15.48 | -15.4 |
| Improved (base + place at revealed) | 11.73 | -19.1 |

Smart placement (-9.2) contributes more than smart draw (-8.4), and they synergize: combined (-15.4) beats the sum of individual contributions (-17.6). Taking good cards matters more when you place them well, and vice versa.

**Cutoff threshold sweep** (cutoff = minimum card score to take from discard):

| Cutoff | Simple (random place) | Base heuristic |
|--------|----------------------|----------------|
| 0 | 27.78 | 17.64 |
| 2 | 23.55 | 15.97 |
| 4 (default) | 22.41 | 15.47 |
| **5** | 21.69 | **15.39** |
| **6** | **21.37** | 15.81 |
| 8 | 22.06 | 18.39 |
| 11 | 30.83 | 30.85 |

Optimal cutoff shifts from 5 (base) to 6 (simple) -- without column awareness, it pays to be greedier about taking cards. Either way, tuning the threshold is worth at most **0.08-1.04 points** depending on variant. The DQN's 0.7 improvement over heuristic is not from finding a better threshold.

**Seat bias:** Improved heuristic shows strong first-mover advantage (seat 0: 10.65, seat 3: 12.62, std=0.73). Base heuristic has mild bias (std=0.16). Random/simple have none.

### Current plateau analysis

The DQfD plateaus at ~13.2 regardless of LR or temperature. Behavioral metrics (see below) confirm the improvement comes from sharper column matching (col 0.53 -> 0.62), not from revealed-card replacement or take-rate changes.

Possible barriers to further improvement:

1. **Margin loss ceiling:** The margin loss anchors to heuristic demos. The heuristic *never* replaces revealed cards, so the margin loss actively penalizes the one action worth 5.5 points. Decaying lambda was blamed for catastrophic forgetting in Exp 4, but that was the stale next_obs bug -- decaying lambda may now be safe.
2. **Exploration limit:** Boltzmann on Q-values only perturbs around the current policy. Discovering "swap a revealed card for a column match" requires trying actions the heuristic never takes.
3. **Opponent diversity:** Self-play is [DQN, H, H, H]. The model only sees games against heuristic opponents, limiting the state distribution.

---

## Behavioral metrics instrumentation (2026-03-07)

Added per-seat behavioral tracking to `_run_eval_config()` in `src/tournament.py` and `count_column_matches()` to `src/vectorized_golf.py`. Four metrics are tracked during live tournament solo eval [DQN, R, H, R]:

| Metric | What it measures | Diagnostic value |
|--------|-----------------|------------------|
| `col_matches` | Avg column matches per hole (0-3, revealed cards only) | Primary skill indicator -- correlates near-perfectly with score |
| `take_rate` | Fraction of stage 0 turns taking the face card vs drawing | Weak differentiator (~0.31-0.37 for all non-random strategies) |
| `rev_replace` | Fraction of stage 1 place actions at already-revealed positions | Signature of "improved" strategy (replacing known cards for column matches) |
| `s1_entropy` | Shannon entropy of stage 1 action distribution | Measures behavioral diversity; high for context-dependent strategies |

### Heuristic baseline reference values

Evaluated each strategy in [STRATEGY, Random, Random, Random] config, 2000 games x 9 holes.

| Strategy | Score | col | take | rev | ent |
|----------|-------|-----|------|-----|-----|
| Random | 30.8 | 0.14 | 0.50 | 0.51 | 2.4 |
| Simple (take low + random place) | 22.4 | 0.22 | 0.31 | 0.00 | 1.8 |
| Heuristic (column-aware) | 13.6 | 0.53 | 0.35 | 0.00 | 2.5 |
| Improved (place at revealed too) | 8.1 | 0.70 | 0.34 | 0.33 | 2.4 |
| simple_s0 + heur_s1 | 15.0 | 0.40 | 0.31 | 0.00 | 2.5 |
| heur_s0 + simple_s1 | 22.2 | 0.27 | 0.37 | 0.00 | 1.8 |

### What the metrics reveal

**Column matching is the dominant skill.** col_matches correlates near-perfectly with score: random=0.14, simple=0.22, heuristic=0.53, improved=0.70. A DQN at solo=16.0 should show col_matches between 0.22-0.53 if it's learning column matching at all.

**Stage 1 placement drives ~90% of the value.** The mix experiments confirm: heur_s0+simple_s1 (22.2) is barely better than pure simple (22.4), while simple_s0+heur_s1 (15.0) nearly matches full heuristic (13.6). When watching DQN learn, col_matches and rev_replace are the metrics to watch, not take_rate.

**rev_replace is the "improved" strategy detector.** Only random (0.51, accidental) and improved (0.33, deliberate) replace revealed cards. Base heuristic never does (0.00). If a DQN shows rev_replace > 0.05, it has discovered the improved strategy worth 5.5 points.

**Entropy reflects strategic diversity, not quality.** Heuristic (2.5) and improved (2.4) have high entropy because they use context-dependent action selection. Simple has low entropy (1.8) because it always places at a random unrevealed slot. A DQN with near-zero entropy is degenerate (always picking the same action).

### Detection thresholds for DQN tournament monitoring

| Signal | Threshold | Interpretation |
|--------|-----------|---------------|
| col_matches > 0.30 | Learning column matching | Corresponds to ~simple-to-heuristic level |
| col_matches > 0.50 | Full column-matching strategy | Matches base heuristic |
| rev_replace > 0.05 | Discovered revealed-card replacement | Beginning of "improved" strategy |
| rev_replace > 0.20 | Systematic revealed-card replacement | Approaching improved heuristic |
| take_rate < 0.25 or > 0.45 | Degenerate take policy | Too greedy or too passive |
| s1_entropy < 0.5 | Action collapse | Always picking same action, likely broken |

### DQN model evaluation (2026-03-07)

Evaluated all key DQN checkpoints with behavioral metrics. [DQN, Random, Heuristic, Random], 2000 games x 9 holes.

| Model | Score | col | take | rev | ent | Notes |
|-------|-------|-----|------|-----|-----|-------|
| imitation | 14.0 | 0.55 | 0.32 | 0.00 | 2.5 | Near-perfect heuristic clone |
| tourn_imitation (hof) | 13.9 | 0.55 | 0.32 | 0.00 | 2.5 | Tournament training preserved imitation policy |
| dqfd_fixed (lr=1e-4) | 13.5 | 0.58 | 0.32 | 0.02 | 2.5 | Slight col improvement, hint of rev_replace |
| dqfd_fast (lr=1e-3) | 13.2 | 0.62 | 0.33 | 0.03 | 2.5 | Best model; col matching explains the gain |
| tourn_nextobs_fix (hof) | 16.2 | 0.20 | 0.33 | 0.78 | 2.2 | Broken: high rev_replace without column matching |
| tourn_v2s (pre-fix hof) | 18.4 | 0.12 | 0.33 | 0.86 | 2.2 | Same broken pattern, worse with stale next_obs |

**Key findings:**

1. **DQfD improves through better column matching.** The best model (dqfd_fast, 13.2) reaches col=0.62 vs the heuristic's 0.53. It is behaviorally identical to the heuristic in every other dimension -- same take_rate, same entropy, near-zero rev_replace. The 0.8-point improvement is entirely from sharper column awareness on unrevealed positions.

2. **Tournament DQN from scratch learns the wrong strategy.** Both pre-fix (18.4) and post-fix (16.2) models have col=0.12-0.20 (near random) with rev_replace=0.78-0.86 (far above random's 0.51). They aggressively replace revealed cards *without* column matching -- the worst combination. They learned "place at positions you can see" without understanding *why*.

3. **The remaining gap is rev_replace.** DQfD is at (col=0.62, rev=0.03); the improved heuristic is at (col=0.70, rev=0.33). The 5-point gap is almost entirely explained by the model not yet learning to replace revealed cards *for column matches*. The margin loss anchoring to heuristic demos (which never do this) is the likely barrier.

4. **take_rate is useless.** Every model from imitation through tournament lands at 0.32-0.33. Stage 0 decisions are uniform across all competent strategies.

---

## Root cause: reward signal bias toward revealed positions (2026-03-07)

### How we found it

The DQN model evaluation above raised a question: why does tournament DQN from scratch converge on the *exact wrong strategy* (high rev_replace, low col_matches)? We investigated in three steps:

**Step 1: Rule out sparse rewards.** Scanned 270k stage-1 decisions and found column-match-creating placements are available in 24% of turns (not rare). The reward when taking them is strong: mean +3.3, 66% positive, 28% above +5. Sparse reward is not the problem.

**Step 2: Probe Q-values.** For 57k column-match opportunities, the tournament DQN selects the match action only 9% of the time (vs 38% for DQfD). The Q-gap is -4.1: the model actively *prefers* non-matching actions. This isn't exploration failure -- the learned Q-function is wrong.

**Step 3: Decompose rewards by action type.** This revealed the root cause:

| Action type | Mean reward (random play) | n |
|---|---|---|
| Place at REVEALED position | +0.00 | 254k |
| Place at UNREVEALED position | -5.17 | 239k |
| Discard + flip | -5.18 | 239k |

There is a **+5.2 point systematic bias** toward placing at revealed positions.

### The mechanism

`compute_score()` in `step_stage1` treats unrevealed cards as contributing 0 to score:

```
reward = score_before - score_after
```

When placing at an **unrevealed** position:
- `score_before`: hidden card contributes 0
- `score_after`: held card is now revealed, contributes its actual score (avg 5.5)
- `reward = 0 - 5.5 = -5.5` (almost always negative)

When placing at a **revealed** position:
- `score_before`: old card contributes its known score (avg 5.5)
- `score_after`: held card replaces it (avg 5.5)
- `reward = 5.5 - 5.5 = 0.0` (roughly neutral)

The reward function penalizes *information gain*. Revealing a new card "increases" the visible score even when the hidden card being replaced was worse. Every position shows the same +5.2 gap regardless of game state.

### Why each approach is affected differently

| Approach | Effect |
|----------|--------|
| **Tournament DQN** | Learns the bias directly. Converges on always placing at revealed positions (rev=0.78). Never discovers column matching because it avoids unrevealed positions entirely. |
| **Heuristic** | Hardcoded to prefer unrevealed positions. Immune to reward bias by construction. |
| **Imitation DQN** | Clones heuristic behavior. Inherits immunity. |
| **DQfD** | Margin loss anchors to heuristic demos (unrevealed placement). DQN loss pushes toward revealed placement. These fight to a draw, producing a mild improvement (col 0.53->0.62) without strategy collapse. |

### Implications

This is the same class of bug as the stale next_obs (Experiment 5). The reward signal violates the assumptions of Q-learning: the immediate reward doesn't reflect the true value of the action because `compute_score` has an observability gap between visible and hidden card contributions.

The fix must make the reward function score-neutral with respect to revealing cards. Options:
1. Use final score (all cards revealed) for rewards instead of intermediate visible score
2. Impute hidden card values in `score_before` (e.g., expected value of a random card)
3. Use only end-of-hole reward (sparse but unbiased)

---

## Experiment 6: Hindsight reward shaping (2026-03-07)

**Commit:** `src/reward_shaping.py`, `src/tournament.py`

**Approach:** Option 1 from above. After each stage-1 action, compute reward using `compute_final_score` (treats all 6 cards as revealed) instead of the biased `compute_score` (treats unrevealed cards as 0). The agent's policy still conditions on partial observations; only the training reward signal uses hindsight.

This is equivalent to how humans learn card games: you can't see hidden cards during play, but afterward you know what they were and learn from the outcome.

**Implementation:** `HindsightRewardShaper` in `src/reward_shaping.py`. Before `step_stage1`, snapshot `player_cards`. After, compute `compute_final_score(before) - compute_final_score(after)`. The shaped reward replaces the biased one in the replay buffer. Enabled by default (`--reward-shaping hindsight`).

**What this fixes:**
- Place at unrevealed: before-score now includes the hidden card's true value, so revealing it is score-neutral
- Place at revealed: unchanged (both before/after are fully visible)
- Column matching: correctly rewarded regardless of whether the position was previously revealed

**Expected effect on behavioral metrics:**
- `rev_replace` should decrease from 0.78 (no longer rewarded for revealed placement per se)
- `col_matches` should increase (column matches rewarded equally at all positions)
- Tournament DQN should no longer converge on the degenerate strategy

**Status:** Not yet evaluated. Next: run tournament with hindsight shaping and compare behavioral metrics.
