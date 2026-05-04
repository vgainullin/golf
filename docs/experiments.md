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

Develop a generalized RL training system applicable to any game-playing domain. Golf is the first test case.

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

**Note:** The 8.1 score is measured in `[improved, R, R, R]` -- three random opponents. In the DQN solo eval config `[player, R, H, R]` (one heuristic opponent), the improved heuristic scores **10.52** (5000 games, confirmed). All DQN solo scores in this document use `[DQN, R, H, R]` and should be compared against 10.52, not 8.1.
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

### Run 1: eps 0.3->0.05 (default), 20 gens, 12 agents, seed 42

Partial results (killed at gen 7 to restart with higher epsilon):

| Gen | best | solo | col | rev | take | ent |
|-----|------|------|-----|-----|------|-----|
| 1 | 22.69 | 22.15 | 0.118 | 0.487 | 0.262 | 2.0 |
| 2 | 19.15 | 18.74 | 0.172 | 0.352 | 0.544 | 2.3 |
| 3 | 15.78 | 14.84 | 0.220 | 0.418 | 0.379 | 2.2 |
| 4 | 16.06 | 14.75 | 0.258 | 0.370 | 0.409 | 2.2 |
| 5 | 14.80 | 13.73 | 0.270 | 0.336 | 0.320 | 2.3 |
| 6 | 15.14 | 14.19 | 0.245 | 0.370 | 0.526 | 2.3 |
| 7 | 13.79 | 13.43 | 0.272 | 0.362 | 0.425 | 2.4 |

**Key findings:**

1. **Reward bias fix confirmed.** `rev_replace` stays at 0.35-0.49 (near random baseline 0.51), never climbing to the 0.78 seen without hindsight shaping. The degenerate "always place at revealed" strategy is gone.

2. **Vanilla DQN from scratch beats the heuristic by gen 5.** Solo=13.73 at gen 5, 13.43 at gen 7, vs heuristic baseline of 14.0. Previous tournament DQN (no shaping) plateaued at 16.0 after 20 gens. No imitation pretraining, no DQfD, no demo buffer.

3. **Low column matching, high score.** col_matches=0.27 at gen 7 -- half the heuristic's 0.53, but the agent scores better. The hindsight reward lets it learn the true value of placing at unrevealed positions (which the biased reward penalized). The agent gets value from better card selection and placement, not column matching.

4. **Column matching is the remaining upside.** The agent discovers column matches (0.27 > random's 0.14) but doesn't systematically seek them. Eps=0.3 limits exploration diversity -- column match discovery is bottlenecked by the chance of randomly placing matching ranks in columns.

### Run 2: eps 1.0->0.05, 20 gens, 12 agents, seed 42

Full 20-generation run with high initial exploration.

| Gen | best | solo | col | rev | take | ent |
|-----|------|------|-----|-----|------|-----|
| 1 | 23.66 | 23.24 | 0.167 | 0.500 | 0.854 | 2.3 |
| 2 | 17.04 | 16.61 | 0.214 | 0.402 | 0.337 | 2.3 |
| 3 | 15.26 | 14.76 | 0.236 | 0.333 | 0.378 | 2.3 |
| 5 | 14.16 | 13.80 | 0.264 | 0.281 | 0.452 | 2.3 |
| 7 | 13.82 | 12.60 | 0.276 | 0.318 | 0.345 | 2.3 |
| 10 | 13.75 | 12.88 | 0.278 | 0.276 | 0.352 | 2.2 |
| 12 | 13.20 | 12.05 | 0.306 | 0.303 | 0.316 | 2.3 |
| 15 | 13.67 | 12.79 | 0.278 | 0.288 | 0.412 | 2.3 |
| 19 | 12.80 | 12.06 | 0.310 | 0.313 | 0.302 | 2.4 |
| 20 | 13.74 | 12.15 | 0.310 | 0.277 | 0.360 | 2.4 |

Champion eval (2000 games, [DQN, R, H, R]):
- **Solo: 12.26** (beats heuristic 14.0 by 1.7, beats DQfD 13.3 by 1.0)
- col_matches=0.30, rev_replace=0.28, rev_col_match=0.055, take_rate=0.36

**Key findings:**

1. **New best: 12.3 solo from vanilla DQN.** No imitation, no demos, no margin loss. Just correct rewards + high exploration. Beats the previous best (DQfD 13.3) by a full point.

2. **High eps helped but didn't change the trajectory.** Both runs converge to similar col_matches (~0.30) and solo scores (~12-13). The high eps let it keep improving through gen 20 instead of plateauing at gen 7, but the strategy is the same.

3. **The DQN discovered value swapping, not column matching.** rev_col_match=0.055 means only 5.5% of revealed-card replacements create column matches. The other 94.5% are pure value swaps: replacing a high revealed card with a lower card from the deck/discard. This is a strategy the heuristic never uses (it only places at unrevealed positions).

4. **Column matching plateaued at 0.30.** Stayed flat from gen 7 through gen 20. Eps-greedy exploration can't systematically discover "match ranks across rows in the same column" -- it requires noticing a rank, holding a matching card, and placing it in the right position. The agent stumbles into it (0.30 > random 0.14) but can't reliably learn it.

### Strategy decomposition: DQN vs heuristic

| Agent | score | col | rev | rcm | Strategy |
|-------|-------|-----|-----|-----|----------|
| Heuristic | 13.9 | 0.55 | 0.00 | 0.00 | Column matching at unrevealed positions only |
| Hindsight DQN | 12.3 | 0.30 | 0.28 | 0.06 | Value swapping at revealed + some column matching |
| Improved heur | 10.52 | 0.70 | 0.33 | n/a | Both strategies combined (vs [R,H,R]) |

The DQN found a **complementary strategy** to the heuristic. The heuristic gets value from column matching (col=0.55) but never touches revealed cards. The DQN gets value from replacing revealed high cards with lower ones (rev=0.28) but doesn't systematically column match. Combining both strategies (as the improved heuristic does) is the goal. The improved heuristic scores 10.52 in the same `[player, R, H, R]` eval config used for DQN (8.1 is its score vs three random opponents, a different and easier config).

---

## The bitter lesson (so far)

Richard Sutton's [bitter lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html) (2019): AI researchers repeatedly try to bake human knowledge about the *solution* into their agents -- chess evaluation functions, Go heuristics, speech phonetics. This helps short-term but plateaus. General methods that scale with computation (search and learning) consistently win in the long run. The lesson is "bitter" because researchers want their domain expertise to matter, but it doesn't.

### Taking the bitter lesson at face value

If we believe Sutton, then a DQN with correct training signals should learn to play 6-card golf well without any human knowledge injected -- no imitation pretraining, no heuristic demos, no margin loss. The algorithm is general. The game is small (8 actions, ~50-dimensional observation, episodes under 30 steps). If it doesn't work, the problem is in our implementation, not in the approach.

This is exactly what happened. The imitation pipeline (Exp 1-4) was never meant to be the final approach -- it was a diagnostic tool to verify the network architecture could represent a good policy at all. When DQfD plateaued and collapsed, the natural conclusion was "RL can't improve on imitation for this game." The actual conclusion should have been: something in the training loop is broken, find it.

Two bugs were found, both in the training signal:

1. **Stale next_obs (Exp 5).** Transitions recorded the board state before opponents acted. The agent was learning Q-values for an MDP that didn't match the actual game.

2. **Reward bias (Exp 6).** `compute_score` treats unrevealed cards as 0, creating a +5.2 bias toward placing at revealed positions. The agent learned exactly what the reward told it.

After both fixes, vanilla DQN from scratch reached 12.3 -- beating the heuristic (14.0) and the imitation-anchored DQfD (13.3). No human knowledge needed. Sutton's lesson confirmed: the general method works when the problem specification is correct.

### What's troubling

The bitter lesson says the general method should work. It did work -- but only after extensive human diagnosis of what the agent was learning and why.

Without behavioral metrics (col, rev, take, ent), we would never have noticed that tournament DQN converged on "always place at revealed positions." Without decomposing rewards by action type, we would never have found the +5.2 bias. Without understanding the multi-player turn structure, we would never have caught the stale next_obs. Each fix required deep domain understanding of both the game mechanics and the RL training loop.

The diagnostic sequence that produced results was:

1. Add behavioral metrics to see *what* the agent learned
2. Notice the agent learned something obviously wrong (rev=0.78, col=0.12)
3. Ask "why would Q-learning converge on this specific wrong strategy?"
4. Trace the answer back to the training signal
5. Fix the signal

This process is the opposite of Sutton's vision. We didn't let the general method run and scale. We dissected its behavior, interpreted it through human understanding of the game, and surgically fixed the training signal. The general method needed a human to debug it before it could work.

The question is whether this invalidates the bitter lesson or just reflects the current state of the implementation. Sutton's examples (chess, Go, speech) all involved teams spending years getting the infrastructure right before the general method took over. Deep Blue's search worked because the game tree was correctly implemented. AlphaGo's self-play worked because the Go engine was bug-free. The general method scales, but only on a correct foundation -- and building that foundation required exactly the kind of domain expertise Sutton says doesn't matter for the solution.

### The uncomfortable middle ground

The bitter lesson draws a clean line: human knowledge about the solution is wasted effort; general methods win. This project suggests the line is blurrier in practice:

- **Human knowledge about the solution:** The heuristic's column-matching rules, the DQfD margin loss anchoring to heuristic demos. These *did* constrain the agent. DQN from scratch (12.3) beat DQfD (13.3). Sutton is right here.

- **Human knowledge about the problem:** Correct state transitions, unbiased rewards, behavioral metrics for diagnosis. These were *essential*. Without them, the general method confidently solved the wrong problem. Sutton doesn't address this because his examples (chess, Go) had correct game engines from the start.

- **Human knowledge about what the agent is learning:** The behavioral metrics, the reward decomposition, the "why is it doing that?" investigation. This is the most uncomfortable category. It's not knowledge about the solution (we didn't tell the agent to column-match). It's not knowledge about the problem (the reward function was already defined). It's knowledge about the learning process itself -- the thing Sutton says we should trust to work on its own.

The current col_matches plateau (0.30, flat from gen 7-20) sits right on this boundary. The agent discovered value swapping on its own -- a strategy the human designer missed. But it hasn't discovered systematic column matching, which the human designer encoded trivially. Is this a bug in the training signal (another stale-next_obs-class problem waiting to be found)? A representation gap? Or just insufficient scale -- and if we trained for 200 generations instead of 20, would the general method find it?

If the pattern from Experiments 1-6 holds, the answer is probably another signal/representation issue. But the bitter lesson says we should at least try scaling first, because the instinct to diagnose and intervene is the same instinct that led to four failed experiments before we found the actual bugs.

---

## MDP Diagnostics Toolkit (2026-03-07)

**File:** `src/diagnostics.py`

Both bugs that took multiple failed experiments to find (stale next_obs, reward bias) are general RL failure modes. Gymnasium's `env_checker` validates the environment interface but doesn't check for these. We built diagnostics that complement Gym's approach: Gym checks the interface is correct, we check the MDP is correct.

### Pre-training probes

Four checks run on random-policy rollouts (~500 games, ~2 seconds on CPU):

| Check | What it catches | Pass/Fail |
|-------|----------------|-----------|
| **Transition fidelity** | Stale next_obs: compares immediate vs deferred next_obs, reports mismatch rate and L1 distance | FLAG if L1 > 1.0 |
| **Reward-action distribution** | Reward bias: groups rewards by action, reports spread across actions. Compares raw vs hindsight reference rewards | FLAG if spread > 2.0 |
| **Determinism** | Hidden state, non-deterministic obs, RNG leaks: runs two episodes with same seed, asserts identical trajectories | FAIL on any mismatch |
| **Observation sanity** | NaN/Inf, shape mismatches, impure observation functions | FAIL on any violation |

### Results on current environment

```
Transition Fidelity (7576 transitions): [FLAG]
  immediate != deferred: 48.7% (3689/7576)
  mean L1 distance: 41.83

Reward-Action Distribution (3631 stage-1 transitions): [FLAG]
  place actions (2-7):          mean reward ~-2.5
  discard+flip actions (9-14):  mean reward ~-5.1
  spread: 3.28
  reference (hindsight) spread: 0.50

Determinism: [OK]
Observation Sanity: [OK]
```

Both FLAGs are expected and already mitigated:
- **Fidelity FLAG:** 48.7% of transitions (all stage-1) show large immediate/deferred gap because 3 opponents act between the player's action and next turn. Training loop stores deferred version (fixed in Exp 5).
- **Reward FLAG:** 3.28 spread from `compute_score`'s revealed/unrevealed asymmetry. Hindsight shaping reduces spread to 0.50 (fixed in Exp 6).

These checks would have caught both bugs immediately if run before the first training attempt. The fidelity check shows the gap exists; the reward check shows the bias. No domain knowledge needed to read the output -- just "large gap, investigate" and "large spread, investigate."

### Integration

- **Standalone:** `uv run python -m src.diagnostics` runs all 4 checks
- **Tournament:** `--sanity-check` flag runs probes before training, aborts on FAIL
- **Passive assertions:** `assert_transition_batch()` available for training loop integration (validates NaN/Inf, shapes, action bounds per batch)

---

## Next: Breaking the column matching plateau

col_matches stuck at 0.30 (gen 7-20). The metrics confirm this is a genuine plateau, not slow growth:

| Gen range | col_matches |
|-----------|-------------|
| 1-3 | 0.17, 0.21, 0.24 |
| 5-7 | 0.26, 0.29, 0.28 |
| 10-12 | 0.28, 0.31, 0.31 |
| 15-20 | 0.28, 0.27, 0.29, 0.29, 0.31, 0.31 |

Score continued improving (12.6 at gen 7 -> 12.1 at gen 19) but the gains come from better value swapping and take_rate refinement, not column matching. More generations won't break this -- the agent is stuck.

### Root cause: stage-0/stage-1 bootstrapping deadlock

Column matching requires two coordinated actions:
1. **Stage 0:** take a card whose rank matches a revealed card in one of your columns
2. **Stage 1:** place that card at the column partner position

Stage-0 gets zero immediate reward. With 1-step TD, its Q-value depends entirely on bootstrapped Q(stage-1 state). But stage-1 Q-values for column-match placements are only learned from episodes where the agent happened to hold the right card AND placed it correctly. At eps=0.05 with wrong Q-rankings, the chance of correct placement is ~1/8 = 12.5%. Column match opportunities exist in ~24% of turns (measured in the reward bias analysis). So the effective reinforcement rate is ~3% of all turns.

This creates a deadlock:
1. Stage-1 Q-values for column-match actions are weak (few reinforcing examples)
2. Stage-0 Q-values for taking matching cards are weak (bootstrapped from weak stage-1 values)
3. Stage-0 rarely takes matching cards (low Q-values)
4. Stage-1 rarely sees column-match opportunities (stage-0 doesn't set them up)
5. Goto 1

The agent escapes this enough to reach col=0.30 (2x random) from accidental successes, but can't go further because the signal-to-noise ratio at 3% reinforcement is too low to sharpen the Q-values past the noise floor.

### Why previous suggestions miss the mark

**Boltzmann exploration** doesn't fix this. At gen 20, eps=0.05 -- only 5% of actions are random. If Q-values correctly ranked column-match placements, col would be much higher. Boltzmann softens the same wrong Q-values. It helps coordination at the margin but doesn't address why the Q-values are wrong in the first place.

**Longer training** doesn't fix this either. Col is flat from gen 7, not slowly climbing. The plateau is structural, not temporal.

**DQfD + hindsight** combines two known approaches but doesn't address the bootstrapping deadlock. The margin loss anchors column matching from imitation, but the DQN loss still can't independently discover or improve column matching because of the same credit assignment gap.

### Proposed fix: n-step returns (n=2)

The bootstrapping deadlock exists because 1-step TD can't propagate the stage-1 column match reward back to the stage-0 action that set it up. With n=2, stage-0 directly sees the stage-1 reward:

```
Q_1step(s0, a0) = r0 + gamma * max Q(s1)     -- r0=0, depends on noisy Q(s1)
Q_2step(s0, a0) = r0 + gamma * r1 + gamma^2 * max Q(s2)  -- directly sees r1 (column match reward)
```

The stage-0 action "take card with matching rank" would get direct credit from the stage-1 column match reward (r1), without relying on bootstrapped Q(s1) being accurate. This breaks the deadlock by providing unambiguous signal: "taking this card led to a +10 reward two steps later."

n=2 is specifically the number that bridges the stage-0/stage-1 gap (one step per stage within a turn). This isn't a generic "use n-step because the DQfD paper does" -- it's the minimum n that solves the credit assignment problem for column matching.

### Alternative hypothesis: representation bottleneck

The n-step diagnosis assumes the problem is credit assignment between stage-0 and stage-1. But stage-1 gets *immediate, large reward* for column matches (+3 to +10 points). If stage-1 itself can't learn to reliably place for column matches despite direct reward signal, the bottleneck isn't temporal -- it's representational.

Cards are encoded as single tokens (0-52) through a learned embedding. A 7 of hearts and a 7 of spades are completely different token IDs. For column matching, the network must independently discover that 4 tokens per rank are equivalent for matching purposes, then compute "does my held card's rank equal the rank of a revealed card in column j?" across all 3 columns. This is a discrete equality check over a 13-class space, masked by suit.

Compare to value swapping, which only requires "is this card's score lower than that card's score?" -- a 1D comparison that's much easier to learn from reward signal.

The imitation model learns rank equivalence easily (col=0.55) because supervised learning directly tells it which action to take. RL from scratch finds value swapping first (easier gradient signal), shapes the embeddings around score comparison, and then the weak column-match gradient (~3% reinforcement rate) can't reshape embeddings that are already committed to a different structure.

### Embedding analysis (2026-03-10)

**Script:** `src/analyze_embeddings.py`

Extracted and compared the card embedding weights (53 tokens, token 0-51 = cards, 52 = unknown) from the hindsight DQN champion (gen19_agent9, solo 12.3) and the imitation model (solo 14.0). Card encoding is `suit * 13 + rank`, so same-rank cards across 4 suits are at indices `{rank, rank+13, rank+26, rank+39}`.

| Metric | Imitation (64d) | Hindsight DQN (128d) |
|--------|----------------|---------------------|
| Within-rank cosine sim | 0.54 | 0.15 |
| Between-rank cosine sim | -0.02 | -0.00 |
| **Separation** | **+0.56** | **+0.15** |

The imitation model's embeddings strongly cluster by rank. Same-rank cards point in roughly the same direction (cosine 0.54), while different-rank cards are near-orthogonal (-0.02). The network can trivially compute rank equality from these embeddings.

The hindsight DQN's embeddings are barely structured by rank. Within-rank similarity (0.15) is only marginally above the between-rank noise floor (-0.00). Same-rank cards are scattered almost randomly in embedding space.

**Per-rank breakdown:**

| Rank | Score | Imitation | Hindsight DQN |
|------|-------|-----------|---------------|
| 2 | -2 | 0.33 | 0.28 |
| 3 | 3 | 0.31 | 0.08 |
| 4 | 4 | 0.64 | 0.05 |
| 5 | 5 | 0.66 | 0.08 |
| 6 | 6 | 0.57 | 0.06 |
| 7 | 7 | 0.66 | 0.06 |
| 8 | 8 | 0.65 | 0.10 |
| 9 | 9 | 0.53 | 0.18 |
| 10 | 10 | 0.61 | 0.25 |
| J | 10 | 0.65 | 0.24 |
| Q | 10 | 0.66 | 0.21 |
| K | 0 | 0.34 | 0.14 |
| A | 1 | 0.38 | 0.16 |

The DQN's embeddings encode card *value*, not card *rank*. The only cards with any within-rank clustering (0.20+) are the extreme-value cards: 2 (score -2), 10/J/Q (score 10). Mid-rank cards (3-8) are at 0.05-0.10 -- indistinguishable from noise.

The score=10 group (10, J, Q) is the smoking gun. Imitation: within_rank=0.64, cross_rank=-0.06 -- it knows 10s, Js, and Qs are different ranks despite identical scores. DQN: within_rank=0.23, cross_rank=0.24 -- it treats them interchangeably because they have the same score. The DQN learned "these are all high cards" but not "these are three distinct ranks for matching purposes."

**Conclusion:** The representation bottleneck hypothesis is confirmed. The hindsight DQN cannot compute rank equality because its embeddings don't encode rank. It learned value-based structure (enough for value swapping) but never developed rank-based structure (required for column matching). The col_matches plateau at 0.30 is a direct consequence: the network literally lacks the representational substrate to identify column match opportunities.

**The general phenomenon:** When RL's dominant early strategy requires only one kind of structure from learned embeddings (here: value ordering for card swapping), the embeddings lock into that structure during early training. A secondary strategy that requires different structure (here: rank equivalence for column matching) becomes unreachable -- not because the architecture can't represent it (the imitation model proves it can), but because the optimization landscape funnels the embeddings toward whichever structure pays off first. The weak gradient from the secondary strategy (~3% reinforcement rate) can't reshape embeddings that are already committed to serving the dominant strategy.

This is a representation learning failure inherent to end-to-end RL with learned embeddings. The training signal determines what structure the embeddings develop, and when multiple strategies require different structures, the winner-take-all dynamics of gradient descent mean the first-discovered strategy monopolizes the representation.

### The general problem: representation monopolization

Adding explicit rank features to the observation would fix golf but not solve anything. The real question is general: how does an RL system learn the right representation when multiple strategies require different structure and the first one discovered monopolizes the embeddings? This is the problem worth solving next -- not as a golf-specific patch, but as a systemic issue that will recur in any domain where learned representations must serve multiple competing strategies.

---

## Representation monopolization: literature and parallels

The col_matches plateau is not an isolated phenomenon. It sits at the intersection of several well-characterized failure modes in deep learning and deep RL.

### Established theory

**Gradient starvation** (Pezeshki et al., NeurIPS 2021). The closest theoretical match. Once a subset of features reduces the loss sufficiently, samples classified correctly by those features stop contributing gradient, starving other features of learning signal. The authors prove this with dynamical systems theory: feature learning dynamics decouple, and the dominant feature suppresses the secondary one. In golf: value swapping provides enough reward signal to shape embeddings; the weak column-matching gradient (~3% reinforcement rate) is starved.

**Simplicity bias** (Shah et al., NeurIPS 2020). SGD systematically learns simpler features first. When a simple attribute correlates with reward, networks latch onto it and fail to learn more complex but useful features. Value ordering (1D comparison) is simpler than rank equality (13-class discrete partition masked by suit). The simpler one wins.

**Primacy bias** (Nikishin et al., ICML 2022). Specific to deep RL: agents overfit to early interactions, shaping representations that make subsequent learning from novel situations impossible. The problem is not data collection but the inability to learn from it. Early experiences monopolize the representation. This explains why more generations don't break the col_matches plateau -- the representation is locked, not undertrained.

**Implicit under-parameterization** (Kumar et al., ICLR 2021). Bootstrapped value learning with gradient descent causes the effective rank of learned features to collapse. Networks with 512-dimensional feature layers show only 20-100 active singular components. The network behaves as low-capacity despite being high-capacity. This is the mechanism by which monopolization becomes irreversible: feature rank collapses around the dominant strategy, leaving no representational room for the secondary one.

**Dormant neuron phenomenon** (Sokar et al., ICML 2023). As training progresses, an increasing fraction of neurons become inactive. A few neurons monopolize activation while many contribute nothing. The neuron-level manifestation of representation monopolization: neurons that could serve column matching go dormant because value swapping dominates.

**Loss of plasticity** (Abbas et al., Nature 2024). The umbrella term. Neural networks in continual learning settings gradually lose the ability to learn from new data. Weights become committed to the existing solution and resist modification, even when new data would benefit from different feature structure.

**Critical learning periods** (Achille et al., ICLR 2019). Analogous to biological critical periods: the first few epochs create strong connections that do not change during additional training. Information plasticity is lost after the initial transient. In golf, the early phase when value swapping is discovered constitutes the critical period that locks embedding structure.

These compound: simplicity bias determines *which* strategy wins. Gradient starvation explains *why* the secondary strategy can't catch up. Primacy bias explains *why it's irreversible*. Implicit under-parameterization and dormant neurons describe the *mechanism* of irreversibility.

### The same failure in other games

**KataGo's cyclic group blind spot** (Wang et al., ICML 2023). The strongest parallel. Superhuman Go AI fails catastrophically on large cyclic groups of stones. The CNN learned local life/death patterns (dominant, easy strategy) and this representation cannot handle global topological reasoning (secondary strategy requiring different structure). An adversary exploiting this beats KataGo in 94/100 games with 8% of KataGo's training compute. Adversarial retraining patches specific patterns rather than learning a general representation of group connectivity -- the representation is too committed.

**AlphaStar strategy collapse** (Vinyals et al., Nature 2019). DeepMind explicitly diagnosed this: "Because some strategies are easier to improve on, naive reinforcement learning would narrowly focus on these. Other strategies may require more learning... This creates a vicious cycle in which some valid strategies appear less and less effective because the agent abandons them in favour of a dominant strategy." Their fix required an entire league of exploiter agents to force representational diversity. The scale of the infrastructure needed is evidence of the severity.

**TD-Gammon's doubling weakness** (Tesauro, 1995). Superhuman positional play but poor doubling decisions. The learned representation encoded positional evaluation features that didn't transfer to cube decisions. Tesauro had to supplement with hand-crafted expert features, acknowledging that the self-taught representation was missing structure.

**Poker RL and bluffing suppression**. Deep RL poker agents learn value betting (exploiting strong hands) but suppress bluffing. Bluffing requires modeling opponent belief states -- a different feature structure than hand-strength evaluation, which pays off first and monopolizes the representation.

**DQN on Montezuma's Revenge**. Usually framed as sparse rewards + poor exploration, but there's a representation component. CNN features learned from dying in room 1 serve "avoid death" but can't represent key-and-door reasoning. Go-Explore sidesteps the representation problem entirely by using hand-designed cell representations.

**AlphaGo's ko avoidance**. AlphaGo appeared to systematically avoid ko fights rather than learn to handle them. Ko reading requires global board assessment of ko threats -- different from the local pattern matching that dominates the learned representation.

### The pattern across games

| Game | Dominant strategy (easy) | Suppressed strategy (hard) | Representation gap |
|------|-------------------------|---------------------------|-------------------|
| Golf | Value swapping (1D ordering) | Column matching (rank equality) | Embeddings encode value, not rank |
| Go (KataGo) | Local life/death patterns | Global cyclic group reasoning | CNN features are local, not topological |
| StarCraft | Dominant race/build order | Counter-strategies | Features serve the winning build, not diverse play |
| Backgammon | Positional evaluation | Doubling cube decisions | Features encode position, not decision theory |
| Poker | Value betting (hand strength) | Bluffing (opponent modeling) | Features encode cards, not beliefs |
| Montezuma | Obstacle avoidance | Key-and-door reasoning | Features encode spatial danger, not object relations |

In every case: the dominant strategy requires simpler representational structure, gets discovered first, and locks the embeddings/features. The secondary strategy requires qualitatively different structure that can't develop once the representation is committed. More training doesn't help because the bottleneck is representational, not temporal.

### Proposed solutions from the literature

**Spectral decoupling** (Pezeshki et al., NeurIPS 2021). Replace weight decay with L2 penalty on network *outputs* (logits). Decouples feature learning dynamics so the dominant feature can't suppress the secondary one. Directly targets gradient starvation. Low implementation cost, theoretically grounded for exactly this failure mode.

**Periodic network resets** (Nikishin et al., ICML 2022). Re-initialize the last few layers while preserving the replay buffer. Overcomes primacy bias by letting the network re-learn from accumulated data without the locked representation. In golf: reset embedding layer every N generations; the buffer already contains column-matching experiences that the locked embeddings can't learn from.

**Dormant neuron recycling / ReDo** (Sokar et al., ICML 2023). Identify dormant neurons and reinitialize their incoming weights. More surgical than full layer resets: targets specifically the neurons that could serve column matching but went dormant.

**Continual backpropagation** (Abbas et al., Nature 2024). Reinitialize a small fraction of less-used units after each example. Maintains a steady stream of fresh neurons available for new features. Maintains plasticity indefinitely.

**Relational attention** (Zambaldi et al., ICLR 2019). Self-attention over entity embeddings computes pairwise relations explicitly, making "does card A have the same rank as card B?" a first-class operation rather than something that must emerge from embedding geometry. Demonstrated in StarCraft mini-games for relational reasoning. Architectural rather than regularization-based.

**Factored embeddings**. Decompose tokens along known compositional axes: `card_repr = concat(rank_embed, suit_embed)`. Forces rank structure into the representation by construction. Domain-general wherever tokens have compositional structure (rank x suit, piece x position, item x modifier).

**Population diversity / DvD** (Parker-Holder et al., NeurIPS 2020). Behavioral diversity pressure alongside fitness in population-based training. Uses determinantal point processes to ensure agents span different strategies rather than clustering. Integrates naturally with tournament training.

**Gradient surgery / PCGrad** (Yu et al., NeurIPS 2020). When gradients from different objectives conflict, project each onto the normal plane of the other. Prevents destructive interference without separate networks.

## Experiment 7: Breaking the col_matches plateau (2026-03-11)

### Setup

Four controlled experiments, each changing one variable from the Exp 6 Run 2 baseline (hindsight DQN, eps 1.0->0.05, 20 gens, 12 agents, seed 42). Baseline reproduced first as control.

### Results

| Experiment | col_matches (gen 7-20) | Best solo | HoF | Hypothesis tested |
|---|---|---|---|---|
| Baseline | 0.261 - 0.314 | 12.03 | 13.12 | (control) |
| 7A: Factored embeddings (v2sf) | 0.265 - 0.319 | 11.82 | 13.07 | Representation bottleneck |
| 7B: Spectral decoupling | 0.253 - 0.315 | 11.71 | 12.73 | Gradient starvation |
| 7C: 2-step returns | 0.256 - 0.310 | 12.27 | 13.07 | Credit assignment delay |
| 7D: Periodic resets (every 5 gens) | 0.274 - 0.338 | 12.16 | 13.13 | Primacy bias / plasticity loss |

### Key finding: representation is not the bottleneck

7A gave the model factored rank+suit embeddings where same-rank cards share identical rank vectors by construction. Embedding analysis confirms:

| Model | Within-rank cosine | Between-rank cosine | Separation |
|---|---|---|---|
| Baseline (monolithic) | 0.103 | -0.010 | +0.113 |
| 7B Spectral (monolithic) | 0.118 | -0.012 | +0.130 |
| 7A Factored (rank+suit) | 0.786 | +0.031 | +0.755 |

7A has 6.7x stronger rank clustering. The rank embedding learned value ordering (face cards > low cards by norm) and suit embeddings are near-orthogonal (suits treated as interchangeable). The representational substrate for column matching is present. Yet col_matches didn't improve -- the agent can see rank equivalence but still doesn't exploit it.

This rules out the representation bottleneck hypothesis from the Exp 6 analysis. The problem is not that the agent can't represent "same rank" -- it's that the reward signal from column matching is too delayed or too sparse relative to the immediate payoff from value swapping.

### 7A: factored embeddings -- faster early learning, same ceiling

Gen-by-gen comparison shows factored is consistently ahead in gen 1-10 (mean +0.031 col_matches in gen 1-6, +0.035 in gen 7-10) but the advantage fades to zero by gen 11-20. The representation gave a head start on discovering column matching but didn't raise the ceiling. Once exploration decays (epsilon drops), both converge to the same plateau.

### 7B: spectral decoupling -- better solo, same col_matches

Spectral decoupling (L2 penalty on Q-values, Adam instead of AdamW) improved solo score (11.71 vs 12.03) and HoF (12.73 vs 13.12) without touching col_matches. Better overall learning quality but gradient starvation is not the specific barrier to column matching.

### 7C: 2-step returns -- faster gen 1, same plateau

2-step returns accelerated early learning dramatically (solo 17.97 at gen 1 vs baseline's 23.18) but col_matches plateaued at the same level. Bridging the stage-0-to-stage-1 credit gap didn't help. The column matching reward is not about the 2-step delay between take and place -- it's about the much longer delay to end-of-hole scoring where column zeroing pays off.

### 7D: periodic resets -- only experiment that moved the ceiling

col_matches range shifted up to 0.274-0.338 (baseline: 0.261-0.314). Visible pattern around resets: gen 10 reset caused a dip (0.274) followed by recovery to 0.338 at gen 12 -- the highest col_matches in any experiment. Subsequent resets show smaller bumps. The fresh embedding weights briefly allow the agent to reorganize toward column matching before re-converging. Plasticity loss is a contributing factor but not sufficient on its own.

### Synthesis

No single intervention broke the plateau. The decision matrix:

- **7A null, 7B null**: representation and gradient dynamics are not the bottleneck
- **7C null**: short-horizon credit assignment (2-step) insufficient
- **7D marginal**: plasticity loss contributes but doesn't explain the plateau alone
- **7A early advantage + 7D ceiling bump**: both help at the margins, suggesting the problem is multifaceted

The remaining hypothesis is **exploration**: eps-greedy with decaying epsilon doesn't generate enough column-matching experiences in later generations. The agent can learn from them when encountered (7A early advantage proves this) but stops encountering them. A directed exploration strategy (e.g., intrinsic reward for column matches, or Boltzmann exploration with temperature tied to column-match Q-values) may be needed rather than representation or optimization fixes.

## Experiment 8: Self-attention DQN (GolfDQNv3) (2026-03-14)

### Hypothesis

Exp 7A showed factored embeddings guarantee rank alignment but a flat MLP over concatenated tokens can't cheaply compute cross-position rank comparisons. Self-attention makes pairwise token comparison a structural primitive: dot-product between two same-rank card embeddings naturally produces high attention weight, letting the downstream MLP read "which positions share a rank" directly. Factored embeddings + attention should be synergistic.

### Architecture: GolfDQNv3

```
29 card tokens --> factored rank+suit embed --> (batch, 29, emb_dim)
stage          --> stage embed               --> (batch,  1, emb_dim)
                                                  |
                                    + learned positional embeddings (30 pos)
                                                  |
                                    TransformerEncoder(2 layers, 4 heads, pre-norm)
                                                  |
                                    mean pool own cards + holding [0:7]
                                                  |
                                    LayerNorm --> concat deck_remaining scalar
                                                  |
                                    Linear(emb+1, hidden) -> ReLU
                                    Linear(hidden, hidden) -> ReLU
                                    Linear(hidden, 16) -> Q-values
```

- Factored embeddings: `rank_embed(14, emb//2) + suit_embed(5, emb - emb//2)`, same decomposition as v2sf
- 2 attention layers, 4 heads (head_dim=32 at emb=128), 4x FFN expansion, no dropout, pre-norm
- Mean pool positions [0:7] (own 6 grid cards + holding card)
- Deck remaining: scalar, normalized /27.0, concatenated after pooling
- MLP head: 3-layer with mutable hidden_dim, outputs 16 Q-values
- Param count at emb=128: ~740K (hidden=512), smaller than v2s because attention replaces the massive flatten layer

### Config

Identical to Exp 6 Run 2 baseline: eps 1.0->0.05, 20 gens, 12 agents, 500 eps/gen, 20 eval games/matchup, hindsight reward shaping, seed 42.

### Results

| Gen | Best competitive | Best solo | col_matches | rev_col_match | HoF |
|-----|-----------------|-----------|-------------|---------------|-----|
| 1 | 26.06 | 25.66 | 0.10 | 0.02 | 26.06 |
| 2 | 21.62 | 20.99 | 0.13 | 0.02 | 21.62 |
| 3 | 19.67 | 19.07 | 0.17 | 0.04 | 19.67 |
| 5 | 18.71 | 18.50 | 0.20 | 0.10 | 18.71 |
| 7 | 17.89 | 16.90 | 0.26 | 0.06 | 18.02 |
| 10 | 16.14 | 14.84 | 0.26 | 0.07 | 16.41 |
| 12 | 15.71 | 14.27 | 0.30 | 0.07 | 15.76 |
| 15 | 15.59 | 13.97 | 0.28 | 0.09 | 14.73 |
| 17 | 13.73 | 13.03 | 0.28 | 0.05 | 13.73 |
| 18 | 13.80 | 12.64 | 0.29 | 0.07 | 13.73 |
| 20 | 13.24 | 13.19 | 0.25 | 0.05 | 13.24 |

Champion: gen20_agent9 (v3, hidden=512, lr=4.3e-4). Best solo: **12.6** (gen 18).

### Comparison with baselines

| Model | Best solo | HoF (gen 20) | col_matches | rev_col_match |
|-------|-----------|-------------|-------------|---------------|
| v2s baseline (Exp 6 Run 2) | **12.03** | 13.12 | 0.31 | 0.06 |
| v3 self-attention | 12.64 | 13.24 | 0.30 | 0.07 |
| v2sf factored (Exp 7A) | 11.82 | 13.07 | 0.32 | 0.06 |

### Conclusion

v3 matches v2s within noise. The self-attention mechanism did not break the col_matches plateau (~0.30 for both). The learning curve shape, convergence speed, and final performance are essentially identical.

The hypothesis was that attention would make pairwise rank comparison a structural primitive, enabling the agent to discover column matching. In practice, the attention heads had no more success exploiting rank structure than the flat MLP. The tournament converged all agents to hidden=512, lr~4e-4 -- the same regime v2s prefers.

s1_entropy was consistently higher for v3 (2.2-2.4 vs ~2.3 for v2s), suggesting the attention model maintains slightly more diverse action distributions, but this didn't translate to better column matching.

This result, combined with 7A (factored embeddings alone) and 7B (spectral decoupling), further confirms that the col_matches plateau is not a representation or architecture problem. The bottleneck is upstream: the agent doesn't encounter enough column-matching experiences under decaying epsilon to learn the strategy, regardless of how well the network can represent rank relationships.


## Experiment 9: Optuna exploration-focused search (2026-03-14)

### Hypothesis

Experiments 7-8 ruled out representation and architecture as the col_matches bottleneck. The remaining hypothesis: eps-greedy with decaying epsilon doesn't generate enough column-matching experiences before exploitation takes over. The agent learns value swapping first (simpler), locks in, and never discovers systematic column matching.

### Setup

Fixed small v3 (emb=64, hidden=128, ~96K params) and searched 9 exploration-related training parameters using Optuna multi-objective optimization (minimize solo_final@gen10 and solo_mid@gen5).

**Fixed params:** model=v3, emb=64, hidden=[128], gamma=0.99, tau=0.0, mutation_rate=0.3, mutation_sigma=0.2, reward_shaping=hindsight

**Search space:**

| Param | Range |
|-------|-------|
| epsilon_start | [0.5, 1.0] |
| epsilon_end | [0.01, 0.20] |
| lr_low | [1e-5, 1e-3] log |
| lr_high | [1e-3, 1e-2] log |
| updates_per_episode | [2, 16] |
| target_update_interval | [100, 1000] |
| buffer_capacity | {50k, 100k, 200k} |
| batch_size | {128, 256, 512} |
| episodes_per_gen | {250, 500, 1000} |

**Tournament config:** 10 gens, 6 agents, 20 eval games/matchup, max 3 adaptive rounds.

### Results (12 completed trials)

| Trial | solo_final | solo_mid | eps | lr_low | lr_high | ups | target | eps/gen | batch | buf |
|-------|-----------|---------|-----|--------|---------|-----|--------|---------|-------|-----|
| **0** | **12.75** | **12.54** | .80->.06 | 2.7e-4 | 1.2e-3 | 7 | 803 | 1000 | 512 | 50k |
| 2 | 13.58 | 14.42 | .89->.19 | 2.5e-4 | 1.7e-3 | 10 | 334 | 1000 | 128 | 50k |
| 7 | 13.59 | 15.93 | .68->.14 | 6.5e-4 | 4.8e-3 | 5 | 320 | 1000 | 128 | 50k |
| 4 | 13.78 | 18.01 | .75->.06 | 2.7e-5 | 4.8e-3 | 11 | 506 | 1000 | 128 | 50k |
| 5 | 14.07 | 18.16 | .98->.03 | 8.7e-5 | 1.4e-3 | 5 | 140 | 1000 | 256 | 200k |
| 3 | 14.88 | 17.14 | .85->.18 | 1.2e-4 | 5.0e-3 | 4 | 844 | 500 | 512 | 50k |
| 10 | 15.42 | 20.47 | .81->.08 | 2.7e-4 | 9.4e-3 | 3 | 314 | 500 | 512 | 200k |
| 1 | 15.51 | 15.67 | .84->.11 | 1.0e-4 | 3.2e-3 | 12 | 108 | 500 | 512 | 50k |
| 11 | 17.37 | 19.88 | .89->.02 | 4.1e-4 | 2.2e-3 | 11 | 610 | 250 | 128 | 100k |
| 9 | 17.84 | 19.83 | .69->.09 | 7.6e-4 | 4.2e-3 | 6 | 935 | 250 | 128 | 100k |
| 8 | 17.90 | 19.23 | .91->.03 | 1.4e-5 | 7.5e-3 | 7 | 525 | 500 | 128 | 200k |
| 6 | 20.12 | 20.22 | .52->.17 | 7.6e-5 | 1.5e-3 | 2 | 888 | 250 | 512 | 200k |

### Key findings

**1. episodes_per_gen is the dominant factor.** All top-5 trials used eps/gen=1000 (the search ceiling). The 250 and 500 trials cluster at the bottom. More episodes per generation = more chances to discover column-matching transitions at the current epsilon level. The search space capped eps/gen at 1000, so we don't know if higher values would help further.

**2. buffer_capacity=50k beats larger buffers.** Every top-4 trial used 50k. Larger buffers (100k, 200k) dilute recent column-match transitions with older data, slowing learning. Smaller buffers keep the training distribution closer to current policy behavior.

**3. col_matches reached 0.34 but didn't sustain above 0.30.** Trial 0 peaked at col=0.34 in gen 8 but settled back to ~0.30 by gen 10. The solo score improvement (12.75 vs previous best 12.3) came from better training dynamics, not a breakthrough in column matching.

**4. Best hyperparameter profile:**
- epsilon: 0.80 -> 0.06 (high start, low floor)
- lr: 2.7e-4 to 1.2e-3 (narrow, moderate range)
- updates_per_episode: 5-7 (moderate)
- target_update_interval: 300-800 (not too fast)
- batch_size: 512 preferred for solo_final, 128 competitive for mid-training
- buffer: 50k

**5. What doesn't matter much:** epsilon_start is broadly good in [0.7, 0.9] -- all trials had high starts. Very low lr_low (< 1e-4) hurts convergence speed (trial 4: good final but terrible mid). Very fast target updates (108) hurt (trial 1).

### Limitations

- Only 12 trials completed in a 9D space -- sparse coverage
- eps/gen=1000 was the ceiling and clearly optimal, so we can't distinguish "1000 is enough" from "more would be better"
- All trials used the same small v3 architecture (96K params)
- The col_matches plateau (~0.30) was not clearly broken -- needs further investigation

Raw output: `data/optuna_v3_explore_v2_stdout.log`
Optuna DB: `data/optuna_v3_explore_v2.db`

### Round 2: narrowed search (2026-03-15)

Narrowed bounds based on round 1 findings. Key changes:
- eps/gen raised to {1000, 1500, 2000, 3000}
- buffer narrowed to {20k, 30k, 50k}
- batch_size: {512, 1024, 2048}
- target_update: [400, 1000], epsilon_start: [0.7, 1.0], epsilon_end: [0.01, 0.10]
- lr_low: [1e-4, 1e-3], lr_high: [1e-3, 5e-3]

**Results (4 completed trials):**

| Trial | solo_final | solo_mid | eps/gen | buf | batch | eps | lr_low | lr_high | ups | target |
|-------|-----------|---------|---------|-----|-------|-----|--------|---------|-----|--------|
| **1** | **12.55** | **12.58** | 1500 | 50k | 512 | .84->.06 | 1.0e-4 | 4.0e-3 | 7 | 821 |
| 0 | 14.40 | 15.38 | 1000 | 20k | 512 | .81->.07 | 2.1e-4 | 4.6e-3 | 7 | 923 |
| 2 | 16.28 | 16.26 | 1000 | 20k | 512 | .71->.07 | 1.7e-4 | 2.8e-3 | 7 | 858 |
| 3 | 19.08 | 19.72 | 3000 | 30k | 2048 | .99->.05 | 5.8e-4 | 1.9e-3 | 5 | 603 |

**New best: trial 1 at 12.55** (beats round 1's 12.75). The improvement came from eps/gen=1500 with buf=50k.

### Key findings from round 2

**1. eps/gen=1500 > 1000, but 3000 is catastrophic.** The ratio of eps/gen to buffer size matters critically. Trial 1 (1500/50k = 3% fill per gen) worked well. Trial 3 (3000/30k = 100% fill per gen) was the worst -- buffer completely overwritten each gen, destroying continuity. Sweet spot: eps/gen should be ~3% of buffer.

**2. buf=20k consistently underperforms.** Trials 0 and 2 both used buf=20k with 1000 eps/gen (5% fill rate) and scored 14.4 and 16.3. Even with the same eps/gen, 20k is too small -- too little history for stable Q-learning.

**3. batch=2048 hurt.** Trial 3 with batch=2048 and buf=30k samples 7% of the buffer per batch. Combined with high eps/gen, this caused severe overfitting to whatever happened to be in the buffer.

**4. Convergence happens by gen 4-5.** Examining per-gen trajectories across both rounds: solo scores flatten by gen 4-5 and oscillate for the remaining gens. Gens 6-10 contribute noise, not learning. Future searches should use 6 gens to save ~40% runtime.

**5. Stable hyperparameter core emerging:**
- epsilon: ~0.8 -> ~0.06
- updates_per_episode: 7
- target_update_interval: ~800
- lr_low: ~1-3e-4, lr_high: ~1-4e-3
- buf=50k, batch=512

Raw output: `data/optuna_v3_explore_r2_stdout.log`
Optuna DB: `data/optuna_v3_explore_r2.db`

### Round 4: informed ranges + hidden_dim search (2026-03-16)

Tightened search space based on r2 findings. Key changes from r2:
- **lr_high**: [2e-3, 5e-3] (cut the low end that never worked)
- **buffer**: {50k, 75k, 100k} (raised floor -- 20k/30k consistently underperformed)
- **batch**: {256, 512} (dropped 2048 which was catastrophic)
- **eps/gen**: {1000, 1500, 2000} (dropped 3000 which was worst)
- **hidden_dim**: {128, 256, 512} (was fixed at 128 -- now searched)
- Tightened epsilon, lr_low, updates ranges around what worked

Fast trials: 6 gens, 1 agent, 1 adaptive round, 100 solo eval games. ~5-7 min/trial.

**Results (50 completed trials):**

Top 10 by final score:

| Trial | Final | Mid | Hidden | lr_high | buf | eps/gen | batch | updates | target |
|-------|-------|-----|--------|---------|-----|---------|-------|---------|--------|
| **27** | **12.05** | 16.20 | 256 | 2.4e-3 | 100k | 1500 | 512 | 8 | 843 |
| 41 | 12.22 | 15.42 | 512 | 4.0e-3 | 50k | 1500 | 256 | 7 | 755 |
| 33 | 12.48 | 12.70 | 512 | 3.5e-3 | 50k | 1500 | 256 | 9 | 867 |
| **4** | 12.59 | 13.33 | 512 | 4.6e-3 | 50k | 1000 | 256 | 9 | 743 |
| 19 | 12.62 | 13.79 | 512 | 3.9e-3 | 100k | 2000 | 512 | 7 | 974 |
| 25 | 12.66 | 13.48 | 256 | 3.1e-3 | 50k | 2000 | 512 | 9 | 742 |
| 29 | 12.80 | 14.56 | 512 | 2.0e-3 | 100k | 1000 | 512 | 7 | 615 |
| 14 | 12.82 | 19.42 | 256 | 3.8e-3 | 100k | 2000 | 512 | 7 | 650 |
| 36 | 12.93 | 14.20 | 128 | 2.4e-3 | 75k | 1500 | 512 | 9 | 760 |
| 43 | 13.19 | 16.86 | 512 | 3.4e-3 | 100k | 1000 | 512 | 9 | 610 |

**New best: trial 4 peaked at 11.98 at gen 5** (best v3 score ever, beats v2s record of 12.3). Trial 27 best at gen 6 (12.05).

### Per-generation trajectories (top 5)

| Trial | Gen 1 | Gen 2 | Gen 3 | Gen 4 | Gen 5 | Gen 6 | Best | @Gen |
|-------|-------|-------|-------|-------|-------|-------|------|------|
| 27 | 17.96 | 18.01 | 16.20 | 13.83 | 12.98 | **12.05** | 12.05 | 6 |
| 41 | 19.14 | 18.78 | 15.42 | **12.57** | 13.24 | 12.22 | 12.22 | 6 |
| 33 | 18.39 | 15.58 | **12.70** | 13.28 | 13.39 | 12.48 | 12.48 | 6 |
| 4 | 20.98 | 15.42 | 13.33 | 12.99 | **11.98** | 12.59 | 11.98 | 5 |
| 19 | 14.64 | 13.44 | 13.79 | 12.71 | 12.78 | **12.62** | 12.62 | 6 |

All top trials still improving at gen 6 -- 6 generations is not enough for convergence. Trial 4 hit 11.98 at gen 5 then regressed to 12.59, suggesting training instability.

### Regression analysis: training instability

Many trials showed significant regression from peak score. Of 50 trials, ~1/3 lost 3+ points from their best generation.

Worst regressions:

| Trial | Best | Final | Regression |
|-------|------|-------|------------|
| 23 | 17.2 | 25.3 | +8.1 |
| 46 | 14.3 | 21.1 | +6.9 |
| 40 | 15.0 | 21.9 | +6.8 |
| 24 | 15.7 | 22.4 | +6.7 |
| 10 | 17.2 | 23.8 | +6.7 |

**Correlation of hyperparameters with regression:**

| Param | Correlation | Interpretation |
|-------|-------------|----------------|
| episodes_per_gen | -0.257 | More data per gen = more stable |
| batch_size | -0.235 | Larger batches = lower gradient variance |
| hidden_dim | -0.142 | Larger models slightly more stable |
| epsilon_start | -0.143 | Higher initial exploration slightly helps |

Comparing stable (regression < 1) vs unstable (regression > 3) trials:

| Param | Stable mean | Unstable mean | Delta |
|-------|-------------|---------------|-------|
| episodes_per_gen | 1500 | 1000 | -54% |
| batch_size | 512 | 256 | -52% |
| hidden_dim | ~350 | ~270 | -22% |

**The two dominant stability factors are episodes_per_gen and batch_size.** Fast-learning configs (eps/gen=1000, batch=256) can hit great peak scores (trial 4: 11.98) but are prone to catastrophic regression. Stable configs (eps/gen=1500, batch=512) converge more slowly but hold their gains (trial 27: monotonic descent to 12.05).

### Key findings from round 4

**1. hidden_dim matters.** 256 and 512 dominate the top 10 (9/10). Only one 128 trial cracked the top 10 (trial 36 at 12.93). The v3 architecture benefits from larger hidden dims despite having only 1 agent per trial.

**2. eps/gen=1500 is the sweet spot for stability.** Top 3 finals all used 1500. It balances data volume (enough experiences per gen) with buffer freshness (3% fill at buf=50k, 1.5% at 100k).

**3. lr_high range is broad.** 2.0e-3 to 4.6e-3 all work in the top 10. No need to narrow further.

**4. buf=50k and 100k both work.** No clear winner -- probably depends on eps/gen ratio.

**5. The stability-performance tradeoff.** Trial 4 (batch=256, eps/gen=1000) hit the highest peak but regressed. Trial 27 (batch=512, eps/gen=1500) had the best final score with monotonic improvement. For a longer run, trial 27's stable config is the better foundation.

### Recommended config for extended training

Based on trial 27 (best final, stable trajectory, still improving at gen 6):

```
model=v3, hidden=256, emb=64
eps/gen=1500, buf=100k, batch=512
eps=0.87->0.05, lr=8.3e-5 to 2.4e-3
updates=8, target_update=843
gamma=0.99, tau=0.0, reward_shaping=hindsight
```

Optuna DB: `data/optuna_v3_explore_r4.db`
Trial data: `data/optuna_trials/v3-explore-r4/`

## Experiment 10: Extended v3 training with Cyclic Epsilon Annealing (2026-03-16)

![Exp 10 training progress](figures/exp10_cyclic_epsilon.png)

*Top: best solo [R,H,R] score per generation crossing the improved heuristic line (~10.52) around cycle 5. Middle: col_matches climbing from ~0.25 to ~0.73, rev_replace declining slightly. Bottom: epsilon schedule with 7 warm restarts (irregular early boundaries from manual resumes).*

### Setup

Extended training run using trial 27's hyperparameters (best stable config from Exp 9 r4):

```
model=v3, hidden=256, emb=64, population=8
eps/gen=1500, buf=100k, batch=512
eps=0.868->0.051, lr=8.3e-5 to 2.4e-3
updates=8, target_update=843
gamma=0.99, tau=0.0, reward_shaping=hindsight
eval_games=50/matchup, solo_eval=200 games
```

### Discovery: Cyclic Epsilon Annealing

The initial run used `--generations 20` with linear epsilon decay 0.868 -> 0.051. When the run completed, we resumed with a higher `--generations` total (40, 60, 100, 150), which recomputed the linear schedule over the new total. This implicitly created **warm restarts** for epsilon: each resume jumped epsilon back up, then decayed it again over more generations.

The epsilon schedule (linear: `eps_start + progress * (eps_end - eps_start)`, where `progress = (gen-1) / (total-1)`):

| Resume | Total gens | Epsilon at resume | Decays to |
|--------|-----------|-------------------|-----------|
| Initial | 20 | 0.868 (gen 1) | 0.051 (gen 20) |
| Gen 21 | 40 | 0.449 | 0.051 (gen 40) |
| Gen 57 | 100 | 0.406 | 0.051 (gen 100) |
| Gen 101 | 150 | 0.320 | 0.051 (gen 150) |

Each cycle: the agent re-explores with elevated epsilon, then consolidates during low-epsilon exploitation. The key insight is that knowledge gained during exploitation **persists** through the re-exploration phase -- the agent doesn't forget column matching when epsilon goes back up. Each subsequent low-epsilon phase achieves a higher performance floor.

This is analogous to **SGDR (Stochastic Gradient Descent with Warm Restarts)** by Loshchilov & Hutter, but applied to exploration rate rather than learning rate. Cyclic LR helps escape loss landscape minima; cyclic epsilon helps escape **strategy minima** by forcing re-exploration after the agent has consolidated a new behavioral level.

As far as we can find, this specific technique (cycling epsilon-greedy with warm restarts in value-based RL) has not been published. Related work:
- "Cyclic Exploration and Exploitation in Surprise Minimizing RL" (IEEE, 2025) -- cycles exploration/exploitation phases via intrinsic reward weighting, not epsilon
- Cyclical Learning Rates in RL (2024) -- applies SGDR to LR in deep RL, not epsilon

### Results: full trajectory (150 generations, 3 cycles)

Selected generations showing key milestones:

| Gen | Eps | Solo | HoF | col | rev | rcm | Cycle |
|-----|-----|------|-----|-----|-----|-----|-------|
| 1 | 0.83 | 16.55 | 17.74 | 0.233 | 0.423 | 0.053 | initial |
| 5 | 0.69 | 11.98 | 13.51 | 0.293 | 0.305 | 0.050 | initial |
| 20 | 0.34 | 12.20 | 12.67 | 0.317 | 0.229 | 0.057 | initial |
| *-- resume to 40 gens, eps 0.34 -> 0.45 --* | | | | | | | |
| 39 | 0.14 | 11.50 | 12.27 | 0.356 | 0.232 | 0.063 | 1st decay |
| 40 | 0.13 | 11.49 | 12.27 | 0.374 | 0.223 | 0.068 | 1st decay |
| *-- resume to 100 gens, eps 0.13 -> 0.41 --* | | | | | | | |
| 52 | 0.07 | 11.09 | 12.27 | 0.413 | 0.243 | 0.105 | 2nd decay |
| 80 | 0.22 | 11.47 | 12.12 | 0.390 | 0.197 | 0.110 | 2nd re-explore |
| 92 | 0.12 | 11.12 | 12.12 | 0.468 | 0.184 | 0.099 | 2nd decay |
| 99 | 0.06 | 10.86 | 12.12 | 0.424 | 0.219 | 0.058 | 2nd decay |
| *-- resume to 150 gens, eps 0.05 -> 0.32 --* | | | | | | | |
| 107 | 0.29 | 10.91 | 12.08 | 0.458 | 0.197 | 0.089 | 3rd re-explore |
| 121 | 0.21 | 11.06 | 12.08 | 0.482 | 0.185 | 0.136 | 3rd re-explore |
| 130 | 0.16 | 10.64 | 11.34 | 0.534 | 0.183 | 0.130 | 3rd decay |
| 134 | 0.14 | 10.46 | 11.34 | 0.577 | 0.194 | 0.136 | 3rd decay |
| 137 | 0.12 | 10.37 | 11.34 | 0.587 | 0.198 | 0.139 | 3rd decay |
| 140 | 0.11 | 10.32 | 11.34 | 0.608 | 0.174 | 0.147 | 3rd decay |
| 142 | 0.09 | 10.22 | 11.34 | 0.601 | 0.178 | 0.144 | 3rd decay |
| 145 | 0.08 | 10.17 | 11.34 | 0.618 | 0.170 | 0.108 | 3rd decay |
| 148 | 0.06 | **9.67** | 10.88 | **0.632** | 0.181 | 0.149 | 3rd decay |
| 150 | 0.05 | 10.03 | 10.88 | 0.612 | 0.192 | 0.139 | 3rd decay |

### Per-cycle analysis

| Cycle | Gens | Eps range | Solo mean | Col mean | rcm mean | Best solo |
|-------|------|-----------|-----------|----------|----------|-----------|
| Initial | 4-20 | 0.72-0.34 | 12.52 | 0.281 | 0.055 | 11.98 |
| 1st decay | 36-56 | 0.16-0.06 | 11.65 | 0.362 | 0.072 | 11.09 |
| 2nd re-explore | 57-84 | 0.41-0.18 | 11.57 | 0.362 | 0.076 | 11.24 |
| 2nd decay | 85-100 | 0.18-0.05 | 11.30 | 0.410 | 0.084 | 10.86 |
| 3rd re-explore | 101-129 | 0.32-0.17 | 11.17 | 0.443 | 0.101 | 10.70 |
| 3rd decay | 130-150 | 0.16-0.05 | 10.52 | 0.584 | 0.136 | **9.67** |

Each low-epsilon phase produces a step improvement in both solo score and col_matches. The 3rd cycle delivered the largest col jump: +0.17 (0.41 -> 0.58), suggesting the gains are *accelerating* for col even as solo gains diminish.

### The col_matches plateau is broken

The col_matches plateau at 0.30 that persisted through Experiments 6-8 (20 generations each) was not a hard representational limit -- it was a training duration problem. The agent needed:

1. **Enough generations** to accumulate column-matching knowledge through sparse reinforcement
2. **Low epsilon** to exploit that knowledge (col improvement correlated with epsilon decay below 0.15)
3. **Cycling** to re-explore with accumulated knowledge and consolidate at a higher level

Col trajectory: 0.23 (gen 1) -> 0.28 plateau (gen 4-20) -> 0.37 (gen 40) -> 0.41 (gen 90) -> 0.47 (gen 92) -> **0.63 (gen 148)**. The agent now has better column matching than the base heuristic (0.53).

### Behavioral evolution

The agent's strategy has evolved through four distinct phases:

1. **Gen 1-10:** Discovers value swapping (replacing high revealed cards with lower ones). rev=0.40 -> 0.25, col stays at random-adjacent levels (~0.23).

2. **Gen 10-40:** Refines value swapping, begins discovering column matching. col slowly climbs 0.28 -> 0.37. rcm stays low (~0.06) -- column matches are incidental, not targeted.

3. **Gen 40-100:** Column matching becomes deliberate. rcm rises from 0.06 to 0.10+. rev *decreases* (0.23 -> 0.19) -- the agent is being more selective about when to replace revealed cards, targeting column matches specifically rather than pure value swaps.

4. **Gen 100-150:** Column matching dominates. col surges from 0.45 to 0.63, surpassing the heuristic's 0.53. rcm reaches 0.15 -- nearly 1 in 6 revealed-card replacements creates a column match. rev continues declining (0.19 -> 0.17) as the agent becomes increasingly selective. The agent has discovered and is exploiting the combined strategy (value swapping + column matching) that no earlier experiment achieved.

### Comparison with baselines

| Agent | Score | col | rev | rcm | Strategy |
|-------|-------|-----|-----|-----|----------|
| Heuristic | 14.0 | 0.53 | 0.00 | 0.00 | Column matching at unrevealed only |
| Exp 6 DQN (20 gens) | 12.3 | 0.30 | 0.28 | 0.06 | Value swapping + some col matching |
| Exp 10 DQN (100 gens) | 10.9 | 0.45 | 0.20 | 0.10 | Both strategies, increasingly targeted |
| **Exp 10 DQN (150 gens)** | **9.67** | **0.63** | **0.18** | **0.15** | **Both strategies, col > heuristic** |
| Improved heuristic | 10.52 | 0.70 | 0.33 | n/a | Both strategies combined (hardcoded, [R,H,R] config) |

The agent has **already beaten the improved heuristic** (9.67 vs 10.52 in the same `[player, R, H, R]` eval config). Note: the improved heuristic scores 8.1 when evaluated against three random opponents `[improved, R, R, R]` -- an easier table than the DQN's eval which includes one heuristic opponent. All scores in this table use the `[player, R, H, R]` config. The remaining behavioral gap is rev_replace (0.18 vs 0.33) and col (0.63 vs 0.70) -- the DQN is finding a path to the same behavioral profile through learning rather than hard-coded rules.

### The bitter lesson, revisited

Experiment 6's conclusion asked whether the col_matches plateau was "a bug in the training signal, a representation gap, or just insufficient scale." Experiments 7-8 ruled out representation and architecture. Experiment 10 answers: **it was insufficient scale**, but with a nuance.

Naive scaling (more generations with monotonic epsilon decay) doesn't work -- col was flat from gen 7 to gen 20 in every experiment. The agent needed *cyclic* scaling: repeated passes through exploration-exploitation phases, each building on the previous cycle's consolidated knowledge. This is a form of curriculum that emerges naturally from the epsilon schedule rather than being explicitly designed.

Sutton's bitter lesson holds, but with a refinement: the general method (DQN + epsilon-greedy) does work, given correct training signals (Exp 5-6) and sufficient *structured* compute (cyclic annealing, not just more of the same). The 150-gen cyclic run uses ~20x the compute of the original 20-gen experiment, but the improvement is not 20x -- it's the cycling structure that unlocks it.

### Extended training: cycles 5-7 (gens 151-300)

Training continued beyond the 150-gen result with three more cycles:

| Cycle | Gens | Eps restart | Best solo [R,H,R] | Best col |
|-------|------|------------|-------------------|---------|
| 5 | 151-200 | 0.252 | 9.349 | 0.73 |
| 6 | 201-250 | 0.212 | **9.101** | 0.75 |
| 7 | 251-300 | 0.185 | 9.15 | 0.73 |

**Diminishing returns:** Per-cycle solo improvement was -1.19 (cycle 4), -0.33 (cycle 5), -0.24 (cycle 6), +0.05 (cycle 7). Cyclic epsilon annealing has reached its ceiling for this configuration. Col_matches also plateaued at 0.70-0.75 after cycle 5.

**Final all-cycle summary:**

| Cycle | Gens | Best solo [R,H,R] | Col end | Improvement |
|-------|------|-------------------|---------|-------------|
| 1 | 1-20 | 11.98 | 0.32 | baseline |
| 2 | 21-40 | 11.49 | 0.37 | -0.49 |
| 3 | 57-100 | 10.86 | 0.46 | -0.63 |
| 4 | 101-150 | **9.67** | 0.61 | **-1.19** |
| 5 | 151-200 | 9.35 | 0.70 | -0.32 |
| 6 | 201-250 | 9.10 | 0.73 | -0.25 |
| 7 | 251-300 | 9.15 | 0.71 | +0.05 |

**Beating the improved heuristic on its own benchmark (2026-03-18):**

The original 8.1 improved heuristic score was measured in `[player, R, R, R]` (3 random opponents). Running the gen 272 champion (mid-cycle 7) in the same config:

| Agent | Score vs [R,R,R] |
|-------|-----------------|
| Improved heuristic (hardcoded) | 8.10 |
| **DQN gen 272 champion** | **7.92** |

The DQN beats the improved heuristic on both eval configs. The remaining behavioral gap is rev_replace (0.18-0.20 vs 0.33) -- the agent is column matching well but not yet replacing revealed cards as aggressively as the improved heuristic. Cyclic epsilon alone is not closing this gap; a targeted intervention is needed.

**Visualization:** `scripts/plot_training_progress.py` generates a 3-panel plot (solo score, behavioral metrics, epsilon schedule) from `metrics_log.jsonl`. The plateau from cycle 5 onward is clearly visible.

### Data preservation

All artifacts saved in `data/exp9_v3_extended/`:
- `metrics_log.jsonl` -- per-gen metrics
- `config.json` -- hyperparameters + resume history (7 cycles)
- `resume_r1.log` through `resume_r7.log` -- full training logs per resume (7 cycles, 300 gens total)
- Per-generation directories with agent checkpoints and summaries
- `champion.pt`, `hall_of_fame.pt` -- best models
- Total size: ~1.5 GB (dominated by per-gen checkpoints)

---

## Experiment 11: Programmatic Cyclic Epsilon Annealing (350 gens, 7 cycles)

**Status: complete (7 cycles, 343 gens reached as of 2026-03-28)**

### Motivation

Exp 10 used ad-hoc cyclic epsilon annealing via manual resumes — each resume recomputed the linear schedule, accidentally creating warm restarts. Exp 11 makes this **programmatic**: a single launch with `--cycle-length 50 --generations 350` runs 7 complete cycles without intervention. The goal is to confirm the technique is robust and to push further with more uniform cycles.

The same config as Exp 10 (Optuna r4 trial 27) is used throughout, with the only structural change being explicit cycle control.

### Command

```
uv run python -u -m src.tournament \
  --model-variant v3 --hidden-dim-choices 256 --embedding-dim 64 \
  --population-size 8 --generations 350 --cycle-length 50 \
  --episodes-per-gen 1500 --buffer-capacity 100000 --batch-size 512 \
  --epsilon-start 0.868 --epsilon-end 0.051 \
  --lr-range 8.3e-5 0.0024 --updates-per-episode 8 \
  --target-update-interval 843 --gamma 0.99 --reward-shaping hindsight \
  --eval-games-per-matchup 50 --solo-eval-games 200 --max-train-rounds 3 \
  --wandb-project golf-dqn --wandb-run-name exp11-cyclic-7cycles \
  --output-dir data/exp11_cyclic
```

### Key difference from Exp 10

Exp 10: 7 manual resumes over ~2 weeks, each recomputing the epsilon schedule from scratch. The cycle boundaries were irregular (20, 20, 43, 50, 50, 50, 50 gens).

Exp 11: Single launch, `--cycle-length 50` produces uniform 50-gen cycles. Epsilon resets to `eps_start` at the start of each cycle and decays to `eps_end` by cycle end. No manual intervention.

### Per-cycle results

| Cycle | Gens | Best competitive | Best solo [R,H,R] | Col end | Rev end |
|-------|------|-----------------|-------------------|---------|---------|
| 1 | 1-50 | 10.38 | 9.36 | 0.80 | 0.19 |
| 2 | 51-100 | 10.11 | 9.18 | 0.81 | 0.16 |
| 3 | 101-150 | 9.99 | 9.00 | 0.79 | 0.18 |
| 4 | 151-200 | 9.96 | 9.02 | 0.74 | 0.17 |
| 5 | 201-250 | 10.12 | **8.82** | **0.84** | 0.16 |
| 6 | 251-300 | 10.19 | 8.88 | 0.85 | 0.15 |
| 7 | 301-343 | 10.25 | 8.98 | 0.84 | 0.15 |

Best solo score: 8.82 at gen 225 (cycle 5). Best col: 0.89 at gen 328 (cycle 7).

### Final evaluation (2026-03-28, 5000 games x 9 holes)

Rigorous comparison of Exp 11 champion (gen343), Exp 11 HoF (gen58), and Exp 10 champion (gen300) using `scripts/eval_compare.py`:

| Agent | [R,H,R] | [R,R,R] | col | rev |
|-------|---------|---------|-----|-----|
| **Exp 11 champion (gen343)** | **9.61** | **8.02** | **0.82** | 0.14 |
| Exp 11 HoF (gen58) | 9.63 | 8.16 | 0.74 | 0.19 |
| Exp 10 champion (gen300) | 9.79 | 8.37 | 0.70 | 0.19 |
| Improved heuristic | 10.52 | 8.10 | 0.70 | 0.33 |
| Base heuristic | 14.00 | -- | 0.53 | 0.00 |

### Observations

**Exp 11 beats all baselines in both eval configs.** The champion edges out the improved heuristic in [R,R,R] (8.02 vs 8.10) and dominates in the harder [R,H,R] config (9.61 vs 10.52). Clear improvement over Exp 10: ~0.18 pts in solo, ~0.35 pts vs random, driven primarily by col_matches jumping from 0.70 to 0.82.

**Programmatic cycling works as well as manual resumes.** Uniform 50-gen cycles produce consistent improvement without Exp 10's irregular boundaries. Col continues to improve beyond Exp 10's 0.73 plateau, confirming Exp 10 didn't run long enough rather than hitting a hard ceiling.

**Rev_replace is diverging from the heuristic.** Rev dropped from 0.19 (Exp 10) to 0.14 (Exp 11 champion) -- the agent is winning through column matching rather than revealed-card replacement. The gap to the heuristic's 0.33 is widening even as overall score improves. The agent has found an alternative strategy that doesn't rely on rev_replace.

**HoF (gen58, early cycle 2) is surprisingly competitive** with the gen343 champion despite much lower col (0.74 vs 0.82). It compensates with higher rev (0.19 vs 0.14), suggesting the population explored diverse strategies before converging on the col-heavy approach.

**Cyclic epsilon annealing is saturated.** Cycles 6-7 show no improvement over cycle 5 in solo score. Further gains will require a different intervention -- either targeted reward shaping for rev_replace, or a fundamentally different exploration strategy.

**Open question: are we near the theoretical floor?** The HoF agent from early cycle 2 (gen58) matches the gen343 champion in score despite very different behavioral profiles (col 0.74/rev 0.19 vs col 0.82/rev 0.14). This suggests the agent may be near-optimal and the remaining variance is card luck, not decision quality. A possible test: build a perfect-information oracle (scripted solver that sees the full deck order and all hidden cards) and measure the gap between its score and the DQN's ~9.6. If the oracle only achieves ~9, the game is effectively solved. If it gets ~7, there's room to improve. Even a greedy oracle (locally optimal swaps with full visibility) would give a useful lower bound.

### Data

Artifacts saved in `data/exp11_cyclic/`:
- `metrics_log.jsonl` -- per-gen metrics
- `config.json` -- hyperparameters
- Per-generation directories with agent checkpoints
- `champion.pt`, `hall_of_fame.pt` -- best models
- wandb run: `exp11-cyclic-7cycles`

## Experiment 12: Belief-augmented Improved Heuristic (2026-04-07)

### Motivation

Exact Bayesian beliefs are tractable in Golf: under shuffle-once-and-deal, every unobserved card from a player's POV has the same uniform-over-unobserved-multiset posterior, so the belief is a single 52-bit "ever observed" mask. Issue #13 framed this as the central baseline. The simplest way to bolt the belief onto an existing strong player is to take the improved heuristic and replace its hard-coded constants with belief-derived versions.

Two changes were tried:
- **Stage 0 cutoff**: replace constant `4` with `E[score(unknown card)]` from the current belief (~5.46 with full deck).
- **Stage 1 layout scoring**: replace `compute_score` (face-down slot = 0) with `expected_score` (face-down slot = belief-multiset draw, with exact sample-without-replacement column-match math).

The new module is `src/bayes_optimal.py` with belief tracker, expected_score helper, both stage functions, an equivalence regression test, and a `use_belief` flag for ablation.

### Methodology

To compare players fairly we built two pieces of infrastructure:

1. **Multi-player simulator support** (`src/vectorized_golf.py`). `reset_games` gains an `n_players` kwarg (default 4, max 8 from a single deck), and the state tracks a `discard_buried` mask and `deck_size` so we can reshuffle the discard pile back into the deck when it empties. Required for 5+ player games where 9-hole hands exhaust the deck.

2. **Seat-cycling evaluation** (`scripts/seat_cycling.py`). Enumerates every distinct seat permutation of a roster and aggregates per-label scores. Without this, seat-0 first-mover advantage and follower-disadvantage contaminate single-config evaluations.

### Final evaluation (1000 games per permutation x 9 holes)

| Roster | n_players | Bayes avg | Improved avg | gap |
|---|---|---|---|---|
| 1B vs 3I | 4 | 12.244 | 11.693 | **+0.551** (Bayes worse) |
| 1B vs 4I | 5 | 14.007 | 13.726 | **+0.281** (Bayes worse) |
| 1B vs 5I | 6 | 16.559 | 16.110 | **+0.449** (Bayes worse) |

| Solo vs all-random | n_players | Bayes | Improved |
|---|---|---|---|
| 1 player vs random fillers | 4 | 9.781 | 8.772 |
| 1 player vs random fillers | 5 | 9.845 | 9.218 |
| 1 player vs random fillers | 6 | 10.708 | 10.182 |

### Ablation (4-player [B,R,R,R], 2000 games x 9 holes, single seed)

`bayes_stage0` and `bayes_stage1` accept a `use_belief` flag. With `use_belief=False` they reproduce the improved heuristic byte-for-byte (asserted by `tests/test_bayes_optimal.py::test_bayes_no_belief_equals_improved_heuristic`). Toggling each independently:

| s0 | s1 | seat-0 score |
|---|---|---|
| heuristic | improved (compute_score) | **8.183** |
| bayes (E[unknown] cutoff) | improved | 8.603 (+0.42) |
| heuristic | bayes (expected_score) | 8.729 (+0.55) |
| bayes | bayes | 9.288 (+1.10) |

Both belief modifications independently make the player worse, and the errors stack roughly additively.

### Observations

**The naive belief-augmentation is strictly worse than the improved heuristic** under apples-to-apples seat cycling, in every player count tested. The earlier informal "Bayes wins by 0.6" finding (single-seed, seat-0 only) was a seat artifact: with seat-cycling, the apparent advantage disappears and reverses.

**The gap does not shrink with more players.** The 4p gap is +0.55, the 6p gap is +0.45. Hypothesis: more observability through more players would help if the belief inference were the bottleneck. It isn't.

**The bug is conceptual, not arithmetic.** The belief tracker is correctly calibrated -- on 1.4M decisions, the posterior `P(deck draw score < face card score)` matches the empirical rate to within 0.002 in every probability bin. The implementations of `expected_unknown_score`, `expected_score`, and `multiset_by_rank` are right. What's wrong is *what we substituted them in for*.

**Stage 0**: replacing `cutoff=4` with `E[unknown]=5.46` is the wrong substitution. The constant 4 in `heuristic_stage0` is not an EV estimate -- it is a behavior threshold encoding the **optionality value of drawing** (drawing lets you see-then-decide; the face card is committed). The right Bayesian generalization of a threshold comparison `face_score < 4` is a **tail probability** `P(deck_draw_score < face_score) > some threshold`, not a posterior mean. With `E[unknown]=5.46`, the player takes face cards 46% of the time (16+8 ranks qualify) vs the heuristic's 31% (only the 4 lowest ranks), and the data shows taking more is worse.

**Stage 1**: `compute_score` (face-down slot = 0) and the constant `RANK_CUTOFF=4` for "big improvement" are tightly coupled. `expected_score` puts every face-down slot at ~5.46, so revealing any slot via place or flip looks like an automatic ~5.46 improvement. The `RANK_CUTOFF=4` "big improvement" trigger then fires on essentially every action, turning the player into "place anywhere that reveals a slot". The cutoff threshold doesn't transfer between scoring scales.

The lesson: hand-tuned heuristic constants encode policy, not estimation. Replacing them with a "more correct" Bayesian expectation breaks the policy unless the threshold is recalibrated against the new scale.

### Open questions (resolved in Exp 12b)

1. **Tail-probability rule for stage 0.** A take-face decision rule based on `P(draw_score < face_score)` and the belief's column-match probability against hidden cards (`P(face rank matches one of own hidden slots)`) is the obvious next experiment. Both quantities are computable from the same multiset and neither resorts to point estimates.
2. **Posterior-aware hidden column match.** The current `bayes_stage0` only checks rank-match against *revealed* own cards (same as IH). It does not consider P(face rank matches a hidden slot), which is exactly the channel a belief should exploit and the only place IH cannot reach by construction.

### Data

- `src/bayes_optimal.py` -- belief tracker, expected_score, bayes_stage0/1, use_belief ablation flag
- `tests/test_bayes_optimal.py` -- 14 unit tests including the IH equivalence regression test
- `scripts/seat_cycling.py` -- seat-cycled head-to-head with multi-player support
- `scripts/log_stage0.py` -- per-decision posterior logger and calibration check
- `scripts/analyze_bayes.py` -- belief snapshots and column-match opportunity counts
- Reproduce: `uv run python -m scripts.seat_cycling --roster B,I,I,I --games-per-perm 1000 --holes 9 --seed 0`

## Experiment 12b: 1-Step Lookahead Bayes Player (2026-04-23)

### Motivation

Exp 12 showed that injecting belief-derived expected values into the improved heuristic's hard-coded thresholds is fundamentally broken: `expected_score` puts hidden slots at ~5.46 instead of 0, and the `RANK_CUTOFF=4` threshold doesn't transfer between scoring scales. All three belief-augmented variants (B, B2, B3) scored worse than the plain improved heuristic.

The core lesson: hand-tuned heuristic constants encode policy, not estimation. Replacing them with Bayesian expectations breaks the policy unless thresholds are recalibrated. Rather than recalibrating, the 1-step lookahead sidesteps thresholds entirely: enumerate all legal actions, score each resulting layout with `expected_score`, and pick the action minimizing expected final score. Zero tunable parameters.

### Design

**Stage 1 (placement/discard decision):** For each of the 6 placement positions, simulate placing the held card there and compute `expected_score` of the resulting layout. Compare the best placement against the current expected score (the discard+flip baseline). Place if placement improves expected score; otherwise discard+flip first unrevealed slot.

Key insight: discard+flip doesn't change E[final score]. By the law of iterated expectations, revealing a hidden card reduces variance but not the mean. So the discard+flip expected score equals `current_e` exactly. This eliminates the need to enumerate hidden-card possibilities for flip actions.

**Stage 0 (take/draw decision):** Two branches:
- *Take face card:* face card is known; simulate optimal stage-1 placement to get `e_take`.
- *Draw from deck:* for each of the 13 possible ranks weighted by belief probability, simulate optimal stage-1 placement with adjusted multiset (one fewer card of drawn rank). `e_draw = sum_r P(r) * min(best_placement_r, current_e_r)`.

Pick whichever gives lower expected final score.

**Cheating verification:** A scramble test confirms the lookahead does not peek at hidden card values. `expected_score` computes `ranks` for all 6 slots but only uses them in branches gated by the `revealed` mask. Shuffling true card values at unrevealed positions produces identical decisions across 200 games (`test_lookahead_does_not_peek_at_hidden_cards`).

### Results

**Solo eval (seat-0 only, 5000 games x 9 holes):**

| Agent | [player,R,H,R] | [player,R,R,R] |
|---|---|---|
| Lookahead (L) | **6.88** | **6.45** |
| DQN champion (Exp 11, gen343) | 9.59 | 8.05 |
| Improved heuristic | 10.52 | 8.10 |
| Base heuristic | 14.00 | -- |
| Random | -- | 31.1 |

**Seat-cycled: L vs D vs I vs R (24 seatings x 1000 games x 9 holes):**

| Label | Avg score/hole |
|---|---|
| L | **9.03** |
| D | 11.14 |
| I | 11.90 |
| R | 32.71 |

**Seat-cycled: L vs D vs 2R (12 seatings x 500 games x 9 holes):**

| Label | Avg score/hole |
|---|---|
| L | **8.42** |
| D | 9.95 |
| R | 32.29 |

**Seat-cycled: L vs 3I (4 seatings x 1000 games x 9 holes):**

| Label | Avg score/hole |
|---|---|
| L | **8.93** |
| I | 12.04 |

The lookahead beats the DQN champion by 2.7 points in solo [R,H,R], 1.5 points in seat-cycled L-vs-D-vs-2R, and 2.1 points in the 4-player L-vs-D-vs-I-vs-R matchup. It dominates in every single seating permutation.

### Analysis

The DQN champion trained for 343 generations with cyclic epsilon annealing and ~740K parameters. The lookahead has zero learned parameters -- it is pure search over the exact posterior with a 1-step horizon.

The lookahead's advantage comes from two sources:
1. **Stage 0:** It considers the full distribution of possible draws and their optimal placements, including column-match probabilities against hidden slots. The IH uses a single constant cutoff; the DQN learned a policy but can't explicitly compute the posterior.
2. **Stage 1:** It compares every placement against the true expected score of not placing, rather than using a crude threshold. This is especially powerful when the held card is mediocre -- the IH and DQN both make binary "place or flip" decisions with limited nuance.

The 1-step horizon is a limitation: the lookahead doesn't consider how today's action affects future turns (e.g., setting up a column match for later). Despite this, the information advantage from exact Bayesian beliefs overwhelms the DQN's implicit multi-step planning.

### Data

- `src/bayes_optimal.py` -- `lookahead_stage0`, `lookahead_stage1`, `_best_placement_score` helper
- `tests/test_bayes_optimal.py` -- 21 tests including scramble-based cheating verification
- `scripts/seat_cycling.py` -- label `"L"` for the lookahead player
- Reproduce solo: `uv run python -m src.bayes_optimal --player lookahead --games 5000 --holes 9 --eval-config R,H,R --seed 0`
- Reproduce seat-cycled: `uv run python -m scripts.seat_cycling --roster L,D,I,R --dqn-checkpoint data/exp11_cyclic/champion.pt --games-per-perm 1000 --holes 9 --seed 0`

## Experiment 14: Win-Bonus Training Run and DQN Champion Benchmark (2026-04-26)

### Motivation

Exp 11 saturated at cycle 5 with no further improvement in cycles 6-7. Two interventions were bundled into Exp 14:

1. **Win bonus / loss penalty** in the terminal reward (`--win-bonus` flag in `src/tournament.py`). Hypothesis: augmenting the hindsight score reward with a signal for winning the hole (lowest score) would drive the agent toward more decisive play.
2. **Extended training** with the same cyclic epsilon config (350 gens, 7 cycles, same Optuna r4 hyperparameters as Exp 11).

### Command

```
uv run python -u -m src.tournament \
  --model-variant v3 --hidden-dim-choices 256 --embedding-dim 64 \
  --population-size 8 --generations 350 --cycle-length 50 \
  --episodes-per-gen 1500 --buffer-capacity 100000 --batch-size 512 \
  --epsilon-start 0.868 --epsilon-end 0.051 \
  --lr-range 8.3e-5 0.0024 --updates-per-episode 8 \
  --target-update-interval 843 --gamma 0.99 --reward-shaping hindsight \
  --eval-games-per-matchup 50 --solo-eval-games 200 --max-train-rounds 3 \
  --wandb-project golf-dqn --wandb-run-name exp14-win-bonus \
  --output-dir data/exp14_win_bonus
```

### Findings: win bonus was not exercised

Post-hoc inspection revealed that `EvalResult.wins/losses/draws` fields are never incremented anywhere in `tournament.py` — `win_rate` always returns 0. This means `eval/best_win_rate=0` throughout and the win-bonus reward shaping may have defaulted to 0 (not logged in the run header). The win-bonus experiment effectively ran as a straight replication of Exp 11's config. This is a known bug; the fields need to be populated from the per-matchup score matrix.

### Per-cycle results

| Cycle | Gens | Best competitive (in-training) | Col end |
|-------|------|-------------------------------|---------|
| 1 | 1-50 | 11.28 | ~0.82 |
| 2 | 51-100 | 9.36 | ~0.90 |
| 3 | 101-150 | 9.45 | ~0.91 |
| 4 | 151-200 | 9.18 | ~0.93 |
| 5 | 201-250 | 9.44 | ~0.93 |
| 6 | 251-300 | 9.28 | ~0.94 |
| 7 | 301-350 | 9.21 | ~0.92 |

Score plateau at ~9.2-9.5 from cycle 2 onward (same saturation pattern as Exp 11, but starting a cycle earlier). Col_matches reached 0.88-0.97 in late training, up from Exp 11's 0.82, with rev_replace holding at 0.14-0.16.

**Note on in-training rankings**: The in-training competitive scores (50 games per matchup vs a mixed league) produced an incorrect ranking. `gen96_agent4` (hall-of-fame from cycle 2, best_ever=9.227) appeared marginally better than the final champion `gen350_agent4` (9.214) in the noisy in-training eval. Rigorous seat-cycling reversed this: gen350 beats gen96 by 0.22 strokes/hole (see below).

### Seat-cycling benchmark results

All seat-cycling evals use 12 distinct permutations × 2000 games × 9 holes for the 4-player D1,D2,R,R roster, eliminating seat bias.

**Exp14 champion vs Exp11 champion (direct):**

| Agent | Avg score/hole |
|---|---|
| **DQN Exp14 gen350** | **8.904** |
| DQN Exp11 gen343 | 9.926 |
| Random | 32.161 |

Exp14 wins by **1.02 strokes/hole** in every permutation.

**Exp14 gen350 vs Exp14 gen96 (internal):**

| Agent | Avg score/hole |
|---|---|
| **DQN gen350** | **8.812** |
| DQN gen96 | 9.029 |

The final champion beats the hall-of-fame agent by 0.22 strokes/hole. In-training rankings were misleading — the 50-game noisy evals suggested gen96 was best, but rigorous evaluation reverses the order.

### Full benchmark: Lookahead vs both DQNs (5-player, 120 perms × 1000 games × 9 holes)

Roster `L, D1, D2, I, R` (5-player game). Chance baseline = 20%.

| Agent | Avg score/hole | Win rate |
|---|---|---|
| Lookahead (L) | **10.10** | **34.1%** |
| DQN Exp14 (D2) | 10.64 | 29.9% |
| DQN Exp11 (D1) | 12.25 | 22.2% |
| Improved Heuristic (I) | 13.43 | 20.6% |
| Random (R) | 33.23 | 0.6% |

Exp14 DQN is now solidly above the 20% chance baseline and above the improved heuristic in win rate. The gap to Lookahead narrowed from 2.11 strokes/hole (Exp11 vs L in the 4-player L,D,I,R eval from Exp 12b) to 0.54 strokes/hole here. The rank distribution of kept cards shows Exp14 closely matching Lookahead's low-card retention profile; Exp11 is noticeably weaker in shedding high cards.

For the 4-player L,D,I,R roster (same as Exp 12b baseline, now with Exp14 DQN):

| Agent | Avg score/hole | Win rate |
|---|---|---|
| Lookahead (L) | **9.113** | **48.3%** |
| DQN Exp14 | 9.582 | 38.5% |
| Improved Heuristic | 12.015 | 13.2% |
| Random | 32.639 | 0.0% |

The gap to Lookahead is 0.47 strokes/hole and 9.8 percentage points in win rate. In individual seat configurations, Exp14 DQN beats Lookahead outright (e.g. `D,R,L,I`: D=9.47/45%, L=9.40/49.9% — near dead heat; `R,I,D,L`: D=9.50/38.8%, L=9.99/30.1% — DQN wins both metrics). The aggregate gap is heavily seat-driven: Lookahead at seat 0 achieves 57-78% win rate vs the DQN's 45-68%.

### Observations

**Exp14 is the new DQN champion**, beating Exp11 by 1.02 strokes/hole under rigorous seat-cycled evaluation. The improvement is driven by higher col_matches (0.88-0.97 vs 0.82) with similar rev_replace (0.14-0.16).

**In-training rankings are unreliable.** The 50-game competitive evals used during training misranked gen96 above gen350. For any future experiment, champion selection should always be validated with at least 1000-game seat-cycled eval before declaring a winner.

**The DQN has substantially closed the gap to Lookahead.** Exp11 trailed by 2.1 strokes/hole in the 4-player L,D,I,R roster; Exp14 trails by 0.47. The rank distribution of kept cards shows Exp14 now matching Lookahead's strong preference for low cards, suggesting the DQN has learned a similar card-value hierarchy implicitly.

**Cyclic epsilon still saturates at cycle 2.** The same pattern as Exp 11: cycle 1 sees rapid improvement from random-play baseline, cycle 2 brings the major jump, cycles 3-7 plateau. More cycles do not help. The next experiments should explore different interventions: reward shaping that rewards win rate directly (requires fixing the EvalResult bug first), or curriculum changes.

### Data

- `data/exp14_win_bonus/` — per-generation checkpoints, `champion.pt`, `hall_of_fame.pt`, `metrics_log.jsonl`
- **Champion checkpoint**: `data/exp14_win_bonus/gen_350/gen350_agent4.pt`
- wandb runs: `exp14-win-bonus`, `exp14-win-bonus-resumed`
- Seat-cycling results: `data/seat_cycling_exp11_vs_exp14_gen350.txt`, `data/seat_cycling_exp14_gen96_vs_gen350.txt`, `data/seat_cycling_exp14_vs_lookahead.txt`
- Figure: `data/figures/agent_comparison.png` (score distributions, kept-card rank distributions, win rates for L/Exp14/Exp11/I/R)
- Reproduce champion benchmark: `uv run python -m scripts.seat_cycling --roster D1,D2,R,R --dqn1-checkpoint data/exp11_cyclic/champion.pt --dqn2-checkpoint data/exp14_win_bonus/gen_350/gen350_agent4.pt --games-per-perm 2000 --holes 9`
- Reproduce full comparison: `uv run python -m scripts.agent_comparison --dqn1-checkpoint data/exp11_cyclic/champion.pt --dqn1-name "DQN Exp11" --dqn2-checkpoint data/exp14_win_bonus/gen_350/gen350_agent4.pt --dqn2-name "DQN Exp14" --games 1000 --holes 9`

---

## Experiment 15: AlphaZero-style Distillation from Bayes Lookahead (2026-04-29)

### Motivation

The policy audit (see below) established that DQN and Lookahead disagree most at stage-1 placement decisions when the board is largely hidden (avg revealed fraction 0.391 at disagreement vs 0.769 at agreement). Lookahead outscores DQN by 1.18 strokes/hole on every hole containing at least one disagreement. The relationship is one-sided: DQN has no information advantage over Lookahead. The AlphaZero framing: treat Bayes Lookahead as the oracle search and distill its policy into the DQN network, without adding belief features as inputs.

### Policy audit

`scripts/policy_audit.py` runs N games twice with the same seed (once Lookahead-driven, once DQN-driven) and queries both agents at every player-0 decision, recording per-action expected scores (BL) and Q-values (DQN).

Results for `gen350_agent4` (2000 games × 9 holes):

| | Stage 0 (take/draw) | Stage 1 (placement) |
|---|---|---|
| Agreement rate | 92.3% | 49.5% |
| Spearman ρ | 0.867 | 0.849 |

Stage-0 disagreements (13,495 total): DQN prefers take 10,746 times vs Lookahead's 2,749 — DQN is more aggressive about taking discards. Stage-1 disagreements cluster at low-revealed boards (avg 0.391 vs 0.769 for agreements). On all holes containing ≥1 disagreement (100% of holes), Lookahead scores 8.549 vs DQN's 9.730 — a 1.18 gap. The DQN has no countervailing advantage.

### Distillation

`scripts/distill_from_bayes.py` implements AlphaZero-style distillation:

1. **Expert data collection**: Run BL for N games, recording `(obs, stage, bl_per_action_scores, valid_mask)` at every player-0 decision. BL drives the trajectory; opponents use heuristic.
2. **Pairwise ranking loss**: For each pair of valid actions (i, j) where BL strictly prefers i (lower expected score), penalise if DQN Q-value ordering disagrees. Loss = `mean(relu(margin - (Q[i] - Q[j])))`. Scale-free; no temperature tuning required.
3. **Fine-tune from gen350_agent4**: 30 epochs, 2000 games × 9 holes = 1.44M decisions, lr=1e-4, margin=0.1.

Distillation results:

| Epoch | Val loss | Agreement |
|---|---|---|
| 1 | 0.0060 | 89.4% |
| 6 | 0.0027 | 89.5% |
| 30 | 0.0021 | 89.9% |

Converged by epoch 6. Agreement rose from ~71% (92.3% stage-0 / 49.5% stage-1 combined) to ~90%. Saved to `data/exp14_win_bonus/distilled.pt`.

**Seat-cycling: distilled vs gen350 (D1,D2,R,R, 12 perms × 2000 games):**

| Agent | Avg score/hole | Win rate |
|---|---|---|
| gen350 (D1) | **8.731** | 56.4% |
| distilled (D2) | 9.349 | 43.6% |

The distilled model is 0.62 strokes/hole *worse* than its starting point. Root cause: the pairwise ranking loss changed Q-value orderings without preserving their scale. Q-values carry absolute magnitude information used in TD bootstrapping; disrupting orderings also shifts magnitudes, corrupting the learned value function.

### RL resume from distilled checkpoint

Following the AlphaZero approach (distillation provides initialisation, not the final policy), RL training was resumed from `distilled.pt` using the full Exp 14 config: v3, hidden=256, population=8, cyclic ε 0.868→0.051, 50 gens/cycle, hindsight reward, win bonus=0.3. Population bootstrapped via `data/exp15_distilled/gen_0/` with 8 agents initialised from the distilled weights and LRs sampled from the Exp 14 range.

**Key observation**: Col_matches was 0.81–0.93 at ε=0.868 in generation 1 — the distilled column-matching behaviour survived re-exploration. In a cold-start run (Exp 14), col_matches was ~0.30–0.50 at the same stage. The distillation accelerated convergence of this behaviour by roughly one full cycle.

**Per-cycle best solo scores (in-training, 500-game eval):**

| Cycle | Gens | Best solo |
|---|---|---|
| 1 | 1–50 | **8.157** (gen 46) |
| 2 | 51–100 | 8.328 (gen 81) |
| 3 | 101–150 | 8.171 (gen 124) |
| 4 | 151–200 | 8.253 (gen 192) |
| 5 | 201–250 | 8.390 (gen 227) |
| 6 | 251–300 | 8.209 (gen 274) |
| 7 | 301–350 | 8.320 (gen 341) |

Cycle 1 best solo of 8.157 nearly equals Exp 14's all-time best (8.18) reached after 350 gens — a strong head start. However, all 7 cycles show no consistent improvement: the model oscillates in the range 8.157–8.390 throughout. Two contributing factors:

1. **Hindsight reward saturation**: Hindsight shaping rewards column matches; since col_matches starts high from distillation, the reward signal driving improvement in Exp 14 is already weak from the first generation.
2. **Belief bottleneck reasserts**: The distillation taught the DQN *what* BL does but not *why* — BL's stage-1 decisions depend on the posterior over hidden cards, which is not in the observation. Under RL the model cannot improve on hidden-card placement decisions beyond the distilled prior.

### Plateau analysis and AlphaZero-consistent options

Three principled paths forward:

**A — Recurrent architecture (most AlphaZero-consistent):** Replace the feedforward v3 network with a Transformer-over-history that processes the sequence of per-turn observations. The network learns to maintain implicit belief state in its attention heads without any external tracker at inference time. Requires retraining from scratch; the distilled weights would not transfer.

**B — Outcome-supervised distillation (longer horizon):** Instead of distilling from BL's 1-step expected-score targets, generate targets by running BL to hole completion and using the actual final-score delta as the supervision signal. The DQN learns from outcomes rather than belief-dependent intermediate scores, staying within the raw-observation constraint.

**C — Belief features as input (pragmatic):** Add the belief posterior (unobserved card count by rank, ~13 floats) as extra observation tokens. At inference time the belief tracker runs as a lightweight companion module. Not strictly AlphaZero-consistent (the network is no longer a pure function of the raw game state), but directly addresses the root cause identified by the audit.

### Final benchmark: Exp15 champion vs Exp14 champion (12 perms × 2000 games × 9 holes)

| Agent | Avg score/hole | Win rate |
|---|---|---|
| Exp14 gen350_agent4 (D1) | **8.907** | 52.2% |
| Exp15 champion (D2) | 9.129 | 47.8% |

Exp15 loses by 0.22 strokes/hole. The RL resume recovered from the distillation regression (0.62 behind → 0.22 behind) but never surpassed the Exp14 starting point. The distillation head start did not produce a higher ceiling: the same belief bottleneck reasserted itself, and the plateau across all 7 cycles confirms the feedforward v3 architecture without belief information cannot improve beyond ~8.15 solo regardless of initialisation.

### Conclusion

Distillation from Bayes Lookahead accelerates early learning (col_matches high from gen 1, near-champion solo score by gen 46) but does not raise the asymptotic ceiling. The ceiling is the belief bottleneck: without knowledge of the posterior over hidden cards, the DQN cannot improve stage-1 placement at low-revealed boards regardless of how it is initialised or how many cycles it trains.

### Data

- `scripts/policy_audit.py` — decision-level DQN vs Lookahead comparison (agreement rate, Spearman ρ, counterfactual scores)
- `scripts/distill_from_bayes.py` — expert data collection + pairwise ranking loss fine-tuning
- `data/exp14_win_bonus/distilled.pt` — distilled checkpoint (gen350_agent4 fine-tuned on BL trajectories)
- `data/exp15_distilled/` — RL training from distilled checkpoint, per-generation checkpoints, `metrics_log.jsonl`
- `data/seat_cycling_gen350_vs_distilled.txt` — gen350 vs distilled (pure distillation result)
- `data/seat_cycling_exp14_vs_exp15.txt` — Exp14 vs Exp15 final champion benchmark
- `data/figures/policy_audit.png` — 6-panel policy audit figure

