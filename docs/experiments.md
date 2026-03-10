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
| Improved heur | 8.1 | 0.70 | 0.33 | n/a | Both strategies combined |

The DQN found a **complementary strategy** to the heuristic. The heuristic gets value from column matching (col=0.55) but never touches revealed cards. The DQN gets value from replacing revealed high cards with lower ones (rev=0.28) but doesn't systematically column match. Combining both strategies (as the improved heuristic does at 8.1) is the theoretical ceiling.

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


