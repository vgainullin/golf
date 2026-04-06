# Findings

A short tour of what this project actually learned. The full lab notebook with every dead end is in [`docs/experiments.md`](docs/experiments.md) — this page is the index.

## TL;DR

- Vanilla DQN trained from scratch beats a strong hand-coded Golf heuristic (**9.61 vs 10.52** in `[player, R, H, R]`, **8.02 vs 8.10** in `[player, R, R, R]`) without imitation, demos, or hand-engineered features.
- The breakthrough came from **fixing two bugs in the training signal**, not from a smarter algorithm.
- A third unlock — **cyclic epsilon annealing** — was discovered by accident and broke through what looked like a hard plateau.
- Along the way, the learned representations got monopolized by the easier strategy, which connects to a well-characterized cluster of failure modes (gradient starvation, primacy bias, KataGo's cyclic blind spot).

![Exp 10 training progress](docs/figures/exp10_cyclic_epsilon.png)

*Exp 10: best solo score per generation, behavioral metrics (col_matches, rev_replace), and the cyclic epsilon schedule. The agent crosses the improved-heuristic line (10.52) around cycle 5.*

## The four substantive findings

### 1. Stale `next_obs` in multi-agent turn-based games

In a 4-player turn-based game, three opponents act between two consecutive decisions by the same agent. We were recording the post-action state as `s'`, but the state the agent actually sees on its next turn is the post-*opponents'-actions* state. The TD target `r + γ Q(s')` was being computed against a phantom state that never occurs in actual play. The Bellman backup couldn't converge.

**Fix:** defer transition recording until the agent's next turn. Halved the gap to the heuristic on its own.

**Generalisation:** any turn-based RL setup with intervening opponents (board games, card games, multi-agent grids) has this exact pitfall. The bug is invisible if you only look at single-step environments.

→ [`docs/experiments.md`, Experiment 5](docs/experiments.md)

### 2. Reward bias from observability gaps

Golf scores hidden cards as 0 in the partial observation. When the agent revealed a card by placing the held card at a face-down position, the visible score "increased" — even though the hidden card being replaced was usually worse. The reward function therefore had a **+5.2 systematic bias toward placing at already-revealed positions**.

The agent learned exactly what the reward told it: never place at unrevealed positions (col_matches = 0.12, rev_replace = 0.78). This is the worst possible strategy for Golf, and the agent learned it confidently.

**Fix:** hindsight reward shaping. Compute reward against the *true final score* (treating all cards as revealed for reward purposes only — the policy still acts on partial observations). The agent then learns the actual value of each action.

After this fix, vanilla DQN from scratch beat the heuristic for the first time (12.3 vs 14.0). With cyclic epsilon annealing on top (Exp 10-11) the same algorithm reaches 9.6 — better than the much stronger improved heuristic.

**Generalisation:** any environment where the observation function and the reward function disagree about what counts as "state" — pretty much any partial-information game where the reward is computed from visible state. The fix is to either make the reward observation-consistent or use end-of-episode reward only.

→ [`docs/experiments.md`, Experiment 6](docs/experiments.md)

### 3. MDP diagnostics toolkit

Both bugs above are general failure modes that any RL practitioner could hit. Gymnasium's `env_checker` validates the environment *interface*, but doesn't check whether the *MDP* is correct. We built four pre-training probes that complement Gym's approach:

| Check | What it catches |
|---|---|
| Transition fidelity | Stale `next_obs` (compares immediate vs deferred next-state) |
| Reward-action distribution | Reward bias (groups rewards by action, flags large spreads) |
| Determinism | Hidden state, RNG leaks (two seeded episodes must be identical) |
| Observation sanity | NaN/Inf, shape mismatches, impure observation functions |

Both bugs would have been caught immediately if these had been run before training. No domain knowledge required to read the output.

→ [`src/diagnostics.py`](src/diagnostics.py), [`docs/experiments.md` "MDP Diagnostics Toolkit"](docs/experiments.md)

### 4. Cyclic epsilon annealing breaks strategy plateaus

After fixing the rewards, the agent reached col_matches ≈ 0.30 and stopped. We ruled out representation (factored embeddings, attention), credit assignment (n-step returns), and gradient dynamics (spectral decoupling) as the bottleneck — none of them moved the plateau. See [Experiments 7-8](docs/experiments.md).

The plateau was broken by accident. We re-launched the run with `--generations 40` after a 20-generation run finished, which recomputed the linear epsilon schedule from scratch — implicitly creating a *warm restart*. Each subsequent resume re-explored from a higher consolidated baseline. After 7 cycles, col_matches reached 0.84 and the agent surpassed the improved heuristic.

This is **SGDR (warm-restart LR schedules) applied to exploration rate instead of learning rate**. As far as we can find, this specific technique hasn't been published in this form. The mechanism: each low-epsilon phase consolidates new behavior; each warm restart lets the agent re-explore *with that consolidated knowledge intact*, finding combinations it couldn't find before.

→ [`docs/experiments.md`, Experiments 10-11](docs/experiments.md)

## A connection worth flagging: representation monopolization

Between the bug fixes and the cyclic-epsilon discovery, we spent a few experiments diagnosing *why* the agent kept getting stuck on value-swapping (the easy strategy) and never learning column-matching (the harder one). Embedding analysis showed that the network had learned to encode card *value* but not card *rank* — the imitation model showed within-rank cosine similarity of 0.54 across same-rank cards, the RL model showed 0.15. The RL model had no representational substrate for "is the held card the same rank as a revealed card in column j?".

This is a specific instance of a failure mode that has many names in the literature:

| Name | Authors | What it describes |
|---|---|---|
| Gradient starvation | Pezeshki et al., NeurIPS 2021 | Dominant features absorb the loss and starve secondary features of gradient |
| Simplicity bias | Shah et al., NeurIPS 2020 | SGD learns simpler features first; complex-but-useful features get left behind |
| Primacy bias | Nikishin et al., ICML 2022 | Deep RL agents overfit to early experience and lose the ability to learn from new data |
| Implicit under-parameterization | Kumar et al., ICLR 2021 | Bootstrapped value learning collapses the effective rank of features |
| Loss of plasticity | Abbas et al., Nature 2024 | Continual-learning agents gradually lose the ability to learn at all |

The same phenomenon appears in much bigger systems: KataGo's cyclic-group blind spot (Wang et al., 2023), AlphaStar's strategy collapse (Vinyals et al., 2019), TD-Gammon's doubling weakness, DQN on Montezuma's Revenge. In every case, the dominant strategy requires simpler representational structure, gets discovered first, and locks the embeddings — and the secondary strategy that needs different structure becomes unreachable.

What's interesting about Golf is that it reproduces this failure mode at toy scale (~100K parameters, 16 actions, 30-step episodes). Whatever fix works here is a candidate for fixes that might work on bigger systems. Cyclic epsilon annealing is one such fix.

→ [`docs/experiments.md`, "Representation monopolization: literature and parallels"](docs/experiments.md)

## What this project is *not*

- **Not a SOTA Golf solver.** The improved heuristic is hand-coded with 5 lines of strategy. The DQN beats it by ~0.2 points. There's almost certainly more room.
- **Not a publishable claim about cyclic epsilon annealing.** The technique works on this game. Whether it generalises is a hypothesis, not a result.
- **Not a clean RL tutorial.** It's a lab notebook. The dead ends are part of the value — most RL writeups hide them.

## What's open

See the bottom of the [README](README.md): strategy extraction from the champion, an LLM benchmark leaderboard, and a playable browser version.
