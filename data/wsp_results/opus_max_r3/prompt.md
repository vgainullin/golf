# Weight-space programming: improve the golf DQN champion by editing its weights

You are a programmer whose program is a neural network's weights. There is no optimizer: you
read the weights, understand what they compute, and write new values directly.

## The network

`data/wsp/opus_max_r3/base.pt` is the DQN champion for 4-player Golf (Exp 14 gen 350, agent 4). It is a
torch checkpoint: `{"model_state_dict": ..., "config": {...}}`. Model class `GolfDQNv3` in
`src/dqn_offline.py`, built by `make_model` in `src/tournament.py`. Input is
`get_observation_v2` in `src/vectorized_golf.py` (30 card tokens, 52 = unknown) plus the stage;
output is Q-values for 16 actions (stage 0: 0 take discard, 1 draw; stage 1: 2-7 place held card
at slot 0-5, 9-14 discard held card and flip slot 0-5). The agent takes the argmax over valid
actions. Rules and scoring are in `README.md`; the game engine is `src/vectorized_golf.py`.

## The goal

Lower its average score per hole (lower is better) in seat-cycled games against the Bayes
lookahead (L), the improved heuristic (I) and random (R). The champion scores about 9.58 on the
held-out benchmark; the lookahead scores about 8.75. Ideas worth knowing are in
`docs/experiments.md` and `docs/beyond-heuristic-rl.md` (for example, where the DQN loses
points to the lookahead).

## Starting point: an already edited network

`base.pt` is not the raw champion. It is the Exp 14 champion after earlier weight-space programming
sessions (Experiment 16 in `docs/experiments.md`), applied in this order: `opus_max_b25`, `opus_max_r2`. Each
session's notes are in `data/wsp/opus_max_r3/prior/<session>/NOTES.md`; its analysis and edit scripts are next
to them for reference only (they expect their old run-directory layout).

On the held-out benchmark the original champion scores 9.55 and `base.pt` scores 8.847;
the lookahead scores about 8.75. Improve on `base.pt`. You may build on the earlier edits, revise
them or remove them.

## Your tools and limits

- Work in Python (`.venv/bin/python`, torch and numpy on CPU) inside `data/wsp/opus_max_r3`. Load the
  weights, analyse them, edit tensors, save candidates as `data/wsp/opus_max_r3/cand_NN.pt` in the same
  format with the same config and tensor shapes (no new layers or units).
- You may evaluate the network on inputs you construct by hand to understand it.
- The ONLY way to see it play is a forward pass. There are two kinds, each with its own budget:
  - **Practice pass** (budget 25, ~3 min):
    `taskset -c 0-3 .venv/bin/python -m scripts.wsp_pass data/wsp/opus_max_r3/cand_NN.pt --run-dir data/wsp/opus_max_r3`
    7,200 games (300 per seating, 24 seatings) on the practice seed.
  - **Confirm pass** (budget 3, ~7 min): the same command with `--confirm`. 24,000 games
    (1,000 per seating) on a separate seed, about 1.8× less noisy than a practice pass. Use it to check a candidate you
    already believe in, not to screen.
  - Both print the candidate's average score per hole, win rate, and its difference from `base.pt`
    on the same tier with a paired standard error over the 24 seatings. `base.pt` was scored on
    both tiers before you started (`base_practice.json`, `base_confirm.json`), so you do not
    spend a pass on it.
  - Every candidate is dealt identical cards per (seating, hole): the deck order of each hole is
    fixed by the seed, whatever happened in earlier holes. Re-scoring the same weights gives the
    same number, and paired differences reflect changed decisions, not different deals.
  - A practice pass writes `data/wsp/opus_max_r3/trace_NNN.jsonl`: 2 games from every seating, so the
    candidate sits in every seat. Each decision has the observation, Q-values, valid actions,
    chosen action, `unseen_by_rank` (how many of each rank C has not seen this hole, from the
    same history-aware tracker the lookahead uses) and `expected_unknown_score`. Each hole ends
    with its scores by seat. Decisions in games that are already over are left out. Results so
    far are in `data/wsp/opus_max_r3/passes.jsonl` and `data/wsp/opus_max_r3/confirms.jsonl`.
  - Near-ties on practice are noise: picking the best of several candidates that tie on practice
    overstates its gain. The confirm tier exists to check the one you pick.
- Forbidden: gradient descent or any optimizer, training scripts (`src.tournament`,
  `src.optuna_search`, distillation), running the game engine, simulator, opponents or seat-cycling
  code yourself, editing `wsp_pass.py`, `budget.json`, the `base_*.json` files or the ledgers, and reading anything
  under `data/` outside `data/wsp/opus_max_r3`. Searching over many random weight perturbations is also out:
  each edit should come from a reason you can state.

## Deliverables in `data/wsp/opus_max_r3`

- `final.pt`: your best candidate. It must be byte-identical to a candidate you ran a confirm pass
  on (`cp` it), or a copy of `base.pt` if nothing beat it.
- `NOTES.md`: what you found the network computes (name specific units, heads or directions),
  every edit you made and why, which tensors and how many weights changed, the pass results,
  and what you would try next. Be plain about what did not work.

The held-out score of `final.pt` is measured after you finish, on games you never see.
