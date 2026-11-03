# Weight-space programming: improve the golf DQN champion by editing its weights

You are a programmer whose program is a neural network's weights. There is no optimizer: you
read the weights, understand what they compute, and write new values directly.

## The network

`{RUN_DIR}/base.pt` is the DQN champion for 4-player Golf (Exp 14 gen 350, agent 4). It is a
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

## Your tools and limits

- Work in Python (`.venv/bin/python`, torch and numpy on CPU) inside `{RUN_DIR}`. Load the
  weights, analyse them, edit tensors, save candidates as `{RUN_DIR}/cand_NN.pt` in the same
  format with the same config and tensor shapes (no new layers or units).
- You may evaluate the network on inputs you construct by hand to understand it.
- The ONLY way to see it play is a forward pass, which spends one unit of a budget of
  {BUDGET} passes:
  `taskset -c 0-3 .venv/bin/python -m scripts.wsp_pass {RUN_DIR}/cand_NN.pt --run-dir {RUN_DIR}`
  It plays 2,400 practice games (100 per seating) and prints the candidate's average score per
  hole, win rate, and its difference from pass 1 with a paired standard error over the 24
  seatings (every pass starts from the same practice seed; decks drift apart once play differs), and writes a trace of 8 games (every decision's
  observation, Q-values, valid actions and chosen action, plus each hole's scores by seat) to
  `{RUN_DIR}/trace_NNN.jsonl`. Results so far are in `{RUN_DIR}/passes.jsonl`. Spend pass 1 on
  `base.pt` itself so you have a baseline.
- Forbidden: gradient descent or any optimizer, training scripts (`src.tournament`,
  `src.optuna_search`, distillation), running the game engine, simulator, opponents or seat-cycling
  code yourself, editing `wsp_pass.py`, `budget.json` or `passes.jsonl`, and reading anything
  under `data/` outside `{RUN_DIR}`. Searching over many random weight perturbations is also out:
  each edit should come from a reason you can state.

## Deliverables in `{RUN_DIR}`

- `final.pt`: your best candidate (copy `base.pt` if nothing beat it).
- `NOTES.md`: what you found the network computes (name specific units, heads or directions),
  every edit you made and why, which tensors and how many weights changed, the pass results,
  and what you would try next. Be plain about what did not work.

The held-out score of `final.pt` is measured after you finish, on games you never see.
