## Starting point: an already edited network

`base.pt` is not the raw champion. It is the Exp 14 champion after one earlier weight-space
programming session (Experiment 16 in `docs/experiments.md`). That session changed 22,876 weights:
a reveal-premium correction circuit built from repurposed transformer FFN units and dead MLP units,
plus a take-bias fix. Its notes are in `{RUN_DIR}/prior/NOTES.md`. Its analysis and edit scripts
are in `{RUN_DIR}/prior/` for reference only; they expect their old run-directory layout.

On the held-out benchmark the original champion scores 9.55 and `base.pt` scores 8.89; the
lookahead scores about 8.75. Improve on `base.pt`. You may build on the earlier edits, revise them
or remove them.
