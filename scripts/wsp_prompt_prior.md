## Starting point: an already edited network

`base.pt` is not the raw champion. It is the Exp 14 champion after earlier weight-space programming
sessions (Experiment 16 in `docs/experiments.md`), applied in this order: {PRIOR_RUNS}. Each
session's notes are in `{RUN_DIR}/prior/<session>/NOTES.md`; its analysis and edit scripts are next
to them for reference only (they expect their old run-directory layout).

On the held-out benchmark the original champion scores 9.55 and `base.pt` scores {BASE_HELDOUT};
the lookahead scores about 8.75. Improve on `base.pt`. You may build on the earlier edits, revise
them or remove them.
