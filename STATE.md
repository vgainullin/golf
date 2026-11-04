# STATE: golf (handoff, 2026-11-04)

## Where things are

- `main` has Experiment 16 (weight-space programming, WSP) and harness v2 (#24, #22), plus
  multi-session continuation and this handoff. CI is green; #23 and #25 fixed two long-standing
  workflow failures.
- Champion: raw Exp 14 `gen350_agent4`, 9.546 per hole held-out (seatcycle-v1, C,L,I,R). Opus
  hand edits, no training:
  - round 1: 8.885
  - round 2: 8.847
  - **round 3 (harness v2): 8.168**, −0.679 ± 0.030 vs round 2, confirm-tier −0.665 ± 0.026. In the
    same games the lookahead scored 8.198.

  Write-up: `docs/experiments.md` Exp 16. Records: `data/wsp_results/`.
- Checkpoints and transcripts are local and git-ignored: `data/wsp/opus_max_b25/`,
  `data/wsp/opus_max_r2/`, `data/wsp/opus_max_r3/` (best: `final.pt` = cand_21, sha256
  `ff3454f0…5d4a`), and `~/wsp_archive/opus_b25/` (the high-effort run).
- Round 3's key finding: going out early was the error. Round 1's reveal penalty was a proxy; a
  go-out penalty plus a count-dependent reveal bonus does better.

## Open choices (V)

- **Publish:** upload round 3's `final.pt` to HF `vgainullin/golf` and add a leaderboard row. 8.168
  would top seatcycle-v1, ahead of the lookahead's 8.751.
- **Replicates:** high vs max effort, 2 runs each from the raw champion. The effort gap is n = 1 per
  level so far; ~$5–40 API-equivalent per run.
- **Round 4:** ideas are in Exp 16's Next list. Harness gap to fix first: `--baseline` should write a
  base trace, so pass 1 isn't spent on base.
- **RL from the edited network:** its Q is no longer in points, so the decision devices need
  re-expressing first.

## Box notes

- GitHub over SSH fails from pop-os (no key on the account). Push over HTTPS:
  `git -c credential.helper= -c credential.helper='!gh auth git-credential' push https://github.com/vgainullin/golf.git <branch>`.
  The gh token has the `workflow` scope.
- Run passes and evals under `taskset -c 0-3`.
