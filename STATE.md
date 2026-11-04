# STATE: golf (handoff, 2026-11-04)

## Where things are

- `main` has Experiment 16 (weight-space programming, WSP) and harness v2 (#24, #22), plus
  multi-session continuation and this handoff. CI is green; #23 and #25 fixed two long-standing
  workflow failures.
- Champion: raw Exp 14 `gen350_agent4`, 9.546 per hole held-out (seatcycle-v1, C,L,I,R). Opus
  hand edits brought it to:
  - round 1, effort max: 8.885
  - round 2, effort max, starting from round 1: **8.847**

  Details are in `docs/experiments.md` (Exp 16) and `data/wsp_results/`.
- Checkpoints and transcripts are local and git-ignored: `data/wsp/opus_max_b25/`,
  `data/wsp/opus_max_r2/`, `~/wsp_archive/opus_b25/` (the high-effort run).

## In flight: round 3 (`data/wsp/opus_max_r3/`)

- Launched with:
  `PRIOR_RUN="data/wsp/opus_max_b25 data/wsp/opus_max_r2" scripts/wsp_run.sh data/wsp/opus_max_r2/final.pt 25 opus_max_r3 max`
- **Settings:** effort max, harness v2.
  - Common deals.
  - Practice: 300 games per seating, ~3 min, budget 25.
  - Confirm: 1,000 per seating, ~7 min, budget 3.
  - `final.pt` must be a confirmed candidate.
- **Outputs in the run dir:**
  - `run_meta.json` (`final_confirmed`)
  - `passes.jsonl`, `confirms.jsonl`
  - `heldout_base.json`, `heldout_final.json`
  - `NOTES.md`
  - `opus_max_r3.log` (one level up)
- **Questions it answers:**
  - Can sub-0.1 edits be found and confirmed now that a practice pass resolves about ±0.05?
  - Does the confirm-tier gain match held-out? Common-deals estimates ran 0.03–0.04 more negative
    than held-out on the round-1 → round-2 pair. Watch this.
- **Audit when it ends:**
  - grep the transcript for tool calls touching `data/` outside the run dir
  - check scripts for engine, opponent, optimizer or gradient use
- **Then:** add round 3 to Exp 16 and copy the small records to `data/wsp_results/opus_max_r3/`.

## Open choices (V)

- **Replicates:** high vs max effort, 2 runs each from the raw champion. The effort gap is n = 1
  per level so far.
- **Publish:** upload the WSP finals to HF `vgainullin/golf` and add a leaderboard row (8.847 would
  be the best non-lookahead entry).
- **RL from the edited network:** does training keep the circuits?

## Box notes

- GitHub over SSH fails from pop-os (no key on the account). Push over HTTPS:
  `git -c credential.helper= -c credential.helper='!gh auth git-credential' push https://github.com/vgainullin/golf.git <branch>`.
  The gh token has the `workflow` scope.
- Run passes and evals under `taskset -c 0-3`.
