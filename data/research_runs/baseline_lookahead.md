# Bayes lookahead (L) on seatcycle-v1

Baseline row for the research-runs leaderboard, measured 2026-10-01 by the research-runs service's runner.

- Player: the builtin 1-step Bayes lookahead (`src/bayes_optimal.py`, label L), seated as the candidate C
- Protocol: seatcycle-v1, roster C,L,I,R, all 24 seatings x 1000 games x 9 holes
- avg_score_per_hole: **8.751** (8.7508)
- Win rate: 43.9%

Every research run is scored on the same protocol, so this is the number a run has to beat to unseat the champion. The seed lives in `service/migrations/0003_lookahead_baseline.sql`.
