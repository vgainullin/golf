# Lookahead draw + improved placement

Scored 11.166 against the best entry's 9.582 (DQN champion (Exp14 gen350 agent4)); it does not beat it.

- Run: `run_6a3b7d9ad27b`
- Protocol: seatcycle-v1, 1000 games per seating, 9 holes
- avg_score_per_hole: **11.166**
- Win rate: 19.1%

## Hypothesis

Most of the lookahead player's edge may come from the draw decision. Pair its belief-based stage 0 with the cheap improved-heuristic stage 1 and see how much of the gap closes. (Scripted plan: no ANTHROPIC_API_KEY configured.)

## Findings

Rank 2 of 3 on seatcycle-v1. Opponents in the same games: L 8.388, I 11.338, R 32.675. This report is a template (no ANTHROPIC_API_KEY configured), so it states the numbers without interpreting them.

## Next directions

- Configure ANTHROPIC_API_KEY so the agent can interpret results and plan follow-ups.
