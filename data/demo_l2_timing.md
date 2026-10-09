# L2 timing replay

This is a real simulator trace from `scripts/demo_bounded_lookahead.py`, seed 0, game batch 512, roster `L2,R,R,R`. The acting player's hidden card identity is intentionally omitted.

At round 5, before the stage-1 decision:

| Observable field | Value |
|---|---|
| Own cards | 39, 1, 35, 15, 29, hidden |
| Revealed | yes, yes, yes, yes, yes, no |
| Holding | 8 |
| Discard top | 32 |
| Last round active | no |
| Deck remaining | 16 |

The two policies see the same state but choose differently:

| Policy | Action |
|---|---|
| L | Discard and flip slot 5 — finishes the layout |
| L2 | Place card 8 into revealed slot 2 — keeps slot 5 hidden |

At L2's next turn, the observable state is:

| Observable field | Value |
|---|---|
| Own cards | 39, 1, 8, 15, 29, hidden |
| Revealed | yes, yes, yes, yes, yes, no |
| Holding | none (stage 0) |
| Last round active | no |
| Deck remaining | 15 |

This directly demonstrates the mechanism: L would finish immediately; L2 preserves one hidden card and receives another turn. The L2 path's eventual final score for this game was 10.0.

Reproduce:

```bash
uv run python -m scripts.demo_bounded_lookahead \
  --seed 0 --games 512 --output data/demo_l2_timing.json
```

Machine-readable trace: `data/demo_l2_timing.json`.
