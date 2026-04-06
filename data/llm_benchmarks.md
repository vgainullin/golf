# LLM Golf Benchmarks

Harness: `src/llm_player.py`, seats: [LLM, Random, Heuristic, Random]

## Baselines (500 games x 9 holes)
| Seat      | Avg/hole |
|-----------|----------|
| Heuristic | 14.1     |
| Random    | 32.2     |

## Results

### gemma-2-9b-it (LM Studio, 1 game x 9 holes, 167s)
| Seat      | Avg/hole |
|-----------|----------|
| LLM       | 31.11    |
| Heuristic | 12.89    |
| Random    | 32.44 / 33.00 |

- take_rate: 0.055, rev_replace: 0.019, col_matches: 0.333
- API: 110 calls, 131K input, 660 output tokens
- Invalid actions: 0
- Notes: No reasoning output (6 tokens/call avg). Almost never takes from discard. Plays at random level.

### deepseek-r1-7b v1 (LM Studio, 1 game x 9 holes, 3949s) -- with conversation history + retry
| Seat      | Avg/hole |
|-----------|----------|
| LLM       | 28.11    |
| Heuristic | 11.89    |
| Random    | 31.89 / 33.56 |

- take_rate: 0.452, rev_replace: 0.514, col_matches: 0.222
- API: 239 calls, 473K input, 70K output tokens
- Invalid actions: 106 / 239 (44%)
- Notes: Conversation history confused the model -- echoed stage 1 action IDs during stage 0. 512 max_tokens truncated `<think>` blocks before action output. Retry mechanism inflated call count.

### deepseek-r1-7b v2 (LM Studio, 1 game x 9 holes, 11257s) -- stateless, 4096 max_tokens
| Seat      | Avg/hole |
|-----------|----------|
| LLM       | 29.67    |
| Heuristic | 15.22    |
| Random    | 31.89 / 32.78 |

- take_rate: 0.597, rev_replace: 0.636, col_matches: 0.111
- API: 124 calls, 40K input, 216K output tokens
- Invalid actions: 19 / 124 (15%)
- Notes: Stateless fixed most parse failures (44% -> 15%). But 3 hours for 9 holes due to verbose `<think>` chains (~1700 output tokens/call). Still plays at random level despite reasoning. High take_rate (0.60) but no column match strategy.

### claude-haiku-4.5 (OpenRouter, 1 game x 1 hole, 40s)
| Seat      | Avg/hole |
|-----------|----------|
| LLM       | 37.00    |
| Heuristic | 36.00    |
| Random    | 37.00 / 45.00 |

- take_rate: 0.143, rev_replace: 1.000, col_matches: 0.000
- API: 14 calls, 20K input, 2.5K output tokens
- Notes: Single hole only (verbose system prompt version). Made bad swaps (replaced 2 with A). Needs more holes for meaningful comparison.

### claude-haiku-4.5 (OpenRouter, 5 runs x 1 game x 9 holes, ~560s/run)
Aggregated across 5 sessions (2026-03-29 and 2026-04-01):

| Run                  | LLM   | Heuristic | Random / Random |
|----------------------|-------|-----------|-----------------|
| 20260329_104816      | 13.11 | 11.89     | 28.89 / --      |
| 20260329_114748      | 17.44 |  9.22     | 33.00 / --      |
| 20260329_120555      | 18.33 | 11.67     | 25.00 / --      |
| 20260401_170258      | 12.78 | 13.67     | 29.67 / --      |
| 20260401_232034      | 18.22 | 16.00     | 35.56 / --      |
| **mean**             | **15.97** | 12.49 | 30.42        |

- take_rate: 0.21-0.40, rev_replace: 0.73-0.82, col_matches: 0.00-0.33
- API: ~125 calls/run, 0 invalid actions across all runs
- Notes: Beats random (~30) decisively, but loses to base heuristic (~12.5) in 4 of 5 runs. Very high rev_replace (~0.78) suggests the model latched onto flipping new cards over column matching. col_matches near zero -- no column-match strategy emerged. Single-game variance is high (12.78 to 18.33) so 5-run mean is a rough estimate. Compare to DQN champion at 9.61 in [LLM,R,H,R]-equivalent harder config.
