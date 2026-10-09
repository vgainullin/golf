-- The Bayes lookahead player (L) is golf's champion, so it is the baseline to
-- beat and the starting point every run modifies.

UPDATE projects SET agent_brief =
  'Golf is a 4-player, 6-card (2x3 grid) card game; lowest score wins; matching ranks in a column cancel. The repo has a vectorized PyTorch engine (src/vectorized_golf.py), population DQN training (src/tournament.py), and the belief-tracking 1-step lookahead player (src/bayes_optimal.py, label L), the current champion. Benchmark: the candidate plays as C in roster C,L,I,R across all 24 seatings, 1000 games each, 9 holes; metric is average score per hole (lower is better). Known results on this protocol: the lookahead itself scores 8.751 (43.9% wins) and the Exp14 DQN champion 9.582 (38.5%). A candidate is Python agent code defining stage0(state, seat) and stage1(state, seat) that return (N,) long tensors of actions, with optional reset(state, seat) and observe(state, seat) hooks for belief tracking (stage 0: 0=take the discard, 1=draw from the deck; stage 1: 2-7 place the held card at grid slot 0-5, 9-14 discard it and flip slot 0-5; mask with src.vectorized_golf.get_valid_action_mask). The lookahead''s stage 1 places the held card where it minimizes expected score (expected_score, _best_placement_score) and otherwise flips the first hidden slot; its stage 0 compares the expected score of taking the discard against the belief-weighted value of drawing.'
WHERE slug = 'golf';

INSERT INTO leaderboard (project_id, protocol, name, metric_value, win_rate, source, notes)
SELECT id, 'seatcycle-v1', 'Bayes lookahead (L)', 8.7508, 0.4391, 'baseline',
       'Builtin L scored as C in C,L,I,R, 24 seatings x 1000 games x 9 holes (measured 2026-10-01).'
FROM projects
WHERE slug = 'golf'
  AND NOT EXISTS (
    SELECT 1 FROM leaderboard l JOIN projects p ON p.id = l.project_id
    WHERE p.slug = 'golf' AND l.protocol = 'seatcycle-v1' AND l.name LIKE 'Bayes lookahead%'
  );
