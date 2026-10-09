INSERT INTO projects (slug, name, description, repo, protocol, metric_name, lower_is_better, run_price_cents, agent_brief)
VALUES (
  'golf',
  'Golf card-game AI',
  'Try to beat the best Golf-playing agent. Each run builds or trains one candidate player and scores it in a seat-cycled match against the Bayes lookahead player, the improved heuristic and a random player.',
  'vgainullin/golf',
  'seatcycle-v1',
  'avg_score_per_hole',
  1,
  2000,
  'Golf is a 4-player, 6-card (2x3 grid) card game; lowest score wins; matching ranks in a column cancel. The repo has a vectorized PyTorch engine (src/vectorized_golf.py), population DQN training (src/tournament.py), and a belief-tracking 1-step lookahead player (src/bayes_optimal.py, label L), the strongest known player. Benchmark: the candidate plays as C in roster C,L,I,R across all 24 seatings, 1000 games each, 9 holes; metric is average score per hole (lower is better). Known results on this protocol: the Exp14 DQN champion scores 9.582 (38.5% wins) and the lookahead opponent scores about 9.11 against it. Candidates can be (a) Python agent code defining stage0(state, seat) and stage1(state, seat) that return (N,) long tensors of actions, using helpers from src.vectorized_golf and src.bayes_optimal (stage 0: 0=take the discard, 1=draw from the deck; stage 1: 2-7 place the held card at grid slot 0-5, 9-14 discard it and flip slot 0-5; mask with src.vectorized_golf.get_valid_action_mask), or (b) a DQN trained with src.tournament flags. CPU runs are capped at population 8, 30 generations, 1000 episodes per generation.'
);

INSERT INTO leaderboard (project_id, protocol, name, metric_value, win_rate, source, notes)
SELECT id, 'seatcycle-v1', 'DQN champion (Exp14 gen350 agent4)', 9.582, 0.385, 'baseline',
       'From data/seat_cycling_exp14_vs_lookahead.txt: played as D in L,D,I,R, 24 seatings x 1000 games x 9 holes.'
FROM projects WHERE slug = 'golf';
