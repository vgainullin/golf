-- The planner reads the project's experiment journal (files in its repo) and
-- the reports of earlier runs before deciding what to try.
ALTER TABLE projects ADD COLUMN journal_paths TEXT NOT NULL DEFAULT '';
ALTER TABLE runs ADD COLUMN plan_review TEXT;

UPDATE projects
SET journal_paths = 'docs/experiments.md,docs/beyond-heuristic-rl.md,data/llm_benchmarks.md'
WHERE slug = 'golf';
