-- Research-run service schema (Cloudflare D1 / SQLite).

CREATE TABLE projects (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  slug            TEXT NOT NULL UNIQUE,
  name            TEXT NOT NULL,
  description     TEXT NOT NULL,
  repo            TEXT NOT NULL,            -- owner/name of the code the runs execute
  protocol        TEXT NOT NULL,            -- benchmark protocol id runs are scored on
  metric_name     TEXT NOT NULL,
  lower_is_better INTEGER NOT NULL DEFAULT 1,
  run_price_cents INTEGER NOT NULL,         -- one research run costs this much
  balance_cents   INTEGER NOT NULL DEFAULT 0, -- donated money not yet spent on a run
  agent_brief     TEXT NOT NULL,            -- what the maintaining agent knows about the project
  created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE donations (
  id           TEXT PRIMARY KEY,
  project_id   INTEGER NOT NULL REFERENCES projects(id),
  donor        TEXT NOT NULL,
  amount_cents INTEGER NOT NULL,
  note         TEXT,
  payment_ref  TEXT NOT NULL,               -- 'stub:...' until a real processor is wired in
  created_at   TEXT NOT NULL DEFAULT (datetime('now'))
);

-- status: funded -> planning -> planned -> dispatched -> running -> evaluated -> analyzing -> reported
--   planning and analyzing are held while the agent works (they double as locks).
--   A run whose result is an error ends in failed, still with a report.
CREATE TABLE runs (
  id           TEXT PRIMARY KEY,
  project_id   INTEGER NOT NULL REFERENCES projects(id),
  donation_id  TEXT REFERENCES donations(id),
  status       TEXT NOT NULL,
  spec_source  TEXT NOT NULL,               -- 'agent' (the maintaining agent plans it) or 'donor'
  spec_json    TEXT,                        -- what to run; see service/runner/execute.py
  title        TEXT,
  hypothesis   TEXT,
  dispatcher   TEXT,
  external_ref TEXT,                        -- e.g. GitHub Actions run URL
  result_json  TEXT,
  metric_value REAL,
  win_rate     REAL,
  simulated    INTEGER NOT NULL DEFAULT 0,  -- 1 when the mock dispatcher produced the numbers
  report_md    TEXT,
  error        TEXT,
  attempts     INTEGER NOT NULL DEFAULT 0,
  created_at   TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at   TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX runs_project_status ON runs(project_id, status);

CREATE TABLE run_events (
  id      INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id  TEXT NOT NULL REFERENCES runs(id),
  at      TEXT NOT NULL DEFAULT (datetime('now')),
  type    TEXT NOT NULL,
  message TEXT NOT NULL
);
CREATE INDEX run_events_run ON run_events(run_id);

CREATE TABLE leaderboard (
  id           INTEGER PRIMARY KEY AUTOINCREMENT,
  project_id   INTEGER NOT NULL REFERENCES projects(id),
  protocol     TEXT NOT NULL,
  name         TEXT NOT NULL,
  metric_value REAL NOT NULL,
  win_rate     REAL,
  source       TEXT NOT NULL,               -- 'baseline' or 'run'
  run_id       TEXT REFERENCES runs(id),
  simulated    INTEGER NOT NULL DEFAULT 0,
  notes        TEXT,
  created_at   TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX leaderboard_project_protocol ON leaderboard(project_id, protocol);
