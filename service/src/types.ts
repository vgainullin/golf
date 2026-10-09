export interface Env {
  DB: D1Database;
  DISPATCHER: "mock" | "queue" | "github";
  GITHUB_REPO: string;
  GITHUB_REF: string;
  AGENT_MODEL: string;
  ALLOW_GPU: string;
  PUBLIC_URL: string;
  ANTHROPIC_API_KEY?: string;
  RUNNER_TOKEN?: string;
  ADMIN_TOKEN?: string;
  GITHUB_TOKEN?: string;
  STRIPE_WEBHOOK_SECRET?: string;
  /** Project credited when a payment names none. */
  DEFAULT_PROJECT?: string;
}

export type RunStatus =
  | "funded"
  | "planning"
  | "planned"
  | "dispatched"
  | "running"
  | "evaluated"
  | "analyzing"
  | "reported"
  | "failed";

export const TERMINAL: RunStatus[] = ["reported", "failed"];

export interface Project {
  id: number;
  slug: string;
  name: string;
  description: string;
  repo: string;
  protocol: string;
  metric_name: string;
  lower_is_better: number;
  run_price_cents: number;
  balance_cents: number;
  agent_brief: string;
  journal_paths: string; // comma-separated repo paths the planner reads
  created_at: string;
}

/** What a run executes. Mirrors service/runner/execute.py. */
export interface RunSpec {
  kind: "agent_code" | "train" | "checkpoint" | "builtin";
  protocol: string;
  compute: "cpu" | "gpu";
  agent_code?: string;
  train?: { args: Record<string, unknown> };
  checkpoint?: { url: string };
  builtin?: { label: string };
}

export interface Run {
  id: string;
  project_id: number;
  donation_id: string | null;
  status: RunStatus;
  spec_source: "agent" | "donor" | "baseline";
  spec_json: string | null;
  title: string | null;
  hypothesis: string | null;
  dispatcher: string | null;
  external_ref: string | null;
  result_json: string | null;
  metric_value: number | null;
  win_rate: number | null;
  simulated: number;
  report_md: string | null;
  plan_review: string | null; // the planner's read of the journal and past runs
  error: string | null;
  attempts: number;
  created_at: string;
  updated_at: string;
}

/** What a runner posts back. Mirrors the JSON execute.py writes. */
export interface RunResult {
  status: "ok" | "error";
  error?: string;
  protocol?: { id: string; games_per_perm: number; holes: number; overridden?: boolean };
  metric?: { name: string; value: number; lower_is_better: boolean };
  win_rate?: number;
  per_label?: Record<string, { avg_score_per_hole: number; win_rate: number }>;
  simulated?: boolean;
  [k: string]: unknown;
}

export interface LeaderboardRow {
  id: number;
  protocol: string;
  name: string;
  metric_value: number;
  win_rate: number | null;
  source: "baseline" | "run";
  run_id: string | null;
  simulated: number;
  notes: string | null;
  created_at: string;
}
