import type { Env, LeaderboardRow, Project, Run, RunStatus } from "./types";

export function newId(prefix: string): string {
  const bytes = crypto.getRandomValues(new Uint8Array(6));
  return prefix + "_" + Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
}

export async function listProjects(env: Env): Promise<Project[]> {
  const { results } = await env.DB.prepare("SELECT * FROM projects ORDER BY id").all<Project>();
  return results;
}

export async function getProject(env: Env, slug: string): Promise<Project | null> {
  return env.DB.prepare("SELECT * FROM projects WHERE slug = ?").bind(slug).first<Project>();
}

export async function getProjectById(env: Env, id: number): Promise<Project> {
  const p = await env.DB.prepare("SELECT * FROM projects WHERE id = ?").bind(id).first<Project>();
  if (!p) throw new Error(`project ${id} missing`);
  return p;
}

export async function getRun(env: Env, id: string): Promise<Run | null> {
  return env.DB.prepare("SELECT * FROM runs WHERE id = ?").bind(id).first<Run>();
}

export async function listRuns(env: Env, projectId: number, limit = 50): Promise<Run[]> {
  const { results } = await env.DB.prepare(
    "SELECT * FROM runs WHERE project_id = ? ORDER BY created_at DESC, rowid DESC LIMIT ?",
  )
    .bind(projectId, limit)
    .all<Run>();
  return results;
}

export async function runsInStatus(env: Env, statuses: RunStatus[], limit = 20): Promise<Run[]> {
  const qs = statuses.map(() => "?").join(",");
  const { results } = await env.DB.prepare(
    `SELECT * FROM runs WHERE status IN (${qs}) ORDER BY created_at, rowid LIMIT ?`,
  )
    .bind(...statuses, limit)
    .all<Run>();
  return results;
}

/**
 * Move a run from one status to another only if it is still in `from`.
 * This is the concurrency guard: two drivers (cron and a request) can't both
 * take the same step. Returns false if someone else got there first.
 */
export async function transition(
  env: Env,
  id: string,
  from: RunStatus | RunStatus[],
  to: RunStatus,
  fields: Partial<Record<keyof Run, string | number | null>> = {},
): Promise<boolean> {
  const froms = Array.isArray(from) ? from : [from];
  const sets = ["status = ?", "updated_at = datetime('now')"];
  const vals: (string | number | null)[] = [to];
  for (const [k, v] of Object.entries(fields)) {
    sets.push(`${k} = ?`);
    vals.push(v as string | number | null);
  }
  const res = await env.DB.prepare(
    `UPDATE runs SET ${sets.join(", ")} WHERE id = ? AND status IN (${froms.map(() => "?").join(",")})`,
  )
    .bind(...vals, id, ...froms)
    .run();
  const ok = (res.meta.changes ?? 0) > 0;
  if (ok) await logEvent(env, id, "status", `${froms.join("|")} -> ${to}`);
  return ok;
}

export async function logEvent(env: Env, runId: string, type: string, message: string): Promise<void> {
  await env.DB.prepare("INSERT INTO run_events (run_id, type, message) VALUES (?, ?, ?)")
    .bind(runId, type, message)
    .run();
}

export async function runEvents(env: Env, runId: string) {
  const { results } = await env.DB.prepare(
    "SELECT at, type, message FROM run_events WHERE run_id = ? ORDER BY id",
  )
    .bind(runId)
    .all();
  return results;
}

export async function leaderboard(
  env: Env,
  project: Project,
  protocol = project.protocol,
): Promise<LeaderboardRow[]> {
  const order = project.lower_is_better ? "ASC" : "DESC";
  const { results } = await env.DB.prepare(
    `SELECT id, protocol, name, metric_value, win_rate, source, run_id, simulated, notes, created_at
     FROM leaderboard WHERE project_id = ? AND protocol = ? ORDER BY metric_value ${order}, id`,
  )
    .bind(project.id, protocol)
    .all<LeaderboardRow>();
  return results;
}
