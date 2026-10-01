/**
 * The run lifecycle. Every step is a guarded status transition, so the cron
 * driver and request handlers can call these concurrently without doubling up.
 */
import { analyze, plan } from "./agent";
import { dispatch } from "./dispatch";
import {
  getProjectById,
  getRun,
  leaderboard,
  listRuns,
  logEvent,
  newId,
  runsInStatus,
  transition,
} from "./db";
import type { Env, Project, Run, RunResult, RunSpec } from "./types";

const MAX_ATTEMPTS = 3;
const STALE_MINUTES = 15;

export interface Submission {
  title?: string;
  hypothesis?: string;
  spec: Partial<RunSpec>;
}

/**
 * Record a donation, add it to the project's pool, and turn every full
 * run_price_cents in the pool into a funded run. Payment is stubbed: the
 * donation is treated as captured.
 */
export async function donate(
  env: Env,
  project: Project,
  donor: string,
  amountCents: number,
  note: string | null,
  submission: Submission | null,
) {
  if (submission && amountCents < project.run_price_cents) {
    throw new HttpError(400, `submitting your own candidate needs a full run (${project.run_price_cents} cents)`);
  }
  const donationId = newId("don");
  await env.DB.batch([
    env.DB.prepare(
      "INSERT INTO donations (id, project_id, donor, amount_cents, note, payment_ref) VALUES (?, ?, ?, ?, ?, ?)",
    ).bind(donationId, project.id, donor, amountCents, note, `stub:${donationId}`),
    env.DB.prepare("UPDATE projects SET balance_cents = balance_cents + ? WHERE id = ?").bind(amountCents, project.id),
  ]);

  const runIds: string[] = [];
  for (;;) {
    const res = await env.DB.prepare(
      "UPDATE projects SET balance_cents = balance_cents - run_price_cents WHERE id = ? AND balance_cents >= run_price_cents",
    )
      .bind(project.id)
      .run();
    if (!res.meta.changes) break;
    const runId = newId("run");
    const own = submission && runIds.length === 0 ? submission : null;
    if (own) {
      const spec = normalizeSubmission(project, own.spec);
      await env.DB.prepare(
        `INSERT INTO runs (id, project_id, donation_id, status, spec_source, spec_json, title, hypothesis)
         VALUES (?, ?, ?, 'planned', 'donor', ?, ?, ?)`,
      )
        .bind(runId, project.id, donationId, JSON.stringify(spec), own.title ?? `Submission from ${donor}`, own.hypothesis ?? note)
        .run();
    } else {
      await env.DB.prepare(
        "INSERT INTO runs (id, project_id, donation_id, status, spec_source) VALUES (?, ?, ?, 'funded', 'agent')",
      )
        .bind(runId, project.id, donationId)
        .run();
    }
    await logEvent(env, runId, "funded", `funded by ${donor} (donation ${donationId})`);
    runIds.push(runId);
  }
  const balance = await env.DB.prepare("SELECT balance_cents FROM projects WHERE id = ?")
    .bind(project.id)
    .first<number>("balance_cents");
  return { donation_id: donationId, runs: runIds, balance_cents: balance };
}

function normalizeSubmission(project: Project, s: Partial<RunSpec>): RunSpec {
  const kind = s.kind;
  if (kind === "agent_code" && typeof s.agent_code === "string" && s.agent_code.trim()) {
    return { kind, agent_code: s.agent_code, protocol: project.protocol, compute: "cpu" };
  }
  if (kind === "checkpoint" && typeof s.checkpoint?.url === "string") {
    return { kind, checkpoint: { url: s.checkpoint.url }, protocol: project.protocol, compute: "cpu" };
  }
  throw new HttpError(400, "submission.spec must be {kind: 'agent_code', agent_code} or {kind: 'checkpoint', checkpoint: {url}}");
}

/** Take at most one step on a run. Returns true if the run moved. */
export async function advance(env: Env, run: Run): Promise<boolean> {
  switch (run.status) {
    case "funded":
      return planStep(env, run);
    case "planned":
      return dispatchStep(env, run);
    case "evaluated":
      return analyzeStep(env, run);
    default:
      return false; // dispatched/running wait on a runner; planning/analyzing are held by someone
  }
}

/** Advance one run as far as it can go right now. */
export async function drive(env: Env, runId: string): Promise<Run | null> {
  for (let i = 0; i < 6; i++) {
    const run = await getRun(env, runId);
    if (!run || !(await advance(env, run))) return run;
  }
  return getRun(env, runId);
}

/** Cron entry point: release stale holds, then push every runnable run. */
export async function tick(env: Env) {
  await env.DB.prepare(
    `UPDATE runs SET status = CASE status WHEN 'planning' THEN 'funded' ELSE 'evaluated' END, updated_at = datetime('now')
     WHERE status IN ('planning', 'analyzing') AND updated_at < datetime('now', ?)`,
  )
    .bind(`-${STALE_MINUTES} minutes`)
    .run();
  const runs = await runsInStatus(env, ["funded", "planned", "evaluated"]);
  const moved: { id: string; status: string }[] = [];
  for (const r of runs) {
    const after = await drive(env, r.id);
    if (after && after.status !== r.status) moved.push({ id: r.id, status: after.status });
  }
  return { considered: runs.length, moved };
}

/**
 * Which leaderboard a result belongs on. A runner that overrode the game count
 * produced a number that isn't comparable with the protocol's rows, so it gets
 * its own board.
 */
function boardProtocol(spec: RunSpec, result: RunResult | null): string {
  return result?.status === "ok" && result.protocol?.overridden
    ? `${spec.protocol}@${result.protocol.games_per_perm}g`
    : spec.protocol;
}

async function planContext(env: Env, run: Run, board?: string) {
  const project = await getProjectById(env, run.project_id);
  const history = (await listRuns(env, project.id, 20)).filter((r) => r.id !== run.id && r.title);
  const donorNote = run.donation_id
    ? await env.DB.prepare("SELECT note FROM donations WHERE id = ?").bind(run.donation_id).first<string>("note")
    : null;
  const ordinal =
    (await env.DB.prepare("SELECT COUNT(*) AS n FROM runs WHERE project_id = ? AND rowid < (SELECT rowid FROM runs WHERE id = ?)")
      .bind(project.id, run.id)
      .first<number>("n")) ?? 0;
  return { project, history, ordinal, donorNote, leaderboard: await leaderboard(env, project, board ?? project.protocol) };
}

async function failOrRetry(env: Env, run: Run, from: Run["status"], back: Run["status"], err: unknown) {
  const msg = err instanceof Error ? err.message : String(err);
  await logEvent(env, run.id, "error", msg);
  const attempts = run.attempts + 1;
  if (attempts >= MAX_ATTEMPTS) {
    await transition(env, run.id, from, "failed", { attempts, error: msg });
  } else {
    await transition(env, run.id, from, back, { attempts });
  }
}

async function planStep(env: Env, run: Run): Promise<boolean> {
  if (!(await transition(env, run.id, "funded", "planning"))) return false;
  try {
    const p = await plan(env, await planContext(env, run));
    await logEvent(env, run.id, "plan", p.title);
    return transition(env, run.id, "planning", "planned", {
      spec_json: JSON.stringify(p.spec),
      title: p.title,
      hypothesis: p.hypothesis,
    });
  } catch (err) {
    await failOrRetry(env, run, "planning", "funded", err);
    return true;
  }
}

async function dispatchStep(env: Env, run: Run): Promise<boolean> {
  const spec = JSON.parse(run.spec_json ?? "null") as RunSpec | null;
  if (!spec) return transition(env, run.id, "planned", "failed", { error: "planned run has no spec" });
  // Claim first so two drivers can't dispatch twice.
  if (!(await transition(env, run.id, "planned", "dispatched", { dispatcher: env.DISPATCHER }))) return false;
  try {
    const out = await dispatch(env, run, spec);
    if (out.external_ref) {
      await env.DB.prepare("UPDATE runs SET external_ref = ? WHERE id = ?").bind(out.external_ref, run.id).run();
    }
    await logEvent(env, run.id, "dispatch", `${env.DISPATCHER}${out.external_ref ? ` ${out.external_ref}` : ""}`);
    if (out.result) await recordResult(env, run.id, out.result);
    return true;
  } catch (err) {
    await failOrRetry(env, run, "dispatched", "planned", err);
    return true;
  }
}

/** Store a runner's result and put a successful score on the leaderboard. */
export async function recordResult(env: Env, runId: string, result: RunResult): Promise<Run> {
  const run = await getRun(env, runId);
  if (!run) throw new HttpError(404, "no such run");
  if (run.status !== "dispatched" && run.status !== "running") {
    throw new HttpError(409, `run is ${run.status}; results are accepted only while dispatched or running`);
  }
  const spec = JSON.parse(run.spec_json ?? "{}") as RunSpec;
  const ok = result.status === "ok" && typeof result.metric?.value === "number";
  if (ok && result.protocol?.id !== spec.protocol) {
    throw new HttpError(400, `result protocol ${result.protocol?.id} does not match the run's ${spec.protocol}`);
  }
  const protocol = boardProtocol(spec, result);
  const moved = await transition(env, runId, ["dispatched", "running"], "evaluated", {
    result_json: JSON.stringify(result),
    metric_value: ok ? result.metric!.value : null,
    win_rate: ok ? result.win_rate ?? null : null,
    simulated: result.simulated ? 1 : 0,
    error: ok ? null : result.error ?? "runner reported an error",
  });
  if (!moved) throw new HttpError(409, "run changed state while recording the result");
  if (ok) {
    await env.DB.prepare(
      `INSERT INTO leaderboard (project_id, protocol, name, metric_value, win_rate, source, run_id, simulated)
       VALUES (?, ?, ?, ?, ?, 'run', ?, ?)`,
    )
      .bind(run.project_id, protocol, run.title ?? run.id, result.metric!.value, result.win_rate ?? null, runId, result.simulated ? 1 : 0)
      .run();
  }
  await logEvent(env, runId, "result", ok ? `${result.metric!.name}=${result.metric!.value}` : `error: ${result.error}`);
  return (await getRun(env, runId))!;
}

async function analyzeStep(env: Env, run: Run): Promise<boolean> {
  if (!(await transition(env, run.id, "evaluated", "analyzing"))) return false;
  const result = JSON.parse(run.result_json ?? "{}") as RunResult;
  try {
    const board = boardProtocol(JSON.parse(run.spec_json ?? "{}") as RunSpec, result);
    const report = await analyze(env, { ...(await planContext(env, run, board)), run, result });
    await logEvent(env, run.id, "report", report.summary);
    return transition(env, run.id, "analyzing", result.status === "ok" ? "reported" : "failed", {
      report_md: report.markdown,
    });
  } catch (err) {
    await failOrRetry(env, run, "analyzing", "evaluated", err);
    return true;
  }
}

export class HttpError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}
